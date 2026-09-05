from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import uuid
import threading
import traceback
import json
import os

import numpy as np
import scipy.io.wavfile
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration


# ============================================================
# SOUNDFORGE
# ============================================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {"origins": "*"},
        r"/audio/*": {"origins": "*"},
        r"/generated/*": {"origins": "*"}
    }
)

BASE_DIR = Path(__file__).resolve().parent

GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)

LIBRARY_FILE = BASE_DIR / "library.json"


# ============================================================
# ORIGINAL 4 TRACKS
# THESE ALWAYS STAY SEPARATE FROM MY LIBRARY
# ============================================================

ORIGINAL_TRACKS = [
    {
        "id": 1,
        "title": "Generated Track 01",
        "genre": "Classical",
        "icon": "🎹",
        "file": "generated_music_1_loud.wav",
        "duration": "3:45"
    },
    {
        "id": 2,
        "title": "Generated Track 02",
        "genre": "Classical",
        "icon": "🎻",
        "file": "generated_music_2_loud.wav",
        "duration": "3:52"
    },
    {
        "id": 3,
        "title": "Generated Track 03",
        "genre": "Classical",
        "icon": "🎼",
        "file": "generated_music_3_loud.wav",
        "duration": "3:58"
    },
    {
        "id": 4,
        "title": "Generated Track 04",
        "genre": "Classical",
        "icon": "🎺",
        "file": "generated_music_4_loud.wav",
        "duration": "4:02"
    }
]


# ============================================================
# BACKGROUND GENERATION JOBS
# ============================================================

generation_jobs = {}

generation_jobs_lock = threading.Lock()

# Only one MusicGen generation at a time.
model_lock = threading.Lock()


# ============================================================
# LIBRARY STORAGE
# ============================================================

def load_library():
    if not LIBRARY_FILE.exists():
        return []

    try:
        with open(
            LIBRARY_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except Exception:
        traceback.print_exc()

    return []


def save_library(tracks):
    try:
        with open(
            LIBRARY_FILE,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                tracks,
                f,
                indent=4,
                ensure_ascii=False
            )

        return True

    except Exception:
        traceback.print_exc()
        return False


# ============================================================
# FILE FINDER
# ============================================================

def find_file_by_name(filename):
    """
    Search the deployed project recursively for a filename.

    This handles:
    - root folder
    - audio/
    - generated/
    - nested folders

    and makes the original 4 tracks much more robust.
    """

    filename = Path(filename).name

    direct_path = BASE_DIR / filename

    if direct_path.is_file():
        return direct_path

    try:
        matches = list(
            BASE_DIR.rglob(filename)
        )

        for match in matches:
            if match.is_file():
                return match

    except Exception:
        traceback.print_exc()

    return None


# ============================================================
# MUSICGEN MODEL
# ============================================================

MODEL_NAME = "facebook/musicgen-small"

processor = None
model = None


def load_musicgen():
    global processor, model

    if processor is not None and model is not None:
        return

    print()
    print("=" * 70)
    print("SOUNDFORGE - LOADING MUSICGEN")
    print("=" * 70)
    print("Model:", MODEL_NAME)
    print("First model load can take some time.")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )

    model.to("cpu")
    model.eval()

    print("=" * 70)
    print("MUSICGEN LOADED SUCCESSFULLY")
    print("=" * 70)
    print()


# ============================================================
# GENERATE MUSIC CHUNK
# ============================================================

def generate_chunk(
    prompt,
    temperature,
    duration_seconds
):
    duration_seconds = max(
        1.0,
        min(
            float(duration_seconds),
            30.0
        )
    )

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    )

    max_new_tokens = (
        int(
            np.ceil(
                duration_seconds * 50
            )
        )
        + 60
    )

    print(
        f"Generating MusicGen audio "
        f"for approximately "
        f"{duration_seconds:.2f} seconds..."
    )

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            guidance_scale=3.0,
            max_new_tokens=max_new_tokens
        )

    audio = (
        audio_values[0]
        .cpu()
        .numpy()
    )

    audio = np.squeeze(audio)

    if audio.ndim > 1:
        audio = audio[0]

    return audio.astype(
        np.float32
    )


# ============================================================
# NORMALIZE AUDIO
# ============================================================

def normalize_audio(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) == 0:
        return audio

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:
        audio = (
            audio / peak
        ) * 0.95

    return audio.astype(
        np.float32
    )


# ============================================================
# CROSSFADE AUDIO
# ============================================================

def crossfade_audio(
    first,
    second,
    sampling_rate,
    crossfade_seconds=1.0
):
    crossfade_samples = int(
        sampling_rate
        * crossfade_seconds
    )

    crossfade_samples = min(
        crossfade_samples,
        len(first),
        len(second)
    )

    if crossfade_samples <= 0:
        return np.concatenate(
            [
                first,
                second
            ]
        )

    first_main = first[
        :-crossfade_samples
    ]

    first_tail = first[
        -crossfade_samples:
    ]

    second_head = second[
        :crossfade_samples
    ]

    second_rest = second[
        crossfade_samples:
    ]

    fade_out = np.linspace(
        1.0,
        0.0,
        crossfade_samples,
        dtype=np.float32
    )

    fade_in = np.linspace(
        0.0,
        1.0,
        crossfade_samples,
        dtype=np.float32
    )

    crossfaded = (
        first_tail * fade_out
        +
        second_head * fade_in
    )

    return np.concatenate(
        [
            first_main,
            crossfaded,
            second_rest
        ]
    )


# ============================================================
# CREATE EXACT DURATION TRACK
# ============================================================

def create_track(
    prompt,
    temperature,
    requested_duration
):
    requested_duration = int(
        max(
            5,
            min(
                int(requested_duration),
                300
            )
        )
    )

    sampling_rate = int(
        model.config.audio_encoder.sampling_rate
    )

    target_samples = (
        sampling_rate
        * requested_duration
    )

    print()
    print("=" * 70)
    print(
        f"REQUESTED DURATION: "
        f"{requested_duration} SECONDS"
    )
    print("=" * 70)

    # ========================================================
    # 5-30 SECOND GENERATION
    # ========================================================

    if requested_duration <= 30:

        audio = generate_chunk(
            prompt,
            temperature,
            requested_duration + 1
        )

        if len(audio) < target_samples:

            current_duration = (
                len(audio)
                / sampling_rate
            )

            missing_duration = (
                requested_duration
                - current_duration
            )

            continuation_prompt = (
                prompt
                +
                ". Continue the same musical composition "
                "naturally with variation and development. "
                "Keep the same mood and instrumentation."
            )

            continuation = generate_chunk(
                continuation_prompt,
                temperature,
                min(
                    30,
                    missing_duration + 1
                )
            )

            audio = np.concatenate(
                [
                    audio,
                    continuation
                ]
            )

        audio = audio[
            :target_samples
        ]

        while len(audio) < target_samples:

            missing_samples = (
                target_samples
                - len(audio)
            )

            missing_seconds = (
                missing_samples
                / sampling_rate
            )

            continuation_prompt = (
                prompt
                +
                ". Continue the instrumental composition "
                "naturally and smoothly."
            )

            continuation = generate_chunk(
                continuation_prompt,
                temperature,
                min(
                    30,
                    missing_seconds + 1
                )
            )

            audio = np.concatenate(
                [
                    audio,
                    continuation
                ]
            )

            audio = audio[
                :target_samples
            ]

        print(
            "Final audio duration:",
            f"{len(audio) / sampling_rate:.2f}s"
        )

        return (
            normalize_audio(audio),
            sampling_rate
        )

    # ========================================================
    # 31-300 SECOND GENERATION
    # ========================================================

    chunks = []

    remaining = requested_duration
    chunk_number = 1

    while remaining > 0:

        chunk_duration = min(
            remaining,
            30
        )

        print(
            f"Generating section "
            f"{chunk_number}: "
            f"{chunk_duration} seconds"
        )

        if chunk_number == 1:

            chunk_prompt = prompt

        else:

            chunk_prompt = (
                prompt
                +
                ". Continue the same instrumental "
                "composition naturally. "
                "Develop the melody and arrangement "
                "with variation. "
                "Do not simply repeat the previous section."
            )

        chunk = generate_chunk(
            chunk_prompt,
            temperature,
            chunk_duration + 1
        )

        chunks.append(
            chunk
        )

        remaining -= chunk_duration
        chunk_number += 1

    final_audio = chunks[0]

    for i in range(
        1,
        len(chunks)
    ):

        final_audio = crossfade_audio(
            final_audio,
            chunks[i],
            sampling_rate,
            crossfade_seconds=1.0
        )

    final_audio = final_audio[
        :target_samples
    ]

    while len(final_audio) < target_samples:

        missing_samples = (
            target_samples
            - len(final_audio)
        )

        missing_seconds = (
            missing_samples
            / sampling_rate
        )

        continuation_prompt = (
            prompt
            +
            ". Continue the same instrumental "
            "composition naturally with new musical "
            "development and variation."
        )

        extra = generate_chunk(
            continuation_prompt,
            temperature,
            min(
                30,
                missing_seconds + 1
            )
        )

        final_audio = crossfade_audio(
            final_audio,
            extra,
            sampling_rate,
            crossfade_seconds=1.0
        )

        final_audio = final_audio[
            :target_samples
        ]

    print(
        "Final audio duration:",
        f"{len(final_audio) / sampling_rate:.2f}s"
    )

    return (
        normalize_audio(final_audio),
        sampling_rate
    )


# ============================================================
# BACKGROUND MUSIC GENERATION
# ============================================================

def run_generation_job(
    job_id,
    full_prompt,
    prompt,
    genre,
    creativity,
    requested_duration
):
    try:

        with generation_jobs_lock:
            generation_jobs[job_id] = {
                "status": "generating",
                "message": "Loading AI model and generating music..."
            }

        print()
        print("=" * 70)
        print("BACKGROUND MUSIC GENERATION")
        print("=" * 70)
        print("Job ID:", job_id)

        with model_lock:

            load_musicgen()

            audio, sampling_rate = create_track(
                full_prompt,
                creativity,
                requested_duration
            )

        # ====================================================
        # SAVE UNIQUE WAV
        # ====================================================

        filename = (
            f"ai_generated_{job_id}.wav"
        )

        output_path = (
            GENERATED_DIR
            / filename
        )

        scipy.io.wavfile.write(
            str(output_path),
            sampling_rate,
            audio
        )

        if not output_path.exists():

            raise RuntimeError(
                "Music was generated but the WAV file "
                "could not be created."
            )

        # ====================================================
        # AUDIO URL
        # ====================================================

        audio_url = (
            f"/generated/{filename}"
        )

        actual_duration = (
            len(audio)
            / sampling_rate
        )

        minutes = int(
            actual_duration // 60
        )

        seconds = int(
            actual_duration % 60
        )

        duration_text = (
            f"{minutes}:{seconds:02d}"
        )

        # ====================================================
        # TRACK
        # ====================================================

        track = {
            "id": job_id,
            "title": "AI Generated Track",
            "genre": genre,
            "icon": "🎵",
            "file": audio_url,
            "audio_url": audio_url,
            "duration": duration_text,
            "prompt": prompt,
            "created_at": datetime.now().isoformat()
        }

        # ====================================================
        # SAVE TO LIBRARY
        # ====================================================

        library = load_library()

        library.insert(
            0,
            track
        )

        save_library(
            library
        )

        # ====================================================
        # JOB COMPLETE
        # ====================================================

        with generation_jobs_lock:
            generation_jobs[job_id] = {
                "status": "completed",
                "message": "Music generated successfully.",
                "track": track
            }

        print()
        print("=" * 70)
        print("GENERATION SUCCESSFUL")
        print("=" * 70)
        print("File:", output_path)
        print(
            "Requested:",
            requested_duration,
            "seconds"
        )
        print(
            "Actual:",
            f"{actual_duration:.2f}",
            "seconds"
        )
        print("=" * 70)

    except Exception as e:

        traceback.print_exc()

        with generation_jobs_lock:
            generation_jobs[job_id] = {
                "status": "failed",
                "message": str(e)
            }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        BASE_DIR,
        "SoundForge.html"
    )


# ============================================================
# CSS
# ============================================================

@app.route("/style.css")
def css():

    return send_from_directory(
        BASE_DIR,
        "style.css"
    )


# ============================================================
# JAVASCRIPT
# ============================================================

@app.route("/script.js")
def javascript():

    return send_from_directory(
        BASE_DIR,
        "script.js"
    )


# ============================================================
# ORIGINAL AUDIO
# ROBUST FILE SEARCH
# ============================================================

@app.route("/audio/<path:filename>")
def original_audio(filename):

    found_file = find_file_by_name(
        filename
    )

    if found_file:

        relative_parent = (
            found_file.parent
        )

        relative_name = (
            found_file.name
        )

        return send_from_directory(
            relative_parent,
            relative_name,
            mimetype="audio/wav"
        )

    return jsonify({
        "success": False,
        "error": "Original audio file not found.",
        "filename": filename
    }), 404


# ============================================================
# GENERATED AUDIO
# ============================================================

@app.route("/generated/<path:filename>")
def generated_audio(filename):

    file_path = (
        GENERATED_DIR
        / Path(filename).name
    )

    if not file_path.exists():

        return jsonify({
            "success": False,
            "error": "Generated audio file not found."
        }), 404

    return send_from_directory(
        GENERATED_DIR,
        file_path.name,
        mimetype="audio/wav"
    )


# ============================================================
# GET LIBRARY
# ============================================================

@app.route(
    "/api/library",
    methods=["GET"]
)
def get_library():

    tracks = load_library()
    valid_tracks = []

    for track in tracks:

        filename = Path(
            track.get(
                "file",
                ""
            )
        ).name

        if (
            filename
            and
            (
                GENERATED_DIR
                / filename
            ).exists()
        ):

            valid_tracks.append(
                track
            )

    if len(valid_tracks) != len(tracks):

        save_library(
            valid_tracks
        )

    return jsonify({
        "success": True,
        "tracks": valid_tracks
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    original_file_status = []

    for track in ORIGINAL_TRACKS:

        found = find_file_by_name(
            track["file"]
        )

        original_file_status.append({
            "file": track["file"],
            "exists": found is not None,
            "path": str(found)
            if found
            else None
        })

    return jsonify({

        "online": True,

        "project":
            "SoundForge",

        "model":
            MODEL_NAME,

        "model_loaded":
            model is not None,

        "original_tracks":
            len(ORIGINAL_TRACKS),

        "original_files":
            original_file_status,

        "library_tracks":
            len(load_library()),

        "ai_generation":
            True,

        "duration_support":
            "5-300 seconds"
    })


# ============================================================
# ORIGINAL TRACK API
# ============================================================

@app.route(
    "/api/tracks",
    methods=["GET"]
)
def get_tracks():

    tracks = []

    for track in ORIGINAL_TRACKS:

        tracks.append({
            **track,
            "file":
                "/audio/"
                +
                track["file"]
        })

    return jsonify({
        "success": True,
        "tracks": tracks
    })


# ============================================================
# START AI GENERATION JOB
# ============================================================

@app.route(
    "/api/generate",
    methods=["POST"]
)
def start_generation():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        prompt = str(
            data.get(
                "prompt",
                ""
            )
        ).strip()

        genre = str(
            data.get(
                "genre",
                "Classical"
            )
        ).strip()

        try:

            requested_duration = int(
                float(
                    data.get(
                        "duration",
                        60
                    )
                )
            )

        except (
            TypeError,
            ValueError
        ):

            requested_duration = 60

        requested_duration = max(
            5,
            min(
                requested_duration,
                300
            )
        )

        try:

            creativity = float(
                data.get(
                    "creativity",
                    1.0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            creativity = 1.0

        creativity = max(
            0.5,
            min(
                creativity,
                1.5
            )
        )

        if not prompt:

            return jsonify({
                "success": False,
                "error":
                    "Please enter a music prompt."
            }), 400

        full_prompt = (
            f"{genre} instrumental music. "
            f"{prompt}. "
            f"No vocals. "
            f"High quality musical composition. "
            f"Natural dynamics, musical progression, "
            f"clear melody and coherent arrangement."
        )

        job_id = uuid.uuid4().hex[:10]

        with generation_jobs_lock:

            generation_jobs[job_id] = {
                "status": "queued",
                "message": "Generation queued.",
                "track": None
            }

        worker = threading.Thread(
            target=run_generation_job,
            args=(
                job_id,
                full_prompt,
                prompt,
                genre,
                creativity,
                requested_duration
            ),
            daemon=True
        )

        worker.start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message":
                "Music generation started."
        }), 202

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# GENERATION STATUS
# ============================================================

@app.route(
    "/api/generation-status/<job_id>",
    methods=["GET"]
)
def generation_status(job_id):

    with generation_jobs_lock:

        job = generation_jobs.get(
            job_id
        )

    if job is None:

        return jsonify({
            "success": False,
            "error": "Generation job not found."
        }), 404

    return jsonify({
        "success": True,
        "job_id": job_id,
        "status": job.get(
            "status",
            "unknown"
        ),
        "message": job.get(
            "message",
            ""
        ),
        "track": job.get(
            "track"
        )
    })


# ============================================================
# RAILWAY / LOCAL SERVER STARTUP
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print()
    print("=" * 70)
    print("SOUNDFORGE AI MUSIC GENERATOR")
    print("=" * 70)
    print("Host: 0.0.0.0")
    print("Port:", port)
    print("AI Model:", MODEL_NAME)
    print(
        "Original tracks:",
        len(ORIGINAL_TRACKS)
    )
    print(
        "Library file:",
        LIBRARY_FILE
    )
    print(
        "Generated folder:",
        GENERATED_DIR
    )
    print(
        "Duration support:",
        "5-300 seconds"
    )
    print("=" * 70)
    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )
