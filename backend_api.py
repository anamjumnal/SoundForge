from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import uuid
import threading
import traceback
import json
import os
import gc

import numpy as np
import scipy.io.wavfile
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration


# ============================================================
# SOUNDFORGE - LIGHTWEIGHT VERSION (Tiny Model)
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
# JOB STATE
# ============================================================

generation_jobs = {}
generation_jobs_lock = threading.Lock()
model_lock = threading.Lock()


# ============================================================
# MUSICGEN TINY MODEL (LIGHTWEIGHT!)
# ============================================================

MODEL_NAME = "facebook/musicgen-tiny"  # Much smaller than musicgen-small!
processor = None
model = None
model_loading_status = {"loaded": False, "loading": False, "error": None}


def load_musicgen():
    global processor, model

    if processor is not None and model is not None:
        return

    print()
    print("=" * 70)
    print("SOUNDFORGE - LOADING LIGHTWEIGHT MUSICGEN")
    print("=" * 70)
    print("Model:", MODEL_NAME)
    print("This is the tiny version (much smaller!)")
    print("=" * 70)

    try:
        # Load with memory optimization
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        # Load model with reduced precision to save memory
        model = MusicgenForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16  # Use half precision
        )

        # Use CPU only
        model.to("cpu")
        model.eval()

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        print("=" * 70)
        print("MUSICGEN TINY LOADED SUCCESSFULLY")
        print("=" * 70)
        print()

    except Exception as e:
        traceback.print_exc()
        raise e


def load_musicgen_background():
    """Background loading so Flask can bind to port immediately"""
    model_loading_status["loading"] = True
    try:
        load_musicgen()
        model_loading_status["loaded"] = True
        model_loading_status["error"] = None
    except Exception as e:
        traceback.print_exc()
        model_loading_status["error"] = str(e)
    finally:
        model_loading_status["loading"] = False


# ============================================================
# LIBRARY
# ============================================================

def load_library():
    if not LIBRARY_FILE.exists():
        return []
    try:
        with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        traceback.print_exc()
    return []


def save_library(tracks):
    try:
        with open(LIBRARY_FILE, "w", encoding="utf-8") as f:
            json.dump(tracks, f, indent=4, ensure_ascii=False)
        return True
    except Exception:
        traceback.print_exc()
        return False


# ============================================================
# FILE SEARCH
# ============================================================

def find_file_by_name(filename):
    filename = Path(filename).name
    direct = BASE_DIR / filename
    if direct.is_file():
        return direct
    try:
        for candidate in BASE_DIR.rglob(filename):
            if candidate.is_file():
                return candidate
    except Exception:
        traceback.print_exc()
    return None


# ============================================================
# AUDIO HELPERS
# ============================================================

def normalize_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.95
    return audio.astype(np.float32)


# ============================================================
# GENERATE MUSIC (LIGHTWEIGHT)
# ============================================================

def generate_music_simple(prompt, genre, duration_seconds, temperature):
    """Simplified generation for tiny model"""
    duration_seconds = max(5.0, min(float(duration_seconds), 30.0))

    full_prompt = (
        f"{genre} music. "
        f"{prompt}. "
        f"No vocals."
    )

    print(f"Generating {duration_seconds:.1f}s of {genre} music...")

    # Process input
    inputs = processor(
        text=[full_prompt],
        padding=True,
        return_tensors="pt"
    )

    # Reduce tokens for tiny model (it's slower)
    max_new_tokens = int(duration_seconds * 25) + 30

    print(f"Generating with {max_new_tokens} tokens...")

    # Generate with reduced precision
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            guidance_scale=1.0,  # Reduced guidance
            max_new_tokens=max_new_tokens
        )

    # Convert to numpy
    audio = audio_values[0].cpu().numpy()
    audio = np.squeeze(audio)

    if audio.ndim > 1:
        audio = audio[0]

    return audio.astype(np.float32)


# ============================================================
# BACKGROUND GENERATION JOB
# ============================================================

def run_generation_job(job_id, prompt, genre, creativity, requested_duration):
    try:
        with generation_jobs_lock:
            generation_jobs[job_id] = {
                "status": "generating",
                "message": "AI is generating your music.",
                "track": None
            }

        # Cap duration for lightweight model
        requested_duration = min(requested_duration, 30)

        with model_lock:
            audio = generate_music_simple(
                prompt,
                genre,
                requested_duration,
                creativity
            )

        # Normalize
        audio = normalize_audio(audio)
        sample_rate = int(model.config.audio_encoder.sampling_rate)

        # Save
        filename = f"ai_generated_{job_id}.wav"
        output_path = GENERATED_DIR / filename

        scipy.io.wavfile.write(str(output_path), sample_rate, audio)

        if not output_path.exists():
            raise RuntimeError("Generated WAV file was not created.")

        actual_duration = len(audio) / sample_rate
        minutes = int(actual_duration // 60)
        seconds = int(actual_duration % 60)
        duration_text = f"{minutes}:{seconds:02d}"

        audio_url = f"/generated/{filename}"

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

        library = load_library()
        library.insert(0, track)
        save_library(library)

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
        print("Job ID:", job_id)
        print("File:", output_path)
        print("Duration:", duration_text)
        print("=" * 70)

        # Clean up memory
        gc.collect()

    except Exception as e:
        traceback.print_exc()
        with generation_jobs_lock:
            generation_jobs[job_id] = {
                "status": "failed",
                "message": str(e),
                "track": None
            }


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "SoundForge.html")


@app.route("/style.css")
def css():
    return send_from_directory(BASE_DIR, "style.css")


@app.route("/script.js")
def javascript():
    return send_from_directory(BASE_DIR, "script.js")


@app.route("/audio/<path:filename>")
def original_audio(filename):
    found_file = find_file_by_name(filename)
    if found_file:
        return send_from_directory(
            found_file.parent,
            found_file.name,
            mimetype="audio/wav"
        )
    return jsonify({
        "success": False,
        "error": "Original audio file not found.",
        "filename": filename
    }), 404


@app.route("/generated/<path:filename>")
def generated_audio(filename):
    file_path = GENERATED_DIR / Path(filename).name
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


@app.route("/api/library", methods=["GET"])
def get_library():
    return jsonify({
        "success": True,
        "tracks": load_library()
    })


@app.route("/api/health", methods=["GET"])
def health():
    original_files = []
    for track in ORIGINAL_TRACKS:
        found = find_file_by_name(track["file"])
        original_files.append({
            "file": track["file"],
            "exists": found is not None
        })

    return jsonify({
        "online": True,
        "project": "SoundForge",
        "ai_provider": "Local MusicGen Tiny",
        "model": MODEL_NAME,
        "model_loaded": model_loading_status["loaded"],
        "model_loading": model_loading_status["loading"],
        "model_error": model_loading_status["error"],
        "original_tracks": len(ORIGINAL_TRACKS),
        "original_files": original_files,
        "library_tracks": len(load_library()),
        "duration_support": "5-30 seconds (lightweight)"
    })


@app.route("/api/tracks", methods=["GET"])
def get_tracks():
    tracks = []
    for track in ORIGINAL_TRACKS:
        tracks.append({
            **track,
            "file": "/audio/" + track["file"]
        })
    return jsonify({
        "success": True,
        "tracks": tracks
    })


@app.route("/api/generate", methods=["POST"])
def start_generation():
    try:
        if not model_loading_status["loaded"]:
            return jsonify({
                "success": False,
                "error": "Model is still loading. Please try again shortly."
                         if model_loading_status["loading"]
                         else f"Model failed to load: {model_loading_status['error']}"
            }), 503

        data = request.get_json(silent=True) or {}

        prompt = str(data.get("prompt", "")).strip()
        genre = str(data.get("genre", "")).strip()

        try:
            requested_duration = int(float(data.get("duration", 10)))
        except (TypeError, ValueError):
            requested_duration = 10
        
        requested_duration = max(5, min(requested_duration, 30))

        try:
            creativity = float(data.get("creativity", 0.8))
        except (TypeError, ValueError):
            creativity = 0.8
        
        creativity = max(0.5, min(creativity, 1.5))

        if not prompt:
            return jsonify({
                "success": False,
                "error": "Please enter a music prompt."
            }), 400

        if not genre:
            return jsonify({
                "success": False,
                "error": "Please choose a genre."
            }), 400

        job_id = uuid.uuid4().hex[:10]

        with generation_jobs_lock:
            generation_jobs[job_id] = {
                "status": "queued",
                "message": "Generation queued.",
                "track": None
            }

        worker = threading.Thread(
            target=run_generation_job,
            args=(job_id, prompt, genre, creativity, requested_duration),
            daemon=True
        )
        worker.start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message": "Music generation started."
        }), 202

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/generation-status/<job_id>", methods=["GET"])
def generation_status(job_id):
    with generation_jobs_lock:
        job = generation_jobs.get(job_id)

    if job is None:
        return jsonify({
            "success": False,
            "error": "Generation job not found."
        }), 404

    return jsonify({
        "success": True,
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "message": job.get("message", ""),
        "track": job.get("track")
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # Load model in background
    threading.Thread(target=load_musicgen_background, daemon=True).start()

    print()
    print("=" * 70)
    print("SOUNDFORGE AI MUSIC GENERATOR (LIGHTWEIGHT)")
    print("=" * 70)
    print("Host: 0.0.0.0")
    print("Port:", port)
    print("AI Provider: Local MusicGen Tiny")
    print("Model:", MODEL_NAME)
    print("Memory: Optimized for 512MB (Render free tier)")
    print("Duration: 5-30 seconds (lightweight)")
    print("=" * 70)
    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )
