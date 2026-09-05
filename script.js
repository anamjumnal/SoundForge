// ============================================================
// SOUNDFORGE - FRONTEND JAVASCRIPT
// ============================================================

// ============================================================
// BACKEND API CONFIGURATION
// When deployed on Railway, frontend and backend are on same domain
// So we use relative paths (empty API_BASE)
// ============================================================

const API_BASE = "";

// ============================================================
// ORIGINAL 4 TRACKS
// ============================================================

const originalTracks = [
    {
        id: 1,
        title: "Generated Track 01",
        genre: "Classical",
        icon: "🎹",
        file: "/audio/generated_music_1_loud.wav",
        duration: "3:45"
    },
    {
        id: 2,
        title: "Generated Track 02",
        genre: "Classical",
        icon: "🎻",
        file: "/audio/generated_music_2_loud.wav",
        duration: "3:52"
    },
    {
        id: 3,
        title: "Generated Track 03",
        genre: "Classical",
        icon: "🎼",
        file: "/audio/generated_music_3_loud.wav",
        duration: "3:58"
    },
    {
        id: 4,
        title: "Generated Track 04",
        genre: "Classical",
        icon: "🎺",
        file: "/audio/generated_music_4_loud.wav",
        duration: "4:02"
    }
];

// ============================================================
// STATE
// ============================================================

let generatedTracks = [];
let currentTrack = null;
let currentOriginalIndex = 0;
let audioPlayer = null;

// ============================================================
// RESOLVE AUDIO URL
// ============================================================

function resolveAudioUrl(filePath) {
    if (!filePath) {
        return "";
    }

    const file = String(filePath).trim();

    if (file.startsWith("http://") || file.startsWith("https://")) {
        return file;
    }

    if (file.startsWith("/")) {
        return file;
    }

    return `/${file}`;
}

// ============================================================
// GET AUDIO PLAYER
// ============================================================

function getAudioPlayer() {
    if (!audioPlayer) {
        audioPlayer = document.getElementById("audioPlayer");
    }
    return audioPlayer;
}

// ============================================================
// PAGE NAVIGATION
// ============================================================

function goToPage(pageName) {
    document.querySelectorAll(".page").forEach(page => {
        page.classList.remove("active-page");
    });

    const selectedPage = document.getElementById(pageName);
    if (selectedPage) {
        selectedPage.classList.add("active-page");
    }

    document.querySelectorAll(".nav-link").forEach(button => {
        button.classList.remove("active");
        const text = button.textContent.trim().toLowerCase();

        if (
            (pageName === "home" && text === "home") ||
            (pageName === "generate" && text === "generate") ||
            (pageName === "tracks" && text === "generated tracks") ||
            (pageName === "library" && text === "my library") ||
            (pageName === "about" && text === "about")
        ) {
            button.classList.add("active");
        }
    });

    window.scrollTo({ top: 0, behavior: "smooth" });
}

// ============================================================
// RENDER ORIGINAL TRACKS
// ============================================================

function renderOriginalTracks() {
    const grid = document.getElementById("originalTracksGrid");

    if (!grid) {
        console.error("originalTracksGrid not found.");
        return;
    }

    grid.innerHTML = "";

    originalTracks.forEach((track, index) => {
        const card = document.createElement("div");
        card.className = "track-card";

        card.innerHTML = `
            <div class="track-icon">${track.icon}</div>
            <div class="track-details">
                <h3>${escapeHTML(track.title)}</h3>
                <p>${escapeHTML(track.genre)}</p>
                <small>${escapeHTML(track.duration)}</small>
            </div>
            <div class="track-actions">
                <button type="button" onclick="playOriginalTrack(${index})" title="Play">▶</button>
                <button type="button" onclick="openPlayerForOriginal(${index})" title="Open Player">🎧</button>
            </div>
        `;

        grid.appendChild(card);
    });

    console.log("Original 4 tracks rendered:", originalTracks.length);
}

// ============================================================
// RENDER LIBRARY
// ============================================================

function renderLibrary() {
    const grid = document.getElementById("libraryGrid");

    if (!grid) {
        console.error("libraryGrid not found.");
        return;
    }

    if (!Array.isArray(generatedTracks)) {
        generatedTracks = [];
    }

    if (generatedTracks.length === 0) {
        grid.innerHTML = `
            <div class="library-empty">
                <div>🎧</div>
                <h3>Your library is empty</h3>
                <p>Generate your first AI track and it will appear here.</p>
                <button type="button" class="primary-btn" onclick="goToPage('generate')">
                    Create Your First Track
                </button>
            </div>
        `;
        return;
    }

    grid.innerHTML = "";

    generatedTracks.forEach((track, index) => {
        const card = document.createElement("div");
        card.className = "track-card";

        card.innerHTML = `
            <div class="track-icon">${track.icon || "✨"}</div>
            <div class="track-details">
                <h3>${escapeHTML(track.title || "AI Generated Track")}</h3>
                <p>${escapeHTML(track.genre || "Music")}</p>
                <small>${escapeHTML(track.duration || "AI Generated")}</small>
            </div>
            <div class="track-actions">
                <button type="button" onclick="playGeneratedTrack(${index})" title="Play">▶</button>
                <button type="button" onclick="openGeneratedPlayer(${index})" title="Open Player">🎧</button>
            </div>
        `;

        grid.appendChild(card);
    });
}

// ============================================================
// LOAD LIBRARY FROM SERVER
// ============================================================

async function loadLibraryFromServer() {
    try {
        const response = await fetch(`${API_BASE}/api/library`, {
            method: "GET",
            cache: "no-store"
        });

        if (!response.ok) {
            throw new Error(`Library request failed: ${response.status}`);
        }

        const data = await response.json();

        if (data && data.success === true && Array.isArray(data.tracks)) {
            generatedTracks = data.tracks.map(track => ({
                ...track,
                file: resolveAudioUrl(track.file)
            }));

            renderLibrary();
            console.log("Saved AI library loaded:", generatedTracks.length, "tracks");
        } else {
            generatedTracks = [];
            renderLibrary();
            console.warn("Backend returned an invalid library response.");
        }
    } catch (error) {
        console.error("Library loading error:", error);
        generatedTracks = [];
        renderLibrary();
    }
}

// ============================================================
// PLAY ORIGINAL TRACK
// ============================================================

function playOriginalTrack(index) {
    const track = originalTracks[index];

    if (!track) {
        console.error("Original track not found:", index);
        return;
    }

    const player = getAudioPlayer();

    if (!player) {
        console.error("audioPlayer element not found.");
        return;
    }

    currentOriginalIndex = index;
    currentTrack = track;

    const audioUrl = resolveAudioUrl(track.file);

    console.log("Playing original track:", audioUrl);

    player.pause();
    player.removeAttribute("src");
    player.load();
    player.src = audioUrl;
    player.load();

    updatePlayerInformation(track);

    const playPromise = player.play();

    if (playPromise) {
        playPromise
            .then(() => updatePlayButtons(true))
            .catch(error => {
                console.error("Original track playback error:", error);
                updatePlayButtons(false);
            });
    }
}

// ============================================================
// OPEN ORIGINAL PLAYER
// ============================================================

function openPlayerForOriginal(index) {
    playOriginalTrack(index);
    goToPage("player");
}

// ============================================================
// PLAY GENERATED TRACK
// ============================================================

function playGeneratedTrack(index) {
    const track = generatedTracks[index];

    if (!track) {
        console.error("Generated track not found:", index);
        return;
    }

    if (!track.file) {
        console.error("Generated track has no audio file:", track);
        return;
    }

    const player = getAudioPlayer();

    if (!player) {
        console.error("audioPlayer element not found.");
        return;
    }

    currentTrack = track;

    const audioUrl = resolveAudioUrl(track.file);

    console.log("Playing generated track:", audioUrl);

    player.pause();
    player.removeAttribute("src");
    player.load();
    player.src = audioUrl;
    player.load();

    updatePlayerInformation(track);

    const playPromise = player.play();

    if (playPromise) {
        playPromise
            .then(() => updatePlayButtons(true))
            .catch(error => {
                console.error("Generated track playback error:", error);
                updatePlayButtons(false);
            });
    }
}

// ============================================================
// OPEN GENERATED PLAYER
// ============================================================

function openGeneratedPlayer(index) {
    playGeneratedTrack(index);
    goToPage("player");
}

// ============================================================
// UPDATE PLAYER INFORMATION
// ============================================================

function updatePlayerInformation(track) {
    if (!track) return;

    const playerTitle = document.getElementById("playerTitle");
    const playerGenre = document.getElementById("playerGenre");
    const miniTitle = document.getElementById("miniTitle");
    const miniGenre = document.getElementById("miniGenre");

    if (playerTitle) playerTitle.textContent = track.title || "Unknown Track";
    if (playerGenre) playerGenre.textContent = track.genre || "Music";
    if (miniTitle) miniTitle.textContent = track.title || "Unknown Track";
    if (miniGenre) miniGenre.textContent = track.genre || "Music";
}

// ============================================================
// TOGGLE PLAY / PAUSE
// ============================================================

function togglePlay() {
    const player = getAudioPlayer();

    if (!player || !player.src) {
        return;
    }

    if (player.paused) {
        const playPromise = player.play();

        if (playPromise) {
            playPromise
                .then(() => updatePlayButtons(true))
                .catch(error => console.error("Playback error:", error));
        }
    } else {
        player.pause();
        updatePlayButtons(false);
    }
}

// ============================================================
// UPDATE PLAY BUTTONS
// ============================================================

function updatePlayButtons(isPlaying) {
    const miniPlay = document.getElementById("miniPlay");

    if (miniPlay) {
        miniPlay.textContent = isPlaying ? "⏸" : "▶";
    }
}

// ============================================================
// PREVIOUS TRACK
// ============================================================

function previousTrack() {
    if (originalTracks.length === 0) return;

    currentOriginalIndex--;

    if (currentOriginalIndex < 0) {
        currentOriginalIndex = originalTracks.length - 1;
    }

    playOriginalTrack(currentOriginalIndex);
}

// ============================================================
// NEXT TRACK
// ============================================================

function nextTrack() {
    if (originalTracks.length === 0) return;

    currentOriginalIndex++;

    if (currentOriginalIndex >= originalTracks.length) {
        currentOriginalIndex = 0;
    }

    playOriginalTrack(currentOriginalIndex);
}

// ============================================================
// SETUP AUDIO EVENTS
// ============================================================

function setupAudioEvents() {
    const player = getAudioPlayer();

    if (!player) {
        console.warn("audioPlayer not found during initialization.");
        return;
    }

    player.addEventListener("play", () => updatePlayButtons(true));
    player.addEventListener("pause", () => updatePlayButtons(false));
    player.addEventListener("ended", () => updatePlayButtons(false));
    player.addEventListener("error", () => {
        console.error("Audio element error:", player.error);
        updatePlayButtons(false);
    });
}

// ============================================================
// UPDATE CREATIVITY
// ============================================================

function updateCreativity() {
    const slider = document.getElementById("creativity");
    const value = document.getElementById("creativityValue");

    if (!slider || !value) return;

    value.textContent = slider.value;
}

// ============================================================
// FORMAT DURATION
// ============================================================

function formatDuration(totalSeconds) {
    const seconds = Number(totalSeconds);

    if (!Number.isFinite(seconds) || seconds < 0) {
        return "0:00";
    }

    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);

    return minutes + ":" + String(remainingSeconds).padStart(2, "0");
}

// ============================================================
// UPDATE DURATION
// ============================================================

function updateDuration() {
    const slider = document.getElementById("duration");
    const value = document.getElementById("durationValue");

    if (!slider || !value) return;

    value.textContent = formatDuration(slider.value);
}

// ============================================================
// GENERATE MUSIC
// ============================================================

async function generateMusic() {
    const promptElement = document.getElementById("prompt");
    const genreElement = document.getElementById("genre");
    const durationElement = document.getElementById("duration");
    const creativityElement = document.getElementById("creativity");
    const button = document.querySelector(".generate-btn");
    const buttonText = document.getElementById("generateText");
    const status = document.getElementById("generationStatus");
    const preview = document.getElementById("previewContent");

    if (!promptElement || !genreElement || !durationElement || !creativityElement || !button || !buttonText || !status) {
        console.error("Required generation UI element is missing.");
        return;
    }

    const prompt = promptElement.value.trim();
    const genre = genreElement.value;
    const duration = Number(durationElement.value);
    const creativity = Number(creativityElement.value);

    // ========================================================
    // VALIDATION
    // ========================================================

    if (!prompt) {
        status.textContent = "Please describe the music you want to create.";
        promptElement.focus();
        return;
    }

    if (!Number.isFinite(duration) || duration < 5 || duration > 300) {
        status.textContent = "Please select a duration between 5 seconds and 5 minutes.";
        return;
    }

    // ========================================================
    // START GENERATION
    // ========================================================

    button.disabled = true;
    buttonText.textContent = "⏳ Starting AI...";
    status.textContent = `Starting your ${formatDuration(duration)} ${genre} track...`;

    try {
        // ====================================================
        // START BACKGROUND JOB
        // ====================================================

        const response = await fetch(`${API_BASE}/api/generate`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                prompt: prompt,
                genre: genre,
                duration: duration,
                creativity: creativity
            })
        });

        let startData;

        try {
            startData = await response.json();
        } catch (jsonError) {
            throw new Error(`Server returned an invalid response (${response.status}).`);
        }

        if (!response.ok || !startData || !startData.success) {
            throw new Error(startData?.error || "Music generation failed to start.");
        }

        const jobId = startData.job_id;

        if (!jobId) {
            throw new Error("The server did not return a generation job ID.");
        }

        console.log("Generation job started:", jobId);

        // ====================================================
        // POLL STATUS
        // ====================================================

        let completedTrack = null;
        const maxAttempts = 900;

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            await new Promise(resolve => setTimeout(resolve, 2000));

            const statusResponse = await fetch(`${API_BASE}/api/generation-status/${jobId}`, {
                method: "GET",
                cache: "no-store"
            });

            if (!statusResponse.ok) {
                console.warn("Status request failed:", statusResponse.status);
                continue;
            }

            const statusData = await statusResponse.json();

            // ==================================================
            // QUEUED
            // ==================================================

            if (statusData.status === "queued") {
                buttonText.textContent = "⏳ Queued...";
                status.textContent = "Your music request is queued.";
                continue;
            }

            // ==================================================
            // GENERATING
            // ==================================================

            if (statusData.status === "generating") {
                buttonText.textContent = "⏳ Generating...";
                status.textContent = `AI is creating your ${formatDuration(duration)} track. Please wait...`;
                continue;
            }

            // ==================================================
            // FAILED
            // ==================================================

            if (statusData.status === "failed") {
                throw new Error(statusData.message || "Music generation failed.");
            }

            // ==================================================
            // COMPLETED
            // ==================================================

            if (statusData.status === "completed") {
                completedTrack = statusData.track;
                break;
            }
        }

        if (!completedTrack) {
            throw new Error("Music generation is taking longer than expected.");
        }

        if (!completedTrack.file) {
            throw new Error("The server completed generation but did not return an audio file.");
        }

        // ====================================================
        // CREATE TRACK OBJECT
        // ====================================================

        const newTrack = {
            id: completedTrack.id || Date.now(),
            title: completedTrack.title || "AI Generated Track",
            genre: completedTrack.genre || genre,
            icon: completedTrack.icon || "🎵",
            prompt: completedTrack.prompt || prompt,
            file: resolveAudioUrl(completedTrack.file),
            duration: completedTrack.duration || formatDuration(duration),
            created_at: completedTrack.created_at || new Date().toISOString()
        };

        // ====================================================
        // ADD TO LIBRARY
        // ====================================================

        generatedTracks.unshift(newTrack);
        renderLibrary();

        // ====================================================
        // PREPARE PLAYER
        // ====================================================

        const player = getAudioPlayer();

        if (player) {
            currentTrack = newTrack;
            player.pause();
            player.removeAttribute("src");
            player.load();
            player.src = resolveAudioUrl(newTrack.file);
            player.load();
            updatePlayerInformation(newTrack);
        }

        // ====================================================
        // PREVIEW
        // ====================================================

        if (preview) {
            preview.className = "empty-preview";
            preview.innerHTML = `
                <div class="generated-preview">
                    <div class="big-note">♫</div>
                    <h3>${escapeHTML(newTrack.title)}</h3>
                    <p>
                        ${escapeHTML(newTrack.genre)} •
                        ${escapeHTML(newTrack.duration)} • Added to My Library
                    </p>
                    <button type="button" class="primary-btn" style="margin-top:22px;" onclick="playGeneratedTrack(0)">
                        ▶ Play Generated Track
                    </button>
                </div>
            `;
        }

        // ====================================================
        // SUCCESS
        // ====================================================

        status.textContent = "Music generated successfully and saved to My Library!";
        promptElement.value = "";

        console.log("AI track generated:", newTrack);

    } catch (error) {
        console.error("Generation error:", error);
        status.textContent = error.message || "Unable to generate music.";

    } finally {
        button.disabled = false;
        buttonText.textContent = "✨ Generate Music";
    }
}

// ============================================================
// ESCAPE HTML
// ============================================================

function escapeHTML(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// ============================================================
// INITIALIZE
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {
    audioPlayer = document.getElementById("audioPlayer");

    // Render original tracks
    renderOriginalTracks();

    // Render empty/library state
    renderLibrary();

    // Sliders
    updateCreativity();
    updateDuration();

    // Audio events
    setupAudioEvents();

    // Load saved AI tracks
    await loadLibraryFromServer();

    console.log("SoundForge initialized successfully.");
    console.log("Backend mode: Same-origin Railway");
    console.log("Original tracks:", originalTracks.length);
    console.log("Library tracks:", generatedTracks.length);
});
