# 🌟 FrameVision
**All-in-one Video & Photo Upscaler/Editor**  
✨ *Upscale, edit, describe, and create — in one streamlined app.*

---

## 🚀 Features

### 🧪 Upscaler
- Real-ESRGAN family & friends — **auto-downloads**, lean models.
- Falls back to FFmpeg scaling when models are missing.
- **Models included:**
  - `realesrgan-x4plus`, `realesrgan-x4plus-anime`
  - `realesr-general-x4v3`, `realesr-general-wdn-x4v3`
  - Waifu2x: `models-cunet`, `models-upconv_7_photo`, `models-upconv_7_anime_style_art_rgb`
- Per-job options with full **queue integration**.

---

### 🖼️ Thumbnail / Meme Creator
- Capture the current frame or load any image.
- Multiple text layers: fonts, outline width, shadows, ALL CAPS.
- Rotate, tilt/skew, and **arc curvature** controls.
- **Interactive crop overlay** (16:9, 9:16, or free crop).
- Export to PNG/JPG, with or without EXIF data.

---

### ⏱️ RIFE Interpolation
- One-click **frame interpolation** (2× default), queue-native.
- Slow-motion **0.15×–1.0×**, or smooth FPS boosts up to **4.0×**.
- Superfast mode with Vulkan NCNN backend.
- **Zero setup** — bundled models auto-extract on first run.

---

### 📝 Describe (AI)
- **Engine:** Qwen2-VL (2B Instruct)
- Generate rich captions, OCR-style text, and multi-turn Q&A.
- Describe files, folders, or the current video frame.
- Export results as `.txt` or `.json`.
- Fine-tune behavior with advanced parameters:
  - Negative prompts
  - Temperature, top-p/k
  - Token limits, no-repeat rules

---

### 🖌️ TXT → IMG (Qwen)
- **Local GGUF backend** — no cloud required.
- Prompt & negative prompt support with quick style presets.
- Seed policies: Fixed / Random / Increment.
- Style Builder: control **Art style**, **Shape**, and more.
- Direct preview in player, queue support included.

---

### 🧰 Tools
- Change speed for videos (with audio sync).
- Resize with aspect-ratio preservation.
- Export GIFs using two-pass palette generation.
- Extract frames to images.
- **Trim videos** with precise start/end markers.
- Crop video regions.
- Quality/Size controls for video & images:
  - H.264, H.265, AV1, NVENC
  - JPG, PNG, WEBP
- Add or replace audio tracks on videos.
- Batch tools for large workflows.

---

### ⚙️ Settings
- Paths for models, FFmpeg, and outputs.
- GPU/CPU/DirectML toggles and performance tuning.
- Theme system:
  - Day 🌞, Evening 🌆, Night 🌙, Cyberpunk, Neon, Ocean, Solarized Light, CRT
- Bug reporting and maintenance tools.
- Temperature unit toggle °C / °F.

---

### 📋 Queue
- **Advanced queue system:**
  - Reorder jobs
  - Pause/resume
  - ETA tracking
  - Detailed logs

---

## 🖥 Workspace
- **Left:** Media viewer with playback controls.
- **Right:** Modular tabs:
  - Upscaler
  - RIFE
  - Describe
  - Tools
  - Queue
  - Settings
- Clean, organized cache and binaries inside the app folder.

---

## 🛠 Installation
Run **`start.bat`** — it automatically:
1. Opens the installer menu if setup is incomplete.
2. Repairs missing components (like `psutil`).
3. Starts the background worker.

### Installer Menu Options
- **Core Install:** App only (minimal).
- **Full CPU Install:** Torch CPU + models/binaries.
- **Full CUDA Install:** Torch CUDA + models/binaries.

---

## 📦 Lean Models
- Only essential configs and tokenizers are downloaded.
- No redundant TensorFlow files.
- Caches are neatly stored inside `./.hf_cache`.

---

## 💡 Why FrameVision is User-Friendly
- Easy installer with **requirements check**.
- Fully automated setup paths (Core, Full CPU, Full CUDA).
- Helpful tooltips everywhere.
- Resizable layout with draggable splitters.
- Clear messages and graceful fallbacks (e.g., CPU fallback when GPU is unavailable).

---

## 🗺 Roadmap
Planned features for future updates:
- **Compare Panel (A/B):** Side-by-side, swipe, overlay, zoom/pan, and composite export.
- **Queue polish:** Persistent jobs after restart, better ETA & logs.
- **Installer upgrades:** Unified progress bars and resume support.
- **TXT → IMG improvements:** VRAM-based defaults and refined hi-res helper.
- **Easter eggs** for extra fun surprises!

---

## 🌐 Platforms
- **Windows**: Primary target
- **Linux / WSL2**: Supported (experimental)

---

## 🤝 Contributing
Pull requests and feature suggestions are welcome!  
If you encounter bugs, please open an issue or use the **Bug Report** button in the app's Settings tab.

---

## 📜 License
[MIT License](LICENSE)

---

> Built by **Contrinsan**  
> Feature writing & installer help by **ChatGPT (assistant)**  
> *This project is a living work — details evolve with releases.*

