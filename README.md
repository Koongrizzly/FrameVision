[README.md](https://github.com/user-attachments/files/22204351/README.md)
# 🌟 FrameVision
**All-in-one Video & Photo Upscaler/Editor**  
✨ *Upscale, convert, edit, describe, and create — in one streamlined app.*
BETA, WORK IN PROGRESS, features can come and go
---

## 🖥 Workspace
- **Left:** Media Player
  - easy slider for fast forward
  - default buttons play pause stop
  - Multi format (img/sound/video)
  - Info button with popup window
  - Compare side by side with pan & 25x zoom (work in progress, html version for now)
  - Ratio : fill, fit, full, center
  - Fullscreen on off (double click and esc shortcuts)
  - Upscale button that works from everywhere in the app (roadmap has other plans)
  
- **Right:** Modular tabs:
  - Multii model Upscaler
  - RIFE 
  - Describer/prompt creator
  - TXT2IMG (work in progress, offline)
  - Multi-Tools
  - Advanced Queue
  - Settings
---


## 🚀 Features

### 🧪 Upscaler
- Real-ESRGAN family & friends — **auto-downloads**, lean models.
- **Models included:**
  - **Real-ESRGAN:** `realesrgan-x4plus`, `realesrgan-x4plus-anime`,
    `realesr-general-x4v3`, `realesr-general-wdn-x4v3`
  - **Waifu2x:** `models-cunet`, `models-upconv_7_photo`, `models-upconv_7_anime_style_art_rgb`
  - **Upscayl:** `digital-art-4x`, `high-fidelity-4x`, `remacri-4x`,
    `ultramix-balanced-4x`, `ultrasharp-4x`, `upscayl-lite-4x`, `upscayl-standard-4x`
- preview last 10 upscales, live progress bar with ETA
- change video codec,bitrate, sound quality etc
- advanced settings for some models
- works both on GPU or CPU

---

### 🖼️ Thumbnail / Meme Creator
- Capture the current frame from a video or load any image.
- Multiple text layers: fonts, outline width, shadows, ALL CAPS.
- Rotate, tilt/skew, glow, letter spacing and **arc curvature** controls.
- **Interactive crop overlay** (16:9, 9:16, or free crop).
- Export to PNG/JPG, with or without EXIF data.

---

### ⏱️ RIFE Interpolation
- One-click **frame interpolation** 
- show last results preview pane
- multiple versions and models (from v4 to UHD)
- Slow-motion **0.15×–1.0×**, or smooth FPS boosts up to **4.0×**.
- Supports both CPU and GPU, GPU is superfast
- With live progress bar and ETA

---

### 📝 Describe (AI)
- **Engine:** Qwen2-VL (2B Instruct)
- Generate rich captions, OCR-style text, and multi-turn Q&A.
- Describe files, folders, objects, current video frame with adjustable details
- Promtify included
- Export results as `.txt` or `.json`.
- Fine-tune behavior with advanced parameters:
  - Negative prompts
  - Temperature, top-p/k
  - Token limits, no-repeat rules

---

### 🖌️ TXT → IMG (Qwen)
WORK IN PROGRESS CURRENTLY OFFLINE
- no cloud required.
- Prompt & negative prompt support with quick style presets.
- Seed policies: Fixed / Random / Increment.
- Style Builder: control **Art style**, **Shape**, and more.
- Direct preview in player, queue support included.
  
---

### 🧰 Multi Tools
- Change speed for videos (with audio sync).
- Resize with aspect-ratio preservation.
- Advanced Gif creator with lot's of saving options
- Extract frames to images. (one/last/all)
- **Trim videos** with advanced preview.
- Crop video 
- Quality/Size/convert for video & images:
- Add or replace audio tracks in video.
- Multi rename tools
...
---

### ⚙️ Settings
- random intro with easter egg (click the intro 4x) default download has 3 intros (day evening and night)
  - optional download has over 100 (roadmap feature)
  - user can add their own backgrounds and app will start with their own random backgrounds
  - overlay matrix rain and other animations (roadmap feature)
- Theme system:10+ themes, Follow time of day, random at every start
  - Day 🌞, Evening 🌆, Night 🌙
  - Cyberpunk, Neon, Ocean, Solarized Light, CRTn Mardi grass, Tropical and many more
  - 3 day themes, 10 dark themes, 2 colorful themes 
- Bug reporting and maintenance tools.
  - User can send email directly from the app with bug report
  - multiple logging options with 'dump to cmd' button
- cleanup (clear cache, temp folder etc)
- Temperature unit toggle °C / °F.
- System Monitor with second little colorized hud that shows up everywhere. (can be on/off)
- Advanced File Menu with converter + save video as mp3, screenshot, open multi format, last 10, favorites,....
- Info menu with Html Features list & extensive Q&A + knowledge base
---

### 📋 Queue
  - Reorder jobs
  - ETA tracking / time done / time added /time finished
  - open finished jobs
  - remove jobs from queue,
  - remove non working jobs,
  - move running job to failed
  - move running job back to pensing
  - live 'led' with colors show status of the queue

---

## 🛠 Installation
Run **`start.bat`** — it automatically:
1. Opens the installer menu if setup is incomplete.
2. Repairs missing components (like `psutil`).
3. Starts the background worker.

### Installer Menu Options
- **Core Install:** App only (minimal).
- **Full CPU Install:** Torch CPU + models/binaries.This installer will not ask to install Qwen, smaller model is on roadmap
- **Full CUDA Install:** Torch CUDA + models/binaries.+ asks to install Qwen TXT 2 IMG (only allows install from 10 gig vram and up)

---

## 💡 User-Friendly
- Easy installer with **requirements check**.
- Fully automated setup paths (Core, Full CPU, Full CUDA).
- Helpful tooltips everywhere.
- Resizable layout with draggable splitters.
- Clear messages and graceful fallbacks (e.g., CPU fallback when GPU is unavailable).

---

## 🗺 Roadmap
Planned features for future updates:
- **Built-in Compare Panel (A/B):** Side-by-side, swipe, overlay, zoom/pan, and composite export.
- **Installer upgrades:** Unified progress bars and resume support.
- **Better integration with queue for the upscalers:** At the moment it is not using the queue until i have more time for this
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

