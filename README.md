<h2 align="center">🖼 Screenshots Preview</h2>

<p align="center">
  <img src="https://github.com/user-attachments/assets/56a1c424-b723-4f12-9837-e3c21507005e"
       alt="FrameVision Screenshots (animated preview)"
       width="800">
</p>

<p align="center"><em>A quick look at the UI and core tools.</em></p>


[README.md](https://github.com/user-attachments/files/22204351/README.md)
# 🌟 FrameVision V1.0.0
**All-in-one Video & Photo Upscaler/Editor**  
✨ *Upscale, convert, edit, describe, and create — in one streamlined app.*
- BETA, WORK IN PROGRESS, features can come and go

---
## Quickstart

```bat
:: open a console in the project root
.\.venv\Scripts\activate 2>nul || py -3.10 -m venv .venv && .\.venv\Scriptsctivate
python -m pip install -U pip
start.bat
```
*Runs the installer menu when needed, repairs missing bits, then launches FrameVision.*

---

## 🛠 Installation
**One file for everything:** start.bat (first time use needs permission to run a bat file on windows pc)
- Run **`start.bat`** — it automatically:
1. Opens the installer menu if setup is incomplete.
2. Repairs missing components (like `psutil`).
3. Starts the background worker.

### Installer Menu Options
- **Sytem check:** check diskspace, Cpu/Gpu, basic system info, python, creates venv etc and gives advice on what to use in the installer
- **Core Install:** App only (minimal).
- **Full CPU Install:** Torch CPU + models/binaries.This installer will not ask to install Qwen, smaller model is on roadmap
- **Full CUDA Install:** Torch CUDA + models/binaries.
  + install Qwen TXT2IMG (only allows install for 12 gig vram and up, download not included yet -> roadmap)

---

## 💡 User-Friendly
- Easy installer with **requirements check**.
- Fully **automated** setup paths (Core, Full CPU, Full CUDA). User can drink coffee, script will deliver out of the box use
- easy to use tabs, everything in collapsible boxes so you see only what you need and without searching
- Helpful **tooltips** everywhere. **Q&A** and knowledge base (info menu)
- Resizable layout with draggable splitters. Fit the player or the tabs to the size you want.
- Clear messages and graceful fallbacks (e.g. CPU fallback when GPU is unavailable, streaming for low mem use etc).

---

## 🔄 How to Update FrameVision

Keeping FrameVision up to date is **super easy** — just use one of the included updater tools!  
Choose the updater based on whether you want to **keep personal files** or **fully reset** to the latest version.
**How to use:**
1. Back up any personal files, edits or custom scripts you don’t want deleted.
2. Place the updaters in the **FrameVision root folder**.
3. Double-click to run.

---

### 🟢 Soft Update — `update_soft.bat`
**Safe & Non-Destructive**  
- ✅ Adds **new files**  
- ✅ Updates **changed files**  
- 🚫 **Never deletes** anything in your folder  
**Best for:**  
- Regular updates when you don’t want to risk losing personal changes or test files.

---

### 🔴 Hard Update — `update_hard.bat`
**Strict Sync (Advanced)**  
- ✅ Adds **new files**  
- ✅ Updates **changed files**  
- ⚠️ **Restores deleted files**  
- ⚠️ **Removes local files** that are not part of the official GitHub repo  
  *(Safe exclusions: `.venv`, `models`, `.hf_cache`, `outputs`, etc.)*
**Best for:**  
- Repairing a broken install.  
- Resetting to a **clean, exact copy** of the latest GitHub version.  
- Preparing a **clean environment** for release or bug reporting.

### 💡 How It Works
Both updaters:
- Try to use **Git** if available (fastest & safest).
- If Git isn’t installed, they **auto-download a ZIP** of the latest version and update from that.
- Writes progress to `update.log`.

---

## 🌐 Platforms
- **Windows**: Primary target
- **Linux / WSL2**: Supported (experimental)

---
---

## 🖥 Workspace
- **Left:** Multi Media Player
  - easy slider for fast forward
  - default buttons play pause stop
  - Multi format (img/sound/video)
  - Info button with popup window
  - Compare side by side with pan & 25x zoom (work in progress, html version for now)
  - Ratio : fill, fit, full, center
  - Fullscreen on off (double click and esc shortcuts)
  - Upscale button that works from everywhere in the app (roadmap has other plans)
  - scroll to zoom in/out & pan (on roadmap)
  
- **Right:** Modular tabs:
  - Multi model Upscaler
  - RIFE interpollator
  - Describer/prompt creator
  - TXT2IMG (work in progress, offline)
  - Multi-Tools
  - Advanced Queue 
  - Settings
---
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
    (needs software installed to send an email)
  - multiple logging options with 'dump to cmd' button
- cleanup (clear cache, temp folder etc)
- Temperature unit toggle °C / °F.
- System Monitor with checks for models etc
  comes with second little colorized hud (anything above above 90% changes colors, gpu above 60celcius vhanges colors)
  mini hud shows up everywhere. (can be on/off)
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
  - live 'led' shows status of the queue


## 🗺 Roadmap
Planned features for future updates:
- **Built-in Compare Panel (A/B):**, not the html version that i use now. Side-by-side, swipe, zoom/pan, and much more.
- Global & per tab save settings after (re)start.
- **Installer upgrades:** Unified progress bars and resume support. app might get it's own windows installer one day
- **Better integration with queue for the upscalers:** At the moment it is not using the queue until i have more time for this
- zip with over **100 intros**
- low weight **txt 2 img** model
- Global action buttons (a button action changes depending on the tab you are
- Multi job where user can select more then one tool and do them all at once (for example upscale->rife->add sound...)
- polishing and bugfixing for the rest of my life i think
- a low weight txt/img to video would be nice but let's get the rest working 100% first
- **Easter eggs** for extra fun surprises! Don't forget to click the intro 4x before it ends ;-)

---

## 🤝 Contributing
Pull requests and feature suggestions are welcome!  
If you encounter bugs, please open an issue or use the **Bug Report** button in the app's Settings tab.

---

## 📜 License
[MIT License](LICENSE)

---

> Built by **Contrinsan (KoonGrizzly)**  
> Feature development & bugfixing help by **ChatGPT 5**  
> *This project is a living work and features may come or go.*


