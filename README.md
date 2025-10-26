[README.md](https://github.com/user-attachments/files/22204351/README.md)
# 🌟 FrameVision V1.0.8
---
**New in v1.0.8**
---
🧩 Features

- Upgraded Qwen VL from V2 to V3, improving chatbot, describer, and prompt enhancer performance.
- The whole app now uses the fancy batch-to-queue popup
- System monitor now (also) checks txt2img & background models
- Prompt Enhancer now has 30+ presets with category filter and favorite selection (favorites show on top). Also added a randomizer for both start prompt and generated prompts.
- Added more presets in Describer for better prompt generation in SDXL, + some txt-to-img and txt/img-to-video prompt presets.

🎨 UI Improvements
- Moved arrows to move between tabs to the front
- small changes in music player overlay to look better in other dpi settings
- Rearranged the Background Remover / Inpainter
- Cropping tool now has separated options for images and video
- Negatives get their own box in the Prompt Enhancer
- Improved some of the easter eggs
- Settings cleanup: removed non-working toggles for intro, added new toggle for probe (logs) on/off, and replaced Sky Light theme with two new grey themes

🐞 Bugfixes
- Fixed 2-line info in media player.
- Cropper fixed so it no longer is able to set crop bigger than original resolution
- changed some settings in the visuals, now tries to preload the next visual so it doesn't jump/don't work first seconds when it starts. (needs more time for finetuning but at least i finally found what caused the problem)
- Fixed recents and favorites in the File menu; added right-click option to remove only one item from the list
- Various fixes to the Background/Inpainter tool, finally starts to work like it is supposed to (easy way to use is to use one of my presets for background removal or inpaint to start, next you can change prompt etc 
- Manual update check fixed, auto update disabled so use manual update for latest file on github
- Fixed the installer not properly installing python on path (when needed) + some small fix for always finding ffmpeg etc in every tool if users don't have this on path 

---
**New in v1.0.7**
---
New Features:
---
- New easter egg system (use the app to unlock fun stuff)
- Music Visuals now fade into next visual for a nonstop visual experience
- Removed the most boring visuals from the music player and replaced them with new visuals (total: 130)
- Music player got playlist open and save feature,
- Right-click on music tracks opens a new side menu with options
- New models for upscaler
- 5 new themes — 2 day themes (Sky & Pastel), 2 colorful themes (Rainbow Riot & Candy Pop), and one night theme (Graphite Dusk)
- Manual 'Update Check' button in the Info menu
---
Bugfixes:
---
- Fixed a bug in music visuals where it stopped reacting to music after 1 track
- Fixed a bug where it would default to 25fps while upscaling
- Txt-to-image no longer overwrites previous image when the same name is given by the user
- Fixed preview in SoundLab
- Menu hover background fix
- Fixed menus in Txt-to-Image that needed two clicks to open after restarting the app
- Music player fine-tuning for visuals
- Music player new pre-analyzer
- Music player now plays long music tracks (e.g., DJ sets) without CPU overloading and app slowdowns
- Music player: fixed the 'bag' for random visuals to show all before starting a new random order list
- Circle brush in background remover is now more responsive and fades out after use
- Full screen now also hides player controls when keyboard or mouse is not used for 6 seconds
---
**New in v1.0.6:** 
---
- Updater (the only place in the app that needs an internet connection)
App now has an option in File menu/Info/update.
Replace all files or helper files only with the files from latest release.
Or you can also update all files/only helper files to the newest files on github to be able to test newest things.
Has an auto check for new updates or new release (daily, can be turned off)
comes with backup function and user can also select (in the update to newest/beta files) select which files to update.

- Background remove/replace tool (Beta)
  First version of a background/object/inpaint tool
  2 models for background removal (fast or quality)
  sd15 loader for stable diffusion inpaint models, comes with default model, User can download other
- Txt-to-image now also supports a queue; batch runs (>1) always use the queue, user can still select queue on/off for creating 1 image
- New upscale model: RealSR (2x/4x)
---
***most important Bugfixes***

- Prompt generator no longer adds negatives unless requested
- MP3 & large media load without needing to “bump” the first file
- Upscaler “Recents” fixed
- Non-working model disabled in the RIFE tab
- Fixed txt-to-image randomly exiting
- Fixed small bug in temp/cache cleanup
- Fixed issue where the upscaler switched between sound options and video quality options
---
**New in v1.0.5:** 
- Full blown music visualizer (100+ visuals) not changing by time but by **BEATS**.
Loading an mp3 or music track will open a new overlay with a playlist and options for visuals where you can set it as low as 8 beats to have non stop changing visuals that change with sound and beats (and some extra tuning to change after a big change in volume in the track) It comes with a little bug where it will skip first 0.1 seconds of very first track that is loaded (workaround to fix another bug lol), after that it works perfect. Sometimes you notice when there is music in first 100ms, sometimes you don't.
- better last results in upscaler and interpollator
- queue now has preview thumbnails
- some batch jobs got a more fancy loader (to add more, delete etc before starting in queue.)
- added a more overlays for the intro (rain, fireflies, starfields, comets, aurora)
---
**New in V1.0.4:**
- TXT 2 IMG with Model selector and Lora use for SD15 & SDXL
  download your favorite model and put it in the models folder to add it, comes with Juggernaut XL as a good
  allround model
  10+ visual presets and about 20 resolution presets (or manual selection)
  I tried Qwen first couldn't make it work so this was 2nd choice, still works great and best of all, this works
  100% offline and is portable even after install. also installed a Prompt creator/enhancer in tools.
- New mini chat bot with q&a for the app and 300 jokes etc. Ask Framie can also look at the screen or create
  screenshots of anything and describe it
----
**New in V1.0.3 :**
New Tools : Multi rename/replace, select file, folder and rename/replace letters with every detail possible
            Meme/thumbnail creator with font selector and effects for text
            updated version of the cropper with resizable preview and user friendly selection

---
**New in V1.0.2 :**
Big update for queue, Now works everywhere in the app with every tool, ETA, auto play last results, move, delete from queue,

...
**New in V1.0.1 :**
Qwen 2.5VL describe anything it can see, with options for short or long descriptions,
now you can add an image, let it describe and create similar images with ease

---
**All-in-one Video & Photo Upscaler/Editor/Creator**  
✨ *Upscale, convert, edit, describe, and create — in one streamlined app.*
- WORK IN PROGRESS, features can come and go

---
## Quickstart

```bat
:: open a console in the project root
.\.venv\Scripts\activate 2>nul || py -3.10 -m venv .venv && .\.venv\Scripts\activate
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
- **Sytem check:** check diskspace, Cpu/Gpu, basic system info, python, venv etc and gives advice on what to use in the installer
                   When not found it installs correct python (on path) and venv environment
- **Core Install:** App only (minimal). also does simple repairs
- **Full CPU Install:** Torch CPU + limited edition models/binaries.
- **Full CUDA Install:** Torch CUDA + models/binaries 
  
---

## 💡 User-Friendly
- Easy installer with **requirements check**.
- Fully **automated** setup paths (Core, Full CPU, Full CUDA). User can drink coffee, script will deliver out of the box use
- easy to use **tabs**, many things in **collapsible** boxes so you see only what you need.
- Helpful **tooltips** everywhere. mini chatbot, **Q&A** and knowledge base (info menu)
- Resizable layout with draggable splitters. Fit the player or the tabs to the size **you** want.
- Clear messages and graceful fallbacks (e.g. CPU fallback when GPU is unavailable, streaming for low mem use etc).

---

## 🌐 Platforms
- **Windows**: Primary target
- **Linux / WSL2**: Supported (experimental)

---

## 🖥 Workspace
- **Left:** Multi Media Player
  - easy slider for fast forward
  - default buttons play pause stop
  - Multi format (img/sound/video)
  - Music playlist & visualizer (100+)
  - Info button with popup window
  - Volume with mute & equalizer
  - Compare side by side with pan & 25x zoom (work in progress, html version for now)
  - Ask Framie (little offline chatbot with screenshot describer etc)
  - Fullscreen on off (double click and esc shortcuts)
  - Upscale button that works from everywhere in the app (roadmap has other plans)
  - scroll to zoom in/out 50x on upscaled images + panning 
  
- **Right:** Modular tabs:
  - Multi model Upscaler
  - RIFE interpollator
  - Describer
  - TXT2IMG + Model loader SD15/SDXL
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
- preview last 15 upscales, batch, queue
- change video codec,bitrate, sound quality etc
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
- Supports both CPU and GPU,
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

### 🖌️ TXT → IMG (SD15+SDXL)
- no cloud requiredn full offline once app is installed
- **queue don't work properly yet so don't use more then 10 umages in batch yet**
- Prompt & negative prompt support with quick style presets.
- Seed policies: Fixed / Random / Increment.
- Mutiple presets
- Lora support
- Multi model selector
  Comes with juggernaut XL + model selector
  

---

### 🧰 Multi Tools
- Change speed for videos (with slow motiion up to 0.15 speed & audio sync).
- Resize with aspect-ratio preservation.
- Prompt enhancer (uses Qwenvl2.5)
- Sound Lab (add/mix multiple sound sources with video)
- Advanced Gif/animation creator 
- Extract frames to images. (last/all)
- Trim videos with advanced preview & crop selection
- Cropping tool 
- Quality/Size/convert for video & images.
- Multi rename/replace tools
...

### ⚙️ Settings
- random intro with easter egg (click the intro 4x) default download has 3 intros (day evening and night)
  - optional download has over 100 (roadmap feature)
  - user can add their own backgrounds and app will start with their own random backgrounds
  - overlay matrix rain and other animations (1 or random)
- Theme system:15 themes, Follow time of day, random at every start
  - Day 🌞, Evening 🌆, Night 🌙
  - Cyberpunk, Neon, Ocean, Solarized Light, CRTn Mardi grass, Tropical and many more
  - 3 day themes, 10 dark themes, 2 colorful themes 
- Bug reporting and maintenance tools.
  - User can send email directly from the app with bug report
    (needs email software installed to send an email)
  - multiple logging options with 'dump to cmd & \log\' button
- cleanup (clear cache, temp folder etc)
- Temperature unit toggle °C / °F.
- System Monitor with checks for models etc
  comes with second little colorized hud (anything above above 90% changes colors, gpu above 60celcius changes 
  colors)
  mini hud shows up everywhere. (can be on/off)
- Advanced File Menu with converter + save video as mp3, screenshot, open multi format, last 10, favorites,...
- Info menu with Html Features list & extensive Q&A + knowledge base
---

### 📋 Queue
  - ETA tracking / time done / time added / time finished
  - open/delete finished jobs, open folder of finished job
  - thumbnails for pending/running.finished jobs
  - auto play last result toggle
  - remove jobs from queue
  - move running jobs to failed
  - move running job back to pending
  - live 'led' shows status of the queue


## 🗺 Roadmap
Planned features for future updates:
- **Built-in Compare Panel (A/B):**, not the html version that i use now. Side-by-side, swipe, zoom/pan, and much 
    more.
- Global & per tab save settings after (re)start.
- **Installer upgrades:** Unified progress bars and resume support. app might get it's own windows installer one 
  day
- **Better integration with queue for the TXT2IMG models:** 
  At the moment it is not using the queue until i have more time for this
- zip with over **100 intros**
- Global action buttons (a button action changes depending on the tab you are
- Multi job where user can select more then one tool and do them all at once (for example upscale->rife->add sound...)
- polishing and bugfixing for the rest of my life i think
- a low weight txt/img to video would be nice but let's get the rest working 100% first
  (many failed attempts so far)
- **Easter eggs** for extra fun surprises! Don't forget to click the intro 4x before it ends ;-)

---

## 🖼 Screenshots 

<p align="center">
  <img src="https://github.com/user-attachments/assets/56a1c424-b723-4f12-9837-e3c21507005e"
       alt="FrameVision Screenshots (animated preview)"
       width="800">
</p>

---
---

## 🔄 How to Update FrameVision

Keeping FrameVision up to date is **super easy** — 
Inside the app go to the file menu that says 'info', in there you find update with 4 option :
Stable : updates to the latest release, option for full update or puthon files only
Beta : updates to the most recent files on the github page, these files can change every day and bring test features but also bugs and unfinished work. Also has option for full update or python file only
---


### 🟢 `update_all.bat`
**Safe & Non-Destructive**  
If you can't get in the app anymore there is also an update_all.bat file, delete folder \helpers\ and if needed also the .venv directory before running this.
**Best for:**  
- app don't start and option 2 from installer (quick fixer) didn't work either, this sets you back to official release.

---

### 🔴 Hard Update — `RESET_ALL.bat`
**Strict Sync (Advanced)**  
- ✅ Adds **new files**  
- ✅ Updates **changed files**  
- ⚠️ **Restores deleted files**  
- ⚠️ **Removes all local files** 
**Best for:**  
- Resetting to a **clean, exact copy** of the latest GitHub version.  

---

## 🤝 Contributing
Pull requests and feature suggestions are welcome!  
If you encounter bugs, please open an issue or use the **Bug Report** button in the app's Settings tab.

---

## 📜 License
[MIT License](LICENSE)

---

> Start Project : August 7th 2025
> Built by **Contrinsan (KoonGrizzly)**  
> Feature development & bugfixing help by **ChatGPT 5**  
> *This project is a living work and features may come or go.*


