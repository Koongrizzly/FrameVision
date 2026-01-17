### ✨ FrameVision 2.2 ✨
**All-in-one Sound/Image/Video Tool**  
*Create, Upscale, edit, describe, play — in one streamlined app.*
---
Video Walktrough (most is speedup x2 to show much in less time)

[![Watch the video](https://img.youtube.com/vi/QLIaL4uAFlE/hqdefault.jpg)](https://youtu.be/QLIaL4uAFlE)

---
<details>
  <summary>Januari 17, 2026. V2.2 update adds qwen2511 & 2512</summary>
  New Features

  Qwen 2512 GGUF loader
  - Install from Optional Installs
  - Find it in txt2img with the other txt2img models
  - Added support for Turbo LoRA (and other LoRAs) — get images in only 4 steps
   
  Qwen2511
  - GGUF loader, download extra models via Optional Installs menu
  - Mainly for testing / not fully working like in ComfyUI yet
  - Use reference images + prompt for edits
  - Offload options
  - Flash Attention
  - LoRA support (tested with the multi angle LoRA)
  - 96 angle toggle for easy use of the multi angle LoRA
  
  New Background Remover + SDXL Inpaint (with model loader)
  - Removed the old all in one background remover / (bad) SD15 inpainter
  - Replaced with 2 complete new (separated) tools
  - Background Remover is more user friendly and simplified (recommend 720p+ start images to avoid bad cutouts)
  - Pan/zoom + brush for high detail background removal
  - User can load an image as new background; original cutout/mask can be moved + resized to fit the new background
  - Inpainter now loads SDXL instead of SD15 models (one included in Optional Installs)
  - Easy flow: load image → load mask → enter prompt → done
  - Background remover works on any PC
  - SDXL inpaint works on GPU as low as 6GB VRAM (8GB recommended) — use this if Qwen Edit 2511 is too VRAM hungry
  
  UI Improvements
  - Interpolator: added “interpolate x3” shortcut + manual number input; compression removed so output files now keep source quality
  - License viewer (MIT, Apache, …) added for all tools — find it in Info menu or in Optional Installs
  - 4 new themes: Blue / Orange / Green / Red in the dark
  - Banner: new function — gradient rainbow colors with slider for speed
  - Better defaults for new user installs (Settings + Videoclip Creator)
  
  Bugfixes
  - Fixed Media Explorer bug where it would become slow after image(s) got scanned
  - Fixed banner showing up after restart in Settings
  - Fixed banner not hiding in Videoclip Creator
</details>
<details>
  <summary>Januari 1, 2026. V2.1.2 Happy Newyear and another update</summary>
  
  
- fixed themes freezing app for 5 seconds after applying (added a 'safe apply' button in case freezing comes back)
- fixed trim tool freezing ui while generating preview
- fixed upscaler defaulting to 30fps
- fixed tab integration still thinking upscaler was a main tab (finaly found why upscaler needed restart to apply new upscaler engine/model)
- fixed compare tool crash when loading new media file while repeat button was on.
- fixed compare tool sync problems for video (85% lol)
- added a date/time changer to the metadata editor
- added a couple of simple effects to the text on top of video overlay tool
- added 'strobe on time' in the video clip creator, now you can pick the perfect moment for a strobe effect
- Behind the scenes upgrade : Moved queue to it's own python file, no longer inside the big main framevision python file. This may come with new bugs, i will fix them as i run into them  

</details>
  <details>
  <summary>December 28, 2025. V2.1.1 mini update</summary>

Bugfixes
- fixed optional installs: background model + Z-image fp16 installer
- fixed move up/down in queue (pending) + right click options restored for moving
- fixed shrink feature for repeat button
- fixed adding Music Clip Creator jobs to the queue, queue-ing button restored
- fixed filename for random seeds in Hunyuan
- not 100% fixed yet: queue finished results — if video playback or visuals from music player stutter: clear finished results in queue
---
UI improvements
- more “View Result” buttons (most tools now have it)
- new Settings toggles ;
- animated hover effects on some buttons (6 animations + random)
- Change font size
- moved buttons to front in queue (finished)
- simplified treeview in Media Explorer
- updated Features HTML + a big part of the Knowledge Base in the Info menu

New features
- VideoText (beta test)
  - allows adding text to video
  - choice of font, size, position
  - unlimited text adding
  - adjustable fade in/out
  - preview toggle to always show all text
  - preview timeline with zoom/pan/drag/drop
  - text effects
  - load/play/pause/stop/save
</details>
<details>
  <summary>December 25, 2025 — v2.1 release notes</summary>

**✨ New Features**

- Music Videoclip Creator
  - Added more transitions, cinematic effects, and drop/chorus effects
  - a couple of extra settings such as 'skip Fx in intro'
  - new 1 click preset : 'get it done FAST' (3 minute videoclip in less then a minute, only uses easy to render effects and transitions)
- Reverse video tool
  - Simple tool to fix AI videos that run backwards
  - Supports batch, queue, use currently playing video, and load video
  - With boomerang (reverse-forward-reverse-...) feature
- HunyuanVideo 1.5
  - TXT → video, IMG → video, video → video + video extender
  - Engine selection for 480p & 720p
  - Model choice: normal, distilled, low step (only for 480 img2vid model)
  - Each model downloads ~30GB extra files, 720p model is 50 gigabyte
  - Aspect ratio for all resolutions (192p–720p): 1:1 / 16:9 / 9:16
- Media explorer tab
  - Replaces the buggy “last result” screens from every model,
  - Most tools now have a 'view results' button which will open the folder, scan for json files and show all results in the media explorer.
  - Double click plays them in the app multi mediaplayer, right click give extra options such as :
  - Search/sort/play/preview/cut/copy/paste/delete/rename/favorite+...
  - User can scan any folder + subfolders (or open basic tree view) (Huge drives with video can take longtime)
- Z-Image Turbo GGUF Model loader
  - Get FP16 or a GGuf variant for the Z-image Turbo image creator
  - use them in a model loader, queue allows combining any model
  - Q4–Q8 variants allow fast image creation on virtually any VRAM
  - may give an error for missing dll on some computers, popup will point to windows file for install of visual studio
- Optional downloads
  - a new menu for optional installs (moved from the installer into a new part of the app, so user can decide later what to install
  - Currently includes : Z-image fp16 + gguf model downloader, Hunyuanvideo 1.5, wan2.2 5b, a model for SDXL, background/inpaint models, ace step music and a face restorer These are no longer installed with the app by default and can be skipped/installed in here to save time on first time install of the app

**🎨 UI Improvements**

- Music video creator beat-synced visuals now come with a preview thumbnail, i included ready thumbnail pack (to avoid long wait at first time use)
- Repeat button for video files (shows up when a video file is loaded, hides again when images are loaded, music player already has it's own repeat features)
- Right mouse button in queue for running & pending get a couple of options (cancel running job, move back to pending, delete job from pending, show JSON info)
- Prompt enhancers directly under prompt box in WAN 2.2 / Hunyuan 1.5 & Z-image / SDXL
- Wheel guard: avoids accidentally changing settings while scrolling over something with mouse
- Reorder tabs & tools (upscaling, Rife, Qwen describe, background remover and Ace Music now are in tools tab)
- Settings have a new toggle that allows reordering the main tabs, Tools tab also gets a reorder toggle.
- HUD now also shows internet traffic up/down (when above 50kb/s) to be able to follow 'silent downloads' of some models 

**✅ Bugfixes**

- Z-image
  - Fixed Z-image not staying at 0 CFG after restart
  - Z-image text encoder CPU offload fixed
  - Queue names for Z-image fixed
  - Added all resolutions up to 4K
  - Added LORAs back to the UI (needs testing)
  - Fixed Z-image info in worker
- Music Videoclip Creator Fixed beat-synced visuals “no beatsynced visual” toggle
  - Fixed random multiply and mosaic feature
  - Fixed Mosaic effect error when using less than 9 clips
  - Fixed loaded JSON for edited timelines not being used
  - Finetuned some effects
  - not a fix but a temporarily solution : disabled use of images in Videoclip Creator to resolve bug when creating video with certain effects
- WAN 2.2
  - Fixed WAN 2.2 FPS (now 16–30fps instead of locked 24) and (lower) resolutions — now allows everything between 360p and 720p for much faster generation time
  - Fixed WAN 2.2 offloading — added toggles to move model (much slower) and text encoder to CPU + auto setting when queue is turned off (e.g., when extending video)
- Tools
  - Slowdown/speedup tool now keeps original source bitrate
- Fixed the Compare tool
  — no more temporary HTML file; clicking Compare lets the user load two media files and then zoom/pan/slide to compare
- Queue
  - Fixed “cancel running job” button not working in queue
- Describer
  - Fixed temporary app freeze when clicking “Describe” button in Describer
- Media Player
  - Fixed (another attempt) media player hiccups when there are too many finished jobs in the queue
  - (queue now has 2 ne toggles : pause auto refresh when playing video & auto cleanup finished results.
  - Turn on auto cleanup if you experience 'hiccups' in video playback, pause refresh also helps a little bit
- Startup
  - Fixed the little “%” toast popup during startup
- Prompt Enhancer
  - Resize bug fixed
- Misc
  - Fixed a bug in one of the easter eggs
</details>

<details>
  <summary>December 01, 2025 — v2.0 full release notes</summary>
  
**🧩 NEW Features**

Music Video Clip Creator_
- [ ] Works with every CPU or Gpu.
- [ ] Load one or multiple clips and/or images, add your music track, and create a complete music video in a couple of minutes*.
- [ ] Three settings for FX intensity or FX OFF (only hardcuts).
- [ ] One-click presets – apply 20+ settings and generate a full video in one click.
- [ ] Random seed toggle so you never get the same video. (or the same with your new edits when turned off)
- [ ] fade in/out with on/off toggle
- [ ] Built-in detector for auto selection of segments and energy peaks.
- [ ] Transitions with selection and randomization of favorites.
- [ ] Auto Ken Burns–style motion for images (+ extra settings for video).
- [ ] Global brightness/contrast/saturation normalization for smooth flow.
- [ ] Aspect ratios: 16:9 or 9:16 with Source, Fill (auto-crop), or Stretch modes.
- [ ] Multiple output resolutions supported.(source/lowest/highest + fixed 480/720/1080)
- [ ] Beat-synced Music visuals (this setting more then doubles generation time)
- [ ] 3 settings
- 1 (random) beat synced visual for the whole clip
- a new (random) one for every segment
- One per Section (intro verse chorus break drop outro) with user selection (200+ visuals)
- Transparency slider 0-100 (100 will only show beat synced visuals)
- [ ] Timeline Editor Tab
- Shows a detailed view of the entire track
- Visual preview of each segment with zoom and pan and right mouse button options
- Rename added clips/images.
- Add clips or images directly to segments.
- (beta) Save or load JSON timeline files for re-use.
-   *Lowering beats per segment, long tracks, and/or adding beat-synced visuals will make generation time longer. 

_Wan 2.2 – Text & Image to Video creation_

- [ ] Optional install (~30 GB on disk).
- [ ] Minimum 16 GB VRAM (RTX Nvidea) advised for stable performance.
- [ ] (Hard-coded by model) output at 1280×708, 24 fps.
- [ ] Choose between Text to Video, Image to Video, or Video to Video modes.
- [ ] Option to use the current frame of a (playing) video as the starting frame.
- [ ] Extend generated videos by re-using the last frame to make longer clips without extra VRAM.
- [ ] Includes (random) Seed, Guidance Scale, Loras, and other fine-tuning parameters.
- [ ] Batch support for:
- [ ] TXT2VID –> multiple videos from one text prompt.
- [ ] IMG2VID –> one image with multiple random seeds or several images with same/random seed.
- [ ] Recent results view with adjustable thumbnails for quick preview and management.

_Z-Image Turbo – Enhanced Image Generator_

- [ ] Creates more realistic images, with improved spelling accuracy and better prompt adherence.
- [ ] Optional install (~30–35 GB on disk). 16 gig vram (RTX Nvidea) advised for 720p and above
- [ ] When installed, it merges seamlessly with the default SDXL Image Creator tab and adds a new Engine Loader dropdown.
- [ ] Auto-switches settings (CFG, Steps, Output Name, etc.) per engine for best results.

_Ace Music creation_

- [ ] Optional install (~6 GB on disk).
- [ ] Genre preset system with favorites for instant style composition.*
 (* ace is still hit and miss, sometimes it's good result many times it is not good result)
 First run may download extra requirement files and may look stuck in the queue for a while.
- [ ] Fast generation, only 6–8 GB VRAM required.
- [ ] Reference track with slider for style matching. (beta)
- [ ] Negatives to exclude undesired elements.
- [ ] Recents list with Play/Delete options.
- [ ] Various fine-tuning controls.

### Audio Edit – mini Audacity-style editor

- [ ] Basic tool to edit sound files
- [ ] Crop MP3/WAV
- [ ] Convert formats
- [ ] Edit metadata
- [ ] Zoom & panning
- [ ] (beta) Right-click menu: Cut / Copy / Paste / Clear / Zoom to Selection / Show Full Waveform


_Video Split & Joiner_ – combine multiple clips or split a video into parts.

_Frame Extractor_ – now has extra option to rejoin frames into a video.

_Metadata Tool_ – edit or delete metadata from videos and images (with batch support.)

**🎨 UI Improvements**

- [ ] Executable launch option – start with FrameVision.exe (works both to install or run the app) instead of start.bat (.bat remains for debugging).
- [ ] Installer Options Menu – choose optional extras (Ace-Step Music, Wan 2.2, Z-Image Turbo).
- [ ] Recent results panel – sorting (date/size/alphabet) + delete function.
- [ ] Right-click menu in Recent Results – options to: Open Folder / Rename / Show Info / Delete from Drive.
- [ ] Emoji pack with on/off toggle + hide tab labels (only show the emoji instead of full tab name) option
- [ ] Colorful tab banners – toggle for on/off and color options.
- [ ] Panel swap toggle – media player left/right.
- [ ] Sticky bottom buttons for each tab.
- [ ] Important Buttons now change background color on hover.
- [ ] All themes now have color-changing edges on hover.

**🐞 Bugfixes**

- [ ] Fixed Z-Image Turbo batch/enqueue problems.
- [ ] Ace Music AI reference track fix – stable alignment and output.
- [ ] RIFE fixed for systems without FFmpeg on PATH.
- [ ] Frame Extractor fix – proper saving for single and batch modes.
</details>

---
FrameVision feature list
---
### 🎵 (music) videoclip creator
### 🧪 Upscaler with model loader (15+ open source upscale models for photo & video)
### ⏱️ RIFE Interpolation
### 📝 Describe in detail with Qwen3 VL 2B
### 🖌️ TXT → IMG (SD15/SDXL/Z-Image fp16/gguf q4-q8)
### 🎬 txt/img/video to video with WAN 2.2 5B and/or HunyuanVideo 1.5 (video extender included)
### 🎵 Ace Music creation
### 📁 Media Explorer
### 🖌 Remove Background + basic (sd15) inpainting
### 👽 Ask Framie, little offline chatbot, works with Qwen3VL 2B
### ⏳ Batch tools for almost all tools
### 📋 Advanced Queue with seperate worker for most tools
### 🧮 System Monitor with Hud
### 🕹️ EASTER EGGS : use the app to unlock them
### 📺 in app resizable Multi media player
### 🎶 Music player with playlist & 250+ beat synced visuals
### 🧰 Other Tools
- Change speed for videos (with audio sync).
- Prompt enhancer (Qwen3VL)
- Reverse video / boomerang
- Thumbnail/meme creator for easy adding text with effects to an image
- Sound edit (mini audacity)
- sound mixer for adding extra sound to video
- Resize image/video 
- Create Gif files : 2 sections -> use currently playing video or load images from a folder
- Extract/join frames
- Trim videos with preview
- Crop videos/images
- Multi rename/replace filenames
- Metadata/date editor/remover
### All kinds of extra's i forgot to mention such as a file menu with recents/favorites/converter, many random 'open app' screens with visual overlays, 20+ themes, an in app updater (to latest stable release or to latest files on github), a huge Knowledge base with q&a etc etc.
---

## 🛠 Installation
Run **`framevision.exe`** — it automatically:
1. Opens the installer menu if setup is incomplete. (give 1 time approval in windows to use an unknown app)
2. select 'check requirements' if you never installed something A.i. related before to have environment, git, python,... installed
3. This app is mainly created for RTX Nvidea users but an option for cpu install is also included (expect big models to not work or take forever to create a video or image)
4. After install it starts the app and background worker. (keep the background worker open when using queue)
5. select an optional install in the app (optional downloads menu) to get the model(s) you need (wan 2.2, hunyuan, z-image,..)

### Installer Menu Options
- 1 **check requirements** + basic 'first time ai' installer.
- 2 **Core Install:** App only (minimal).use this if app stops working after install to get a quick reset
- 3 **Full CPU Install:** Torch CPU + models/binaries.
- 4 **Full CUDA Install:** Torch CUDA + models/binaries.
---
User friendly, Portable & offline
(install must go to c:\framevision-main, once everything is installed no more internet is needed and you can move the folder to any other location on the drive.
One click install tries to make everything work out of the box (git, python, environment..) even for 'new to ai' users (Z-image gguf models may need extra install of visual studio (when you get error of missing dll files.)
Tooltips everywhere (can be turned of in settings tab)
Draggable middle splitter & swap ui to make the app look the way user wants. (Drag completely to one side to get a full screen for the tabs or the media player)
---

**Not working/bugs/todo:**
---
- To many finished results in queue still makes video stutter, clean finished results manually when you see it happen.
- image selection disabled in music video clip creator (needs to much re writing code to get it to work properly again, maybe one day
- timeline editing in music video ceator is not working properly anymore after adding to many new features, is on todo list to fix this.
- more ? let me know :-)
---

## 🤝 Contributing
bug reports or feature suggestions are welcome!  

---

## 📜 License
[MIT License](LICENSE)

---

> Built by **Contrinsan**  
> Debug help by **ChatGPT**  
> *This project is a living work — details evolve with releases. Features can come and go as the app matures*

