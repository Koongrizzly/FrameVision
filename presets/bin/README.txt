FrameVision tools (RIFE + FFmpeg)

This folder is used by the app to discover tools if they are not present in ./bin.
Safe locations the app searches:
 - ./bin/
 - ./presets/bin/
 - ./tools/rife/ (for RIFE only)
 - {tools_dir}/rife/ if set in config.json

You can place ffmpeg.exe and rife-ncnn-vulkan.exe here. Models/ folder for RIFE should sit next to the exe.
