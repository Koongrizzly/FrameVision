@echo off
setlocal EnableExtensions
REM FrameVision LTX installer (minimal downloads)

REM Auto-root from this script location: <ROOT>\presets\extra_env\ltx_install.bat
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"

echo [LTX] Root: %ROOT%

REM Keep HuggingFace caches inside the app folder (portable)
set "HF_HOME=%ROOT%\models\_hf"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "TRANSFORMERS_CACHE=%HF_HOME%\transformers"
set "HF_DATASETS_CACHE=%HF_HOME%\datasets"

if not exist "%HF_HOME%" mkdir "%HF_HOME%" >nul 2>nul
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%" >nul 2>nul
if not exist "%TRANSFORMERS_CACHE%" mkdir "%TRANSFORMERS_CACHE%" >nul 2>nul
if not exist "%HF_DATASETS_CACHE%" mkdir "%HF_DATASETS_CACHE%" >nul 2>nul

REM Create venv if needed
if not exist "%ROOT%\.ltx_env\Scripts\python.exe" (
  echo [LTX] Creating venv: %ROOT%\.ltx_env
  py -3 -m venv "%ROOT%\.ltx_env"
  if errorlevel 1 (
    echo [LTX] ERROR: Failed to create venv. Install Python 3.10+ and ensure "py" launcher works.
    exit /b 1
  )
)

call "%ROOT%\.ltx_env\Scripts\activate.bat"
if errorlevel 1 (
  echo [LTX] ERROR: Failed to activate venv.
  exit /b 1
)

echo [LTX] Upgrading pip...
python -m pip install -U pip setuptools wheel

echo [LTX] Installing core deps...
python -m pip install -U huggingface_hub diffusers transformers accelerate sentencepiece safetensors

REM Torch install is left to your main FrameVision installer (to avoid mismatches).
REM If you need torch here, uncomment one of these lines:
REM python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
REM python -m pip install torch torchvision torchaudio

echo [LTX] Downloading minimal models (ONLY 2 GGUF files + decoder essentials)...
python "%ROOT%\presets\extra_env\ltx_download.py" --root "%ROOT%" --transformer_gguf ltx-video-2b-v0.9-q5_k_m.gguf --t5_gguf t5xxl_fp16-q4_0.gguf
if errorlevel 1 (
  echo [LTX] ERROR: Download failed.
  exit /b 1
)

echo [LTX] Install complete.
exit /b 0
