@echo off
setlocal EnableExtensions

REM Runs from: root\presets\extra_env\
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%..\..\"
pushd "%ROOT%" >nul

echo [Qwen3-TTS] One-click installer
echo ----------------------------------------------------------
echo This will create environments\.qwen3tts and download models into models\
echo.

REM Create folders
if not exist "environments" mkdir "environments"
if not exist "models" mkdir "models"
if not exist "output" mkdir "output"
if not exist "presets\setsave" mkdir "presets\setsave"

set "VENV=environments\.qwen3tts"

if not exist "%VENV%" (
  echo Creating virtual environment %VENV% ...
  python -m venv "%VENV%"
  if errorlevel 1 (
    echo ERROR: Failed to create venv. Make sure Python is installed.
    popd >nul
    pause
    exit /b 1
  )
)

call "%VENV%\Scripts\activate.bat" >nul 2>nul
if errorlevel 1 (
  echo ERROR: Failed to activate venv.
  popd >nul
  pause
  exit /b 1
)

echo [1/6] Upgrading pip tooling
python -m pip install -U pip setuptools wheel
if errorlevel 1 (
  echo ERROR: pip tooling upgrade failed.
  popd >nul
  pause
  exit /b 1
)

echo [2/6] Installing PyTorch (CUDA attempt, then fallback)
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio >nul 2>nul
if errorlevel 1 (
  echo CUDA wheels not available ^(or no compatible GPU^). Installing default torch...
  python -m pip install -U torch torchvision torchaudio
  if errorlevel 1 (
    echo ERROR: torch install failed.
    popd >nul
    pause
    exit /b 1
  )
)

echo [3/6] Installing Qwen3-TTS UI requirements
python -m pip install -r "presets\extra_env\requirements_qwentts.txt"
if errorlevel 1 (
  echo ERROR: dependency install failed.
  popd >nul
  pause
  exit /b 1
)

echo [4/6] Ensuring SoX is available
call "presets\extra_env\install_sox.bat"
if errorlevel 1 (
  echo ERROR: SoX setup failed.
  popd >nul
  pause
  exit /b 1
)

echo [5/6] Downloading models (can take a while)
call "presets\extra_env\download_qwentts_models.bat"
if errorlevel 1 (
  echo ERROR: Model download failed.
  popd >nul
  pause
  exit /b 1
)

echo [6/6] Cloning repo (optional)
where git >nul 2>nul
if errorlevel 1 (
  echo Git not found. Skipping repo clone.
) else (
  if not exist "repo" (
    git clone https://github.com/QwenLM/Qwen3-TTS.git "repo"
  ) else (
    echo Repo folder already exists. Skipping clone.
  )
)

echo.
echo Done.
echo - You can now run: helpers\qwentts_ui.py  (it will auto-use environments\.qwen3tts)
echo - Optional: presets\extra_env\install_flashattn_optional.bat

popd >nul
pause
exit /b 0
