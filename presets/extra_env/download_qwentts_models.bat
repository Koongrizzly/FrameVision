@echo off
setlocal EnableExtensions

REM Runs from: root\presets\extra_env\
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%..\..\"
pushd "%ROOT%" >nul

echo.
echo [Qwen3-TTS] Downloading models to models\
echo ----------------------------------------------------------

if not exist "models" mkdir "models"

if not exist "environments\.qwen3tts\Scripts\python.exe" (
  echo ERROR: Environment not found at environments\.qwen3tts
  echo Please run presets\extra_env\install_qwentts.bat first.
  popd >nul
  exit /b 1
)

call "environments\.qwen3tts\Scripts\activate.bat" >nul 2>nul

REM Force-disable hf_transfer if user has it enabled globally (prevents hf_transfer errors).
set HF_HUB_ENABLE_HF_TRANSFER=0
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

REM Use hf if available, else huggingface-cli
where hf >nul 2>nul
if errorlevel 1 (
  set HF_CMD=huggingface-cli
) else (
  set HF_CMD=hf
)

REM CustomVoice model
%HF_CMD% download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir "models\Qwen3-TTS-12Hz-1.7B-CustomVoice"
if errorlevel 1 (
  echo ERROR: Failed to download CustomVoice model.
  popd >nul
  exit /b 1
)

REM Tokenizer
%HF_CMD% download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir "models\Qwen3-TTS-Tokenizer-12Hz"
if errorlevel 1 (
  echo ERROR: Failed to download Tokenizer.
  popd >nul
  exit /b 1
)

echo.
echo OK: Models downloaded.

popd >nul
exit /b 0
