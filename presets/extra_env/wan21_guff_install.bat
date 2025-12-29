@echo off
setlocal ENABLEDELAYEDEXPANSION

rem Run from: <root>\presets\extra_env\wan21_guff_install.bat
cd /d "%~dp0\..\.."
set ROOT=%cd%

echo.
echo ==========================================================
echo  WAN 2.1 GGUF - One Click Installer (no ComfyUI required)
echo ==========================================================
echo Root: %ROOT%
echo.

if not exist ".wan21gguf_env" mkdir ".wan21gguf_env"

if not exist ".wan21gguf_env\venv\Scripts\python.exe" (
  echo [1/5] Creating venv...
  python -m venv ".wan21gguf_env\venv"
  if errorlevel 1 (
    echo.
    echo ERROR: Failed to create venv. Make sure Python is installed and on PATH.
    pause
    exit /b 1
  )
) else (
  echo [1/5] venv already exists.
)

call ".wan21gguf_env\venv\Scripts\activate.bat"

echo [2/5] Updating pip...
python -m pip install --upgrade pip

echo [3/5] Installing requirements...
pip install -r "presets\extra_env\wan21_guff_req.txt"
if errorlevel 1 (
  echo.
  echo ERROR: pip install failed.
  pause
  exit /b 1
)

echo [4/5] Downloading stable-diffusion.cpp (sd-cli) binaries...
python "presets\extra_env\wan21_guff.py" ensure-sdcpp --root "%ROOT%"
if errorlevel 1 (
  echo.
  echo ERROR: Failed to download stable-diffusion.cpp binaries.
  pause
  exit /b 1
)

echo [5/5] Creating folders + default settings...
python "presets\extra_env\wan21_guff.py" ensure-modeldirs --root "%ROOT%"
python "presets\extra_env\wan21_guff.py" write-default-settings --root "%ROOT%"

echo.
echo Done!
echo - GGUF models live in: %ROOT%\models\wan21gguf\
echo - sd-cli lives in:     %ROOT%\.wan21gguf_env\sdcpp\
echo.
pause
