@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==========================================================
REM FrameVision Optional Install: GLM-Image (CUDA)
REM Creates venv at:    environments\.glm_image
REM Downloads to:       models\glm-image\
REM ==========================================================

REM Resolve FrameVision root (this .bat lives in presets\extra_env)
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "FV_ROOT=%%~fI"

set "ENV_DIR=%FV_ROOT%\environments\.glm_image"
set "REQ_FILE=%SCRIPT_DIR%glm_req.txt"
set "PY_EXE=%ENV_DIR%\Scripts\python.exe"

echo.
echo ==========================================================
echo GLM-Image installer (CUDA) - FrameVision
echo Root:   "%FV_ROOT%"
echo Env:    "%ENV_DIR%"
echo ==========================================================
echo.

REM --- Create environment (requires Python on PATH)
if not exist "%ENV_DIR%\Scripts\python.exe" (
  echo [1/5] Creating virtual environment...
  py -3 -m venv "%ENV_DIR%"
  if errorlevel 1 (
    echo.
    echo ERROR: Failed to create venv. Make sure Python 3 is installed and 'py' launcher works.
    echo.
    pause
    exit /b 1
  )
) else (
  echo [1/5] Virtual environment already exists.
)

echo [2/5] Upgrading pip tooling...
"%PY_EXE%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo ERROR: Failed to upgrade pip tooling.
  pause
  exit /b 1
)

REM --- Install PyTorch CUDA wheels
REM Default: cu121 (works on most modern NVIDIA setups)
REM If you need a different CUDA wheel, edit the index-url below.
echo [3/5] Installing PyTorch (CUDA wheels)...
"%PY_EXE%" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
if errorlevel 1 (
  echo.
  echo ERROR: Failed to install PyTorch CUDA wheels.
  echo Tip: If your GPU/driver requires a newer CUDA wheel ^(e.g., cu124^), edit this .bat to use:
  echo      https://download.pytorch.org/whl/cu124
  echo.
  pause
  exit /b 1
)

echo [4/5] Installing GLM-Image requirements...
"%PY_EXE%" -m pip install --upgrade -r "%REQ_FILE%"
if errorlevel 1 (
  echo ERROR: Failed to install Python requirements.
  pause
  exit /b 1
)

echo [5/5] Downloading GLM-Image model + repo (this can take a long time)...
"%PY_EXE%" "%SCRIPT_DIR%glm_downloads.py" --framevision-root "%FV_ROOT%"
if errorlevel 1 (
  echo ERROR: Download step failed.
  pause
  exit /b 1
)

echo.
echo ==========================================================
echo GLM-Image install complete.
echo Models: "%FV_ROOT%\models\glm-image"
echo Env:    "%ENV_DIR%"
echo ==========================================================
echo.
pause
exit /b 0
