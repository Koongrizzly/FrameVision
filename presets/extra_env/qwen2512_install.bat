@echo off
setlocal EnableExtensions

REM ==========================================================
REM FrameVision One-Click Installer: Qwen-Image-2512 (GGUF)
REM Installs a small Python venv under /.qwen2512/ and downloads:
REM   - qwen-image-2512-Q4_K_M.gguf  (~13.1 GB)
REM   - Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf (~4.79 GB)
REM   - qwen_image_vae.safetensors  (~254 MB)
REM ==========================================================

set "SCRIPT_DIR=%~dp0"
REM qwen2512_install.bat lives in: <root>\presets\extra_env\
REM So ROOT is 2 levels up from SCRIPT_DIR.
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"

set "ENV_DIR=%ROOT%\.qwen2512\venv"
set "MODELS_DIR=%ROOT%\models\Qwen-Image-2512 GGUF"
set "BIN_DIR=%ROOT%\.qwen2512\bin"
set "DL_SCRIPT=%ROOT%\presets\extra_env\qwen2512_download.py"
set "MARKER=%ROOT%\.qwen2512\installed_qwen2512.txt"

if not exist "%ROOT%\.qwen2512" mkdir "%ROOT%\.qwen2512"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "%BIN_DIR%" mkdir "%BIN_DIR%"
if not exist "%ROOT%\outputs" mkdir "%ROOT%\outputs\images\qwen2512"

REM ---- Try to locate FrameVision's embedded Python first ----
set "PY_EXE="
if exist "%ROOT%\python\python.exe" set "PY_EXE=%ROOT%\python\python.exe"
if not defined PY_EXE if exist "%ROOT%\_internal\python\python.exe" set "PY_EXE=%ROOT%\_internal\python\python.exe"
if not defined PY_EXE if exist "%ROOT%\runtime\python\python.exe" set "PY_EXE=%ROOT%\runtime\python\python.exe"

if not defined PY_EXE (
  REM Fall back to system python
  set "PY_EXE=python"
)

echo.
echo [.QWEN2512] Root:      "%ROOT%"
echo [.QWEN2512] Python:    "%PY_EXE%"
echo [.QWEN2512] Env dir:   "%ENV_DIR%"
echo [QWEN2512] Models dir:"%MODELS_DIR%"
echo [.QWEN2512] Bin dir:   "%BIN_DIR%"
echo.

REM ---- Create venv (idempotent) ----
if not exist "%ENV_DIR%\Scripts\python.exe" (
  echo [QWEN2512] Creating venv...
  "%PY_EXE%" -m venv "%ENV_DIR%"
  if errorlevel 1 (
    echo [QWEN2512] ERROR: Failed to create venv.
    exit /b 1
  )
)

set "VPY=%ENV_DIR%\Scripts\python.exe"

echo [QWEN2512] Upgrading pip...
"%VPY%" -m pip install --upgrade pip --quiet
if errorlevel 1 (
  echo [QWEN2512] WARNING: pip upgrade failed - continuing.
)

echo [QWEN2512] Installing Python deps: requests, tqdm...
"%VPY%" -m pip install --upgrade requests tqdm --quiet
if errorlevel 1 (
  echo [QWEN2512] ERROR: Failed to install dependencies.
  exit /b 1
)

REM ---- Run downloader ----
if not exist "%DL_SCRIPT%" (
  echo [QWEN2512] ERROR: Missing downloader script: "%DL_SCRIPT%"
  exit /b 1
)

echo.
echo [QWEN2512] Downloading model files (Q4 preset)...
"%VPY%" "%DL_SCRIPT%" --root "%ROOT%" --models-dir "%MODELS_DIR%" --bin-dir "%BIN_DIR%" --ensure-cli
if errorlevel 1 (
  echo [QWEN2512] ERROR: Download failed.
  exit /b 1
)

echo.
echo [QWEN2512] Writing install marker...
echo Qwen-Image-2512 GGUF installed on %DATE% %TIME%> "%MARKER%"

echo.
echo [QWEN2512] DONE.
echo Models are in: "%MODELS_DIR%"
echo Marker: "%MARKER%"
echo.
pause
exit /b 0
