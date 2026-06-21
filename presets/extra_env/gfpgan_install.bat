@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==========================================================
REM  GFPGAN one-click installer (portable venv)
REM  - Creates venv at:  <root>\models\gfpgan\.GFPGAN
REM  - Stores models at: <root>\models\gfpgan\
REM  - Installs CUDA PyTorch when NVIDIA GPU is available (best-effort)
REM ==========================================================

REM --- Resolve paths ---
set "SCRIPT_DIR=%~dp0"
REM SCRIPT_DIR ends with \presets\extra_env\
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT_DIR=%%~fI"

set "MODELS_DIR=%ROOT_DIR%\models\gfpgan"
set "ENV_DIR=%MODELS_DIR%\.GFPGAN"
set "REQ_FILE=%ROOT_DIR%\presets\extra_env\gfpgan_req.txt"

echo.
echo [GFPGAN] Root      = "%ROOT_DIR%"
echo [GFPGAN] Models    = "%MODELS_DIR%"
echo [GFPGAN] Env       = "%ENV_DIR%"
echo [GFPGAN] Req       = "%REQ_FILE%"
echo.

REM --- Basic checks ---
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found on PATH.
  echo         Install Python 3.10+ and make sure "python" works in CMD.
  echo.
  pause
  exit /b 1
)

if not exist "%REQ_FILE%" (
  echo [ERROR] Requirements file missing:
  echo         "%REQ_FILE%"
  echo.
  pause
  exit /b 1
)

REM --- Make folders ---
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%" >nul 2>nul

REM --- Create venv if needed ---
if not exist "%ENV_DIR%\Scripts\python.exe" (
  echo [GFPGAN] Creating venv...
  python -m venv "%ENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create venv at "%ENV_DIR%".
    echo.
    pause
    exit /b 1
  )
) else (
  echo [GFPGAN] Venv already exists. Reusing it.
)

REM --- Activate venv ---
call "%ENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate venv.
  echo.
  pause
  exit /b 1
)

REM --- Upgrade pip tooling ---
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [ERROR] pip upgrade failed.
  echo.
  pause
  exit /b 1
)

REM ==========================================================
REM  Install PyTorch (CUDA when available)
REM  Strategy:
REM   - If NVIDIA GPU seems present (nvidia-smi exists), try cu121 then cu118.
REM   - Otherwise install CPU wheels.
REM ==========================================================
set "HAS_NVIDIA=0"
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  nvidia-smi >nul 2>nul
  if not errorlevel 1 set "HAS_NVIDIA=1"
)

if "%HAS_NVIDIA%"=="1" (
  echo [GFPGAN] NVIDIA GPU detected. Trying CUDA PyTorch wheels...
  echo [GFPGAN]   Attempt: cu121
  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  if errorlevel 1 (
    echo [GFPGAN]   cu121 failed. Attempt: cu118
    python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  )
  if errorlevel 1 (
    echo [GFPGAN]   CUDA install failed. Falling back to CPU wheels...
    python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  )
) else (
  echo [GFPGAN] NVIDIA GPU not detected. Installing CPU PyTorch wheels...
  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

if errorlevel 1 (
  echo [ERROR] PyTorch install failed.
  echo.
  pause
  exit /b 1
)

REM --- Install GFPGAN deps ---
echo.
echo [GFPGAN] Installing requirements...
python -m pip install -r "%REQ_FILE%"
if errorlevel 1 (
  echo [ERROR] Requirements install failed.
  echo.
  pause
  exit /b 1
)

REM ==========================================================
REM  Download model weights
REM    - GFPGANv1.4.pth
REM    - FaceXlib weights used by GFPGAN (detection/parsing/alignment)
REM  These are placed into FaceXlib's installed weights folder to avoid
REM  "first run downloads" later.
REM ==========================================================

set "GFPGAN_PTH=%MODELS_DIR%\GFPGANv1.4.pth"
set "URL_GFPGAN=https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"

if not exist "%GFPGAN_PTH%" (
  echo.
  echo [GFPGAN] Downloading: GFPGANv1.4.pth
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { Invoke-WebRequest -Uri '%URL_GFPGAN%' -OutFile '%GFPGAN_PTH%' -UseBasicParsing } catch { exit 1 }"
  if errorlevel 1 (
    echo [ERROR] Failed to download GFPGANv1.4.pth
    echo         You can download it manually and place it here:
    echo         "%GFPGAN_PTH%"
    echo.
    pause
    exit /b 1
  )
) else (
  echo [GFPGAN] Model exists: "%GFPGAN_PTH%"
)

REM --- Find FaceXlib weights directory in this venv ---
for /f "usebackq delims=" %%I in (`python -c "import facexlib, os; print(os.path.join(os.path.dirname(facexlib.__file__), 'weights'))"`) do set "FACEX_WEIGHTS=%%I"
if "%FACEX_WEIGHTS%"=="" (
  echo [ERROR] Could not resolve facexlib weights path.
  echo.
  pause
  exit /b 1
)
if not exist "%FACEX_WEIGHTS%" mkdir "%FACEX_WEIGHTS%" >nul 2>nul

echo.
echo [GFPGAN] FaceXlib weights dir: "%FACEX_WEIGHTS%"

REM --- Weight URLs (FaceXlib release assets) ---
set "URL_DET=https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
set "URL_ALIGN=https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth"
set "URL_PARSE=https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"

call :download_if_missing "%URL_DET%"   "%FACEX_WEIGHTS%\detection_Resnet50_Final.pth"
if errorlevel 1 goto :download_fail
call :download_if_missing "%URL_ALIGN%" "%FACEX_WEIGHTS%\alignment_WFLW_4HG.pth"
if errorlevel 1 goto :download_fail
call :download_if_missing "%URL_PARSE%" "%FACEX_WEIGHTS%\parsing_parsenet.pth"
if errorlevel 1 goto :download_fail

REM --- Quick sanity check ---
echo.
echo [GFPGAN] Sanity check imports...
python -c "import torch; from gfpgan import GFPGANer; print('OK: torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
if errorlevel 1 (
  echo [ERROR] Import sanity check failed.
  echo.
  pause
  exit /b 1
)

echo.
echo [GFPGAN] Done.
echo [GFPGAN] Environment: "%ENV_DIR%"
echo [GFPGAN] Model file : "%GFPGAN_PTH%"
echo.
pause
exit /b 0

:download_fail
echo.
echo [ERROR] One or more FaceXlib weights failed to download.
echo         You can re-run this installer, or download them manually into:
echo         "%FACEX_WEIGHTS%"
echo.
pause
exit /b 1

:download_if_missing
set "DL_URL=%~1"
set "DL_OUT=%~2"
if exist "%DL_OUT%" (
  echo [GFPGAN] Weight exists: "%DL_OUT%"
  exit /b 0
)
echo [GFPGAN] Downloading: "%DL_OUT%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { Invoke-WebRequest -Uri '%DL_URL%' -OutFile '%DL_OUT%' -UseBasicParsing } catch { exit 1 }"
if errorlevel 1 exit /b 1
exit /b 0
