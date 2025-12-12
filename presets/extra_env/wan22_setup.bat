@echo off
REM WAN 2.2 5B - FINAL CLEAN INSTALLER (FIXED)
REM - Uses Python snapshot_download (no hf CLI)
REM - Downloads WAN-AI/Wan2.2-TI2V-5B into models\wan22
REM - Downloads GitHub Wan2.2 repo and merges scripts
REM - Pins CUDA PyTorch and prevents it from being overwritten

echo.
echo ===============================================
echo   WAN 2.2 5B - Installer / Repair (FIXED)
echo ===============================================
echo.

REM ------------------------------------------------
REM 1) Resolve APP_ROOT relative to this script
REM    Script is expected in: APP_ROOT\presets\extra_env\
REM ------------------------------------------------

set "SCRIPT_DIR=%~dp0"

pushd "%SCRIPT_DIR%\..\.."
set "APP_ROOT=%CD%"
popd

set "VENV_DIR=%APP_ROOT%\.wan_venv"
set "MODEL_DIR=%APP_ROOT%\models\wan22"
set "REQ_FILE=%SCRIPT_DIR%wan22_requirements.txt"

echo [INFO] Script directory  : %SCRIPT_DIR%
echo [INFO] App root directory: %APP_ROOT%
echo [INFO] Venv directory    : %VENV_DIR%
echo [INFO] Model directory   : %MODEL_DIR%
echo [INFO] Requirements file : %REQ_FILE%
echo.

REM ------------------------------------------------
REM 2) Basic checks
REM ------------------------------------------------

if not exist "%REQ_FILE%" (
    echo [ERROR] Requirements file not found: %REQ_FILE%
    goto :fatal_error
)

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No 'python' found in PATH.
    goto :fatal_error
)

REM ------------------------------------------------
REM 3) Create / reuse virtual environment
REM ------------------------------------------------

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtualenv.
        goto :fatal_error
    )
) else (
    echo [INFO] Reusing existing virtual environment.
)

echo.
echo [INFO] Upgrading pip, setuptools, wheel...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fatal_error

REM ------------------------------------------------
REM 4) Install CUDA PyTorch (pinned) FIRST
REM    We install torch/vision/audio explicitly from the cu121 index
REM    so they do NOT get replaced by CPU wheels.
REM ------------------------------------------------

echo.
echo [INFO] Installing PyTorch (CUDA 12.1, pinned)...
"%VENV_DIR%\Scripts\python.exe" -m pip install ^
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 ^
  --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :fatal_error

REM ------------------------------------------------
REM 5) Install WAN 2.2 requirements (without torch)
REM    torch/vision/audio are NOT in wan22_requirements.txt.
REM    We also avoid --upgrade here to stop pip from trying to
REM    "improve" torch to 2.9.1+cpu.
REM ------------------------------------------------

echo.
echo [INFO] Installing WAN 2.2 requirements (no torch)...
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%REQ_FILE%"
if errorlevel 1 goto :fatal_error

REM ------------------------------------------------
REM 6) Ensure WAN core deps are pinned WITHOUT touching torch
REM    Important: we use --no-deps so these packages do not try
REM    to pull in another torch version.
REM ------------------------------------------------

echo.
echo [INFO] Ensuring WAN core Python deps (transformers/diffusers/peft) are pinned...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade --no-deps ^
  "transformers>=4.52.0,<5.0.0" ^
  "diffusers==0.35.2" ^
  "peft==0.18.0"
if errorlevel 1 goto :fatal_error

echo.
echo [INFO] Ensuring compatible huggingface_hub (>=0.30,<1.0)...
"%VENV_DIR%\Scripts\python.exe" -m pip install "huggingface_hub>=0.30,<1.0"
if errorlevel 1 goto :fatal_error

REM ------------------------------------------------
REM 7) Download WAN 2.2-5B weights from Hugging Face via Python
REM     Repo: Wan-AI/Wan2.2-TI2V-5B (gated, requires HF login)
REM ------------------------------------------------

if not exist "%MODEL_DIR%" (
    echo [INFO] Creating model directory: %MODEL_DIR%
    mkdir "%MODEL_DIR%" 2>nul
)

echo.
echo [INFO] Checking for existing WAN 2.2-5B weights...
if exist "%MODEL_DIR%\Wan2.2_VAE.pth" goto have_weights

echo [INFO] Wan2.2_VAE.pth not found, downloading WAN-AI/Wan2.2-TI2V-5B...
echo        If you get a 401 / RepositoryNotFound error, run:
echo        - .wan_venv\Scripts\activate
echo        - huggingface-cli login
echo        and make sure you have access to the model on Hugging Face.

"%VENV_DIR%\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.2-TI2V-5B', local_dir=r'%MODEL_DIR%', local_dir_use_symlinks=False)"
if errorlevel 1 goto :fatal_error

:have_weights
echo.
echo [INFO] Contents of model directory after HF download:
dir "%MODEL_DIR%"
echo.

REM ------------------------------------------------
REM 8) Download + merge GitHub repo (Wan2.2 main)
REM ------------------------------------------------

echo [INFO] Downloading Wan2.2 GitHub repo (generate.py and scripts)...

set "WAN_REPO_URL=https://github.com/Wan-Video/Wan2.2/archive/refs/heads/main.zip"
set "WAN_REPO_ZIP=%APP_ROOT%\wan22_repo.zip"
set "WAN_REPO_TMP=%APP_ROOT%\wan22_repo"

if exist "%WAN_REPO_ZIP%" del /f /q "%WAN_REPO_ZIP%" >nul 2>&1
if exist "%WAN_REPO_TMP%" rmdir /s /q "%WAN_REPO_TMP%" >nul 2>&1

powershell -NoProfile -Command "Invoke-WebRequest -Uri '%WAN_REPO_URL%' -OutFile '%WAN_REPO_ZIP%'" 
if errorlevel 1 goto :fatal_error

powershell -NoProfile -Command "Expand-Archive -LiteralPath '%WAN_REPO_ZIP%' -DestinationPath '%WAN_REPO_TMP%' -Force"
if errorlevel 1 goto :fatal_error

echo [INFO] Merging Wan2.2 repo files into model directory...
xcopy "%WAN_REPO_TMP%\Wan2.2-main\*" "%MODEL_DIR%\" /E /I /Y >nul
if errorlevel 1 goto :fatal_error

del /f /q "%WAN_REPO_ZIP%" >nul 2>&1
rmdir /s /q "%WAN_REPO_TMP%" >nul 2>&1

echo.
echo [SUCCESS] WAN 2.2 5B installation / repair finished.
echo          Model + scripts are in: %MODEL_DIR%
REM FINAL STEP: apply wan22.zip patch bundle into models root
REM ------------------------------------------------
echo.
set "WAN_PATCH_ZIP=%SCRIPT_DIR%wan22.zip"
echo [INFO] Applying WAN patch bundle from: %WAN_PATCH_ZIP%

if exist "%WAN_PATCH_ZIP%" (
    powershell -NoProfile -Command "Expand-Archive -LiteralPath '%WAN_PATCH_ZIP%' -DestinationPath '%APP_ROOT%\models' -Force"
    if errorlevel 1 (
        echo [WARN] Could not extract wan22.zip patch bundle.
    ) else (
        echo [OK] wan22.zip extracted into %APP_ROOT%\models
    )
) else (
    echo [WARN] wan22.zip not found in %SCRIPT_DIR% - skipping patch bundle.
)

echo.
pause
exit /b 0

REM ------------------------------------------------
REM FATAL ERROR HANDLER
REM ------------------------------------------------
:fatal_error
echo.
echo [FATAL] WAN 2.2 setup did NOT complete successfully.
echo         Check the last messages above for the failing step.
echo.
pause
exit /b 1
