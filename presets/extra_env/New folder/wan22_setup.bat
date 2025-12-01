@echo off
REM WAN 2.2 5B - FINAL CLEAN INSTALLER
REM - Uses Python snapshot_download (no hf CLI)
REM - Downloads WAN-AI/Wan2.2-TI2V-5B into models\wan22
REM - Downloads GitHub Wan2.2 repo and merges scripts

echo.
echo ===============================================
echo   WAN 2.2 5B - Installer / Repair (FINAL)
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
REM 4) Install PyTorch + WAN requirements + huggingface_hub
REM ------------------------------------------------

echo.
echo [INFO] Installing PyTorch (CUDA 12.1)...
"%VENV_DIR%\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :fatal_error

echo.
echo [INFO] Installing WAN 2.2 requirements...
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%REQ_FILE%"
if errorlevel 1 goto :fatal_error
echo.
echo [INFO] Installing optimization deps (Triton, Sage, TeaCache, MagCache)...
"%VENV_DIR%\Scripts\pip.exe" install triton-windows==3.5.0
if errorlevel 1 goto :fatal_error

"%VENV_DIR%\Scripts\pip.exe" install git+https://github.com/thu-ml/SageAttention.git
if errorlevel 1 goto :fatal_error

"%VENV_DIR%\Scripts\pip.exe" install git+https://github.com/ali-vilab/TeaCache.git
if errorlevel 1 goto :fatal_error

"%VENV_DIR%\Scripts\pip.exe" install git+https://github.com/Zehong-Ma/MagCache.git
if errorlevel 1 goto :fatal_error

"%VENV_DIR%\Scripts\pip.exe" install bitsandbytes>=0.44.1
if errorlevel 1 goto :fatal_error
echo.
echo [INFO] Ensuring compatible huggingface_hub (>=0.30,<1.0)...
"%VENV_DIR%\Scripts\python.exe" -m pip install "huggingface_hub>=0.30,<1.0"
if errorlevel 1 goto :fatal_error

REM ------------------------------------------------
REM 5) Download WAN 2.2-5B weights from Hugging Face via Python
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
REM 6) Download + merge GitHub repo (Wan2.2 main)
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
REM FINAL STEP: copy patched generate.py into models\wan22
REM ------------------------------------------------
echo.
REM After WAN's generate.py is downloaded, overwrite it with our patched versions
echo [INFO] Patching WAN generate.py and attention.py...

copy /Y "%APP_ROOT%\presets\extra_env\generate.py" ^
    "%APP_ROOT%\models\wan22\generate.py"

copy /Y "%APP_ROOT%\presets\extra_env\attention.py" ^
    "%APP_ROOT%\models\wan22\wan\modules\attention.py"

if errorlevel 1 (
    echo [WARN] Could not copy one or more WAN patch files.
) else (
    echo [OK] WAN patch files updated (generate.py + attention.py)
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
REM ------------------------------------------------

echo.
pause
exit /b 1
