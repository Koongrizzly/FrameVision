@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ------------------------------------------------------------------
REM ACE-Step portable environment installer (clean, no TorchCodec)
REM - Creates .ace_env venv under the app root
REM - Installs PyTorch + Torchaudio (CUDA 12.6 build)
REM - Installs app deps from ace_requirements.txt
REM - Clones ACE-Step repo into .ace_env\ACE-Step and installs it -e
REM ------------------------------------------------------------------

REM Determine app root (folder where this .bat lives)
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"
set "ROOT_DIR=%CD%"
popd

set "VENV_DIR=%ROOT_DIR%\.ace_env"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"
set "ACE_REPO_DIR=%VENV_DIR%\ACE-Step"

echo [ACE] Root dir:  %ROOT_DIR%
echo [ACE] Venv dir:  %VENV_DIR%
echo [ACE] Repo dir:  %ACE_REPO_DIR%
echo.

REM --- 1. Create virtual env -------------------------------------------------
if not exist "%VENV_DIR%" (
    echo [ACE] Creating virtualenv...
    python -m venv "%VENV_DIR%"
) else (
    echo [ACE] Using existing virtualenv.
)

if not exist "%PYTHON_EXE%" (
    echo [ACE] ERROR: Could not find venv Python at:
    echo        %PYTHON_EXE%
    echo        Did venv creation fail?
    pause
    exit /b 1
)

REM --- 2. Upgrade pip --------------------------------------------------------
"%ACE_ENV_DIR%\Scripts\python.exe" -m pip install "transformers>=4.44.0,<5.0.0" "peft>=0.13.0"

echo.
echo [ACE] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

REM --- 3. Install PyTorch + Torchaudio (CUDA 12.6 build) ---------------------
echo.
echo [ACE] Installing PyTorch + Torchaudio (cu126)...
"%PYTHON_EXE%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

REM --- 4. Install the core deps (NO torch / torchcodec here) -----------------
echo.
echo [ACE] Installing Python dependencies from ace_requirements.txt...
"%PYTHON_EXE%" -m pip install -r "%ROOT_DIR%\ace_requirements.txt"

REM --- 5. Clone ACE-Step repo if needed --------------------------------------
if not exist "%ACE_REPO_DIR%" (
    echo.
    echo [ACE] Cloning ACE-Step repo...
    git clone https://github.com/ace-step/ACE-Step.git "%ACE_REPO_DIR%"
) else (
    echo.
    echo [ACE] ACE-Step repo already exists:
    echo        %ACE_REPO_DIR%
)

REM --- 6. Install ACE-Step in editable mode ----------------------------------
echo.
echo [ACE] Installing ACE-Step (editable)...
pushd "%ACE_REPO_DIR%"
"%PYTHON_EXE%" -m pip install -e .
popd

REM --- 7. Ensure compatible transformers + peft versions ---------------------
echo.
echo [ACE] Ensuring compatible transformers/peft versions...
"%PYTHON_EXE%" -m pip install --upgrade "transformers==4.55.0" "peft>=0.13.0"

echo.
echo [ACE] Copying ACE-Step patches...

if not exist "%ACE_REPO_DIR%" mkdir "%ACE_REPO_DIR%"
if not exist "%ACE_REPO_DIR%\acestep" mkdir "%ACE_REPO_DIR%\acestep"
if not exist "%ACE_REPO_DIR%\acestep\music_dcae" mkdir "%ACE_REPO_DIR%\acestep\music_dcae"

copy /Y "%ROOT_DIR%\framevision_ace_runner.py" "%ACE_REPO_DIR%\"
copy /Y "%ROOT_DIR%\pipeline_ace_step.py" "%ACE_REPO_DIR%\acestep\"
copy /Y "%ROOT_DIR%\music_dcae_pipeline.py" "%ACE_REPO_DIR%\acestep\music_dcae\"



echo.
echo [ACE] Installation finished.
echo     Env:  %VENV_DIR%
echo     Repo: %ACE_REPO_DIR%
echo.
echo You can now run your helpers\\ace.py using this environment.
echo The first generation will download the model weights once (needs internet).
echo After that, it can run offline from this folder.
echo.

pause
endlocal
