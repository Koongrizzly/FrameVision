@echo off
setlocal ENABLEDELAYEDEXPANSION

echo.
echo === ComfyUI one-click setup (RTX / CUDA) ===
echo This will auto-download ComfyUI (if needed), create a FRESH .comfy_env,
echo install CUDA-enabled PyTorch for NVIDIA RTX, and install ComfyUI requirements.
echo.

rem Determine project root (this .bat is expected in presets\extra_env)
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\..\.."
set "PROJECT_ROOT=%CD%"

echo Project root: %PROJECT_ROOT%
echo.

rem -------------------------------------------------------------------
rem 1) Ensure ComfyUI folder exists (auto clone or zip-download if missing)
rem -------------------------------------------------------------------
if not exist "ComfyUI" (
    echo ComfyUI folder not found, attempting to download...

    set "USE_PWSH=0"

    rem Try git first
    where git >NUL 2>&1
    if %ERRORLEVEL%==0 (
        echo Git found, cloning from GitHub...
        git clone https://github.com/comfyanonymous/ComfyUI.git "ComfyUI"
        if %ERRORLEVEL% NEQ 0 (
            echo ERROR: Failed to clone ComfyUI with git.
            echo Will try PowerShell ZIP download instead...
            set "USE_PWSH=1"
        )
    ) else (
        echo Git not found, will use PowerShell ZIP download...
        set "USE_PWSH=1"
    )

    if "!USE_PWSH!"=="1" (
        echo Downloading ComfyUI.zip via PowerShell...
        powershell -Command "Invoke-WebRequest 'https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip' -OutFile 'ComfyUI.zip'"
        if %ERRORLEVEL% NEQ 0 (
            echo ERROR: Failed to download ComfyUI.zip
            echo Check your internet connection or download ComfyUI manually into:
            echo   %PROJECT_ROOT%\ComfyUI
            pause
            exit /b 1
        )

        echo Extracting ComfyUI.zip ...
        powershell -Command "Expand-Archive -Force 'ComfyUI.zip' '.'"
        if %ERRORLEVEL% NEQ 0 (
            echo ERROR: Failed to extract ComfyUI.zip
            pause
            exit /b 1
        )

        if exist "ComfyUI-master" (
            ren "ComfyUI-master" "ComfyUI"
        )

        del "ComfyUI.zip"
    )
) else (
    echo ComfyUI folder already exists. Skipping download.
)

rem Double-check that requirements.txt exists
if not exist "ComfyUI\requirements.txt" (
    echo ERROR: ComfyUI\requirements.txt not found.
    echo The ComfyUI folder may be incomplete or corrupted.
    echo Please delete the ComfyUI folder and re-run this installer,
    echo or download ComfyUI manually into:
    echo   %PROJECT_ROOT%\ComfyUI
    pause
    exit /b 1
)

rem -------------------------------------------------------------------
rem 2) Create FRESH venv (.comfy_env)
rem -------------------------------------------------------------------
if exist ".comfy_env" (
    echo Existing .comfy_env found - deleting to start from scratch...
    rmdir /S /Q ".comfy_env"
)

echo [1/4] Creating virtual environment: .comfy_env
python -m venv ".comfy_env"
if errorlevel 1 (
    echo Failed to create virtual environment. Make sure Python is installed and on PATH.
    pause
    exit /b 1
)

call ".comfy_env\Scripts\activate"
if errorlevel 1 (
    echo Failed to activate .comfy_env. Aborting.
    pause
    exit /b 1
)

echo.
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip

rem -------------------------------------------------------------------
rem 3) Install CUDA-enabled PyTorch (for NVIDIA RTX)
rem    Adjust CUDA version here if needed.
rem -------------------------------------------------------------------
echo.
echo [3/4] Installing CUDA-enabled PyTorch (torch/vision/audio, cu121)...
python -m pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install CUDA-enabled PyTorch.
    echo - Check that you have an NVIDIA GPU and recent drivers installed.
    echo - If needed, see https://pytorch.org/get-started/locally/
    pause
    exit /b 1
)

rem -------------------------------------------------------------------
rem 4) Install ComfyUI requirements (torch already satisfied by GPU build)
rem -------------------------------------------------------------------
echo.
echo [4/4] Installing ComfyUI requirements...
python -m pip install -r "ComfyUI\requirements.txt"
if errorlevel 1 (
    echo Failed to install ComfyUI requirements.
    pause
    exit /b 1
)

echo.
echo Forcing numpy<2 to avoid NumPy 2.x ABI issues...
python -m pip install "numpy<2"


echo.
echo === Done ===
echo Virtual env: %PROJECT_ROOT%\.comfy_env
echo To start ComfyUI server:
echo   cd %PROJECT_ROOT%
echo   call .comfy_env\Scripts\activate
echo   python ComfyUI\\main.py
echo.

pause
endlocal

