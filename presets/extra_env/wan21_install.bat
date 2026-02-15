@echo off
setlocal ENABLEDELAYEDEXPANSION

REM -----------------------------------------------------
REM WAN 2.1 VACE 1.3B - One Click Environment Installer
REM (Patched: UTF-8 console + use `hf download` without
REM unsupported flags)
REM -----------------------------------------------------

REM Force UTF-8 to avoid UnicodeEncodeError in Windows consoles
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo =====================================================
echo   WAN 2.1 VACE 1.3B - One Click Environment Installer
echo =====================================================
echo.

REM Resolve root folder (two levels up from this script: presets\extra_env\)
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\..\.." >nul
set "ROOT=%CD%"

echo Root folder detected:
echo   %ROOT%
echo.

REM Create virtual environment if it does not exist yet
if not exist "%ROOT%\.wan21_env" (
    echo Creating Python virtual environment in:
    echo   %ROOT%\.wan21_env
    echo.
    python -m venv "%ROOT%\.wan21_env"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Make sure Python 3.10+ is installed and available on PATH.
        echo.
        pause
        goto :end
    )
) else (
    echo Reusing existing virtual environment:
    echo   %ROOT%\.wan21_env
    echo.
)

REM Activate the venv
call "%ROOT%\.wan21_env\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    echo.
    pause
    goto :end
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch 2.5.1 (CUDA 12.4 wheels) ...
echo (This may take a few minutes.)
echo.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [WARNING] PyTorch install reported an error. You may need to install it manually later.
    echo.
)

echo.
echo Installing remaining Python dependencies from presets\extra_env\wan21_req.txt ...
echo.
python -m pip install -r "%ROOT%\presets\extra_env\wan21_req.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install WAN 2.1 Python requirements.
    echo.
    pause
    goto :end
)

REM Ensure models directory exists
if not exist "%ROOT%\models\wan21" (
    mkdir "%ROOT%\models\wan21"
)

echo.
echo Ensuring Hugging Face CLI is available...
python -m pip install "huggingface_hub[cli]>=0.26.0"
if errorlevel 1 (
    echo [ERROR] Failed to install huggingface_hub CLI tools.
    echo.
    pause
    goto :end
)

REM Prefer venv hf.exe if present
set "HFEXE=hf"
if exist "%ROOT%\.wan21_env\Scripts\hf.exe" (
    set "HFEXE=%ROOT%\.wan21_env\Scripts\hf.exe"
)

echo.
echo Downloading Wan-AI/Wan2.1-VACE-1.3B weights to models\wan21\Wan2.1-VACE-1.3B ...
echo (This is several GB and can take a long time on slow connections.)
echo.
"%HFEXE%" download Wan-AI/Wan2.1-VACE-1.3B --local-dir "%ROOT%\models\wan21\Wan2.1-VACE-1.3B"
if errorlevel 1 (
    echo [ERROR] Failed to download Wan2.1-VACE-1.3B from Hugging Face.
    echo Make sure your internet connection is OK and, if needed, configure HF mirrors or token.
    echo.
    pause
    goto :end
)

echo.
echo Downloading Wan-AI/Wan2.1-T2V-1.3B-Diffusers weights to models\wan21\Wan2.1-T2V-1.3B-Diffusers ...
echo (Used by the PySide6 text-to-video UI.)
echo.
"%HFEXE%" download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir "%ROOT%\models\wan21\Wan2.1-T2V-1.3B-Diffusers"
if errorlevel 1 (
    echo [ERROR] Failed to download Wan2.1-T2V-1.3B-Diffusers from Hugging Face.
    echo.
    pause
    goto :end
)

echo.
echo =====================================================
echo   WAN 2.1 environment is ready!
echo -----------------------------------------------------
echo   Virtual env : %ROOT%\.wan21_env
echo   Models path : %ROOT%\models\wan21
echo -----------------------------------------------------

echo.
echo =====================================================
echo.
pause

:end
endlocal
