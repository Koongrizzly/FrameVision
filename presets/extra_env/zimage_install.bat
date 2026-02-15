@echo off
setlocal ENABLEDELAYEDEXPANSION

echo ===============================================
echo   Z-Image-Turbo offline CUDA setup
echo   This will create .zimage_env,
echo   install CUDA PyTorch and deps,
echo   This does NOT download any models
echo   use their individual installer (gguf or fp16)
echo ===============================================
echo.

REM Get directory of this script (presets\extra_env)
set SCRIPT_DIR=%~dp0

REM Go to project root (two levels up)
cd /d "%SCRIPT_DIR%\..\.."

echo Working directory: %CD%
echo.

REM Create virtual environment if it does not exist yet
if not exist ".zimage_env\Scripts\python.exe" (
    echo Creating virtual environment in .zimage_env ...
    python -m venv ".zimage_env"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Make sure Python is installed and on PATH.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists. Skipping creation.
)

set PYTHON_EXEC=.zimage_env\Scripts\python.exe

echo.
echo Upgrading pip ...
"%PYTHON_EXEC%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo Installing CUDA-enabled PyTorch (cu121) ...
echo   (If this fails, check your NVIDIA driver and CUDA support.)
"%PYTHON_EXEC%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [ERROR] Failed to install CUDA PyTorch with cu121 wheels.
    echo You can try a different index-url from pytorch.org for your setup,
    echo or install torch manually inside .zimage_env.
    pause
    exit /b 1
)

echo.
echo Installing remaining Python dependencies from presets\extra_env\zimage_req.txt ...
"%PYTHON_EXEC%" -m pip install -r "presets\extra_env\zimage_req.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install Python packages.
    echo Tip: Make sure you have internet access for this step
    echo so packages and the model can be downloaded once.
    pause
    exit /b 1
)

echo.

echo.
echo =========================================
echo   Z-Image-Turbo CUDA setup complete!
echo   Environment: .zimage_env
echo   Model path:  models\Z-Image-Turbo
echo.
echo =========================================
echo.
pause
endlocal
