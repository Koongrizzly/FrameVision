@echo off
setlocal ENABLEDELAYEDEXPANSION

echo =========================================
echo   Z-Image Turbo FP16 model downloader
echo   - Downloads Diffusers FP16 model into: models\Z-Image-Turbo
echo   - Requires existing environment: .zimage_env
echo =========================================
echo.

REM Get directory of this script (presets\extra_env)
set SCRIPT_DIR=%~dp0

REM Go to project root (two levels up)
cd /d "%SCRIPT_DIR%\..\.."

echo Working directory: %CD%
echo.

set PYTHON_EXEC=.zimage_env\Scripts\python.exe

if not exist "%PYTHON_EXEC%" (
    echo [ERROR] Missing .zimage_env
    echo Run presets\extra_env\zimage_install.bat first to create the environment.
    echo.
    pause
    exit /b 1
)

REM Ensure huggingface_hub is available for snapshot_download
"%PYTHON_EXEC%" -c "import huggingface_hub" >nul 2>&1
if errorlevel 1 (
    echo Installing downloader dependency: huggingface_hub ...
    "%PYTHON_EXEC%" -m pip install --upgrade huggingface_hub
    if errorlevel 1 (
        echo [ERROR] Failed to install huggingface_hub in .zimage_env
        pause
        exit /b 1
    )
)

echo.
echo Downloading Z-Image Turbo FP16 model into models\Z-Image-Turbo ...
echo.

"%PYTHON_EXEC%" -c "from huggingface_hub import snapshot_download; from pathlib import Path; p = Path('models') / 'Z-Image-Turbo'; p.mkdir(parents=True, exist_ok=True); snapshot_download('Tongyi-MAI/Z-Image-Turbo', local_dir=str(p), local_dir_use_symlinks=False)"
if errorlevel 1 (
    echo [ERROR] Failed to download Z-Image Turbo FP16 model.
    echo Re-run this installer when your connection is stable.
    pause
    exit /b 1
)

echo.
echo Verifying model_index.json ...
"%PYTHON_EXEC%" -c "from pathlib import Path; import sys; sys.exit(0 if (Path('models')/'Z-Image-Turbo'/'model_index.json').exists() else 1)" >nul 2>&1
if errorlevel 1 (
    echo [WARN] model_index.json not found at models\Z-Image-Turbo
    echo This usually indicates an incomplete download.
    echo Try deleting models\Z-Image-Turbo and running this installer again.
) else (
    echo OK - model_index.json found.
)

echo.
echo =========================================
echo   Z-Image Turbo FP16 model download complete
echo   Model path: models\Z-Image-Turbo
echo =========================================
echo.
pause
endlocal
