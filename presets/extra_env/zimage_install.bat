@echo off
setlocal ENABLEDELAYEDEXPANSION

echo =========================================
echo   Z-Image-Turbo FP8 offline CUDA setup
echo   This will create .zimage_env,
echo   install CUDA PyTorch and deps,
echo   and download the model into /models/.
echo =========================================
echo.

REM Get directory of this script (presets\extra_env)
set "SCRIPT_DIR=%~dp0"

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

set "PYTHON_EXEC=.zimage_env\Scripts\python.exe"

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
    echo [ERROR] Failed to install Python requirements.
    pause
    exit /b 1
)

echo.
echo [4/4] Downloading Z-Image-Turbo (full) into models\Z-Image-Turbo ...
echo   This may take a while on first run.

"%PYTHON_EXEC%" -c "from huggingface_hub import snapshot_download, hf_hub_download; from pathlib import Path; import os, glob, shutil; model_dir = Path(r'models/Z-Image-Turbo'); model_dir.mkdir(parents=True, exist_ok=True); snapshot_download(repo_id='Tongyi-MAI/Z-Image-Turbo', local_dir=model_dir, local_dir_use_symlinks=False); tdir = model_dir / 'transformer'; tdir.mkdir(parents=True, exist_ok=True); [os.remove(p) for p in glob.glob(str(tdir / 'diffusion_pytorch_model*.safetensors')) if os.path.exists(p)]; fp8_path = hf_hub_download(repo_id='T5B/Z-Image-Turbo-FP8', filename='z-image-turbo-fp8-e4m3fn.safetensors', local_dir=tdir, local_dir_use_symlinks=False); fp8 = Path(fp8_path); target = tdir / 'diffusion_pytorch_model.safetensors'; import os, shutil; os.remove(target) if target.exists() else None; shutil.move(str(fp8), str(target))"
if errorlevel 1 (
    echo [ERROR] Failed to download or prepare Z-Image-Turbo FP8 model.
    echo You may need to run this installer again when your connection is stable.
    pause
    exit /b 1
)

echo.
echo =========================================
echo   Z-Image-Turbo CUDA FP8 offline setup complete!
echo   Environment: .zimage_env
echo   Model path:  models\Z-Image-Turbo
echo =========================================
echo.
pause
endlocal
