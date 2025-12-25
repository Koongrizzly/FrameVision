@echo off
setlocal EnableExtensions

REM ===========================
REM  HunyuanVideo-1.5 Installer
REM  CUDA-only (RTX) setup
REM ===========================

REM Resolve root (two levels up from this .bat)
set "THIS_DIR=%~dp0"
for %%I in ("%THIS_DIR%..") do set "ROOT=%%~fI"
for %%I in ("%ROOT%\..") do set "ROOT=%%~fI"

set "VENV_DIR=%ROOT%\.hunyuan15_env"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=%ROOT%\presets\extra_env\hunyuan15_req.txt"

REM Default CUDA wheel index (override by setting PYTORCH_CUDA=cu124 / cu121 / cu118)
if "%PYTORCH_CUDA%"=="" set "PYTORCH_CUDA=cu124"

echo.
echo ===========================
echo  HunyuanVideo-1.5 Installer
echo ===========================
echo Root: "%ROOT%"
echo Venv: "%VENV_DIR%"
echo Torch CUDA wheels: %PYTORCH_CUDA%
echo Flash attention
echo.

REM --- Basic checks
if not exist "%ROOT%" (
  echo [ERROR] Root folder not found: "%ROOT%"
  exit /b 1
)
if not exist "%REQ_FILE%" (
  echo [ERROR] Requirements file not found: "%REQ_FILE%"
  exit /b 1
)

REM --- Create venv if missing
if not exist "%VENV_PY%" (
  echo [1/6] Creating venv...
  py -3 -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create venv. Make sure Python 3 is installed.
    exit /b 1
  )
)

echo [2/6] Upgrading pip/setuptools/wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip tools.
  exit /b 1
)

REM --- Install CUDA torch (pinned for stability + FlashAttention wheel compatibility)
echo [3/7] Installing CUDA PyTorch ^(NOT CPU^)...
"%VENV_PY%" -m pip install --upgrade --force-reinstall --index-url "https://download.pytorch.org/whl/%PYTORCH_CUDA%" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
if errorlevel 1 (
  echo [ERROR] Failed to install CUDA PyTorch wheels.
  echo        Try setting PYTORCH_CUDA=cu124 ^(or cu118^) and re-run.
  exit /b 1
)

REM --- Guard: refuse CPU torch
"%VENV_PY%" -c "import torch; import sys; ok=torch.cuda.is_available(); print('torch', torch.__version__); print('cuda_available', ok); sys.exit(0 if ok else 2)"
if errorlevel 2 (
  echo [ERROR] Torch installed but CUDA is NOT available.
  echo        This installer is CUDA-only. Fix NVIDIA driver / CUDA runtime and re-run.
  exit /b 1
)

REM --- Optional: kernels attention backends (Diffusers)
REM Set HY15_NO_KERNELS=1 to skip.
echo [4/7] Optional speedup: kernels attention backends...
if /i "%HY15_NO_KERNELS%"=="1" (
  echo       Skipping kernels ^(HY15_NO_KERNELS=1^)
) else (
  echo       Installing "kernels" package for prebuilt attention kernels...
  "%VENV_PY%" -m pip install --upgrade kernels
  if errorlevel 1 (
    echo       [WARN] Failed to install kernels. Attention backend may stay "default".
  ) else (
    echo       kernels installed.
    echo       NOTE: First use of flash_hub/flash_varlen_hub/sage_hub may download prebuilt kernels on first run.
  )
)

REM --- Optional: FlashAttention (prebuilt wheel for Windows)
REM Set HY15_NO_FLASH=1 to skip.
echo [4/7] Optional speedup: FlashAttention...
if /i "%HY15_NO_FLASH%"=="1" (
  echo       Skipping FlashAttention ^(HY15_NO_FLASH=1^)
) else (
  REM If it's already importable, don't reinstall.
  "%VENV_PY%" -c "import flash_attn,sys; print('flash_attn', getattr(flash_attn,'__version__','unknown')); sys.exit(0)" >nul 2>nul
  if not errorlevel 1 (
    echo       flash_attn already installed.
  ) else (
    REM Only auto-install the known-good prebuilt wheel on: Python 3.11 + torch 2.5.1+cu124 + cu124
    "%VENV_PY%" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 2)" >nul 2>nul
    if errorlevel 2 (
      echo       Detected: py!=3.11  ^(skipping auto-install^)
    ) else (
      "%VENV_PY%" -c "import torch,sys; sys.exit(0 if getattr(torch,'__version__','')=='2.5.1+cu124' else 2)" >nul 2>nul
      if errorlevel 2 (
        echo       Detected: torch!=2.5.1+cu124  ^(skipping auto-install^)
      ) else (
        if /i "%PYTORCH_CUDA%"=="cu124" (
          echo       Attempting FlashAttention wheel install ^(py311 + torch 2.5.1+cu124 + cu124^)...
          "%VENV_PY%" -m pip install --no-deps --force-reinstall "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.0.post2+cu124torch2.5.1cxx11abiFALSE-cp311-cp311-win_amd64.whl"
          if errorlevel 1 (
            echo       [WARN] FlashAttention wheel install failed.
          ) else (
            echo       flash_attn installed.
          )
        ) else (
          echo       Not auto-installing FlashAttention for this combo.
          echo       Required for auto-install: PYTORCH_CUDA=cu124, Python 3.11, torch 2.5.1+cu124
        )
      )
    )
  )
)

echo [5/7] Installing Python requirements...
"%VENV_PY%" -m pip install --upgrade -r "%REQ_FILE%"
if errorlevel 1 (
  echo [ERROR] Failed to install Python requirements.
  exit /b 1
)

REM --- Verify key imports used by the CLI
echo [6/7] Verifying imports...
"%VENV_PY%" -c "import diffusers, huggingface_hub; print('diffusers', diffusers.__version__); print('huggingface_hub', huggingface_hub.__version__)"
if errorlevel 1 (
  echo [ERROR] Installed, but key modules still fail to import.
  echo        Check antivirus / proxy / pip logs above and re-run.
  exit /b 1
)

REM --- Download default model (distilled 480p) unless explicitly skipped
if /i "%SKIP_HUNYUAN15_MODEL%"=="1" (
  echo [7/7] Skipping model download ^(SKIP_HUNYUAN15_MODEL=1^)
) else (
  echo [7/7] Downloading default model ^(480p distilled^)...
  "%VENV_PY%" "%ROOT%\helpers\hunyuan15_cli.py" download --model 480p_t2v_distilled
  if errorlevel 1 (
    echo [ERROR] Model download failed.
    echo        You can retry later with:
    echo          "%VENV_PY%" "%ROOT%\helpers\hunyuan15_cli.py" download --model 480p_t2v_distilled
    exit /b 1
  )
)


echo.
echo [OK] Environment ready.
"%VENV_PY%" -c "import importlib.util as u; print('flash_attn_installed', bool(u.find_spec('flash_attn')))" 2>nul
echo You can now run:
echo   "%VENV_PY%" "%ROOT%\helpers\hunyuan15_cli.py" download --model 480p_t2v_distilled
echo   "%VENV_PY%" "%ROOT%\helpers\hunyuan15_cli.py" generate --model 480p_t2v_distilled --prompt "..."
echo.
exit /b 0
