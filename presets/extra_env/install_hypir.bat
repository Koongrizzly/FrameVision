@echo off
setlocal EnableExtensions
chcp 65001 >nul

title FrameVision - Install HYPIR

rem This installer lives in FrameVision\presets\extra_env.
set "EXTRA=%~dp0"
for %%I in ("%EXTRA%\..\..") do set "ROOT=%%~fI"
set "ENV=%ROOT%\environments\.hypir"
set "ENV_PY=%ENV%\python.exe"
set "BOOT=%EXTRA%_miniforge"
set "CONDA=%BOOT%\Scripts\conda.exe"
set "MINIFORGE_EXE=%EXTRA%Miniforge3-Windows-x86_64.exe"
set "INSTALL_PY=%EXTRA%install_hypir.py"

echo ============================================================================
echo FrameVision HYPIR installer
echo ============================================================================
echo Root:        %ROOT%
echo Environment: %ENV%
echo Installer:   %EXTRA%
echo.

if not exist "%INSTALL_PY%" (
  echo [ERROR] Missing installer helper: %INSTALL_PY%
  goto :fail
)

if not exist "%ROOT%\environments" mkdir "%ROOT%\environments"

if not exist "%CONDA%" (
  echo [1/6] Installing private Miniforge bootstrap under presets\extra_env...
  if not exist "%MINIFORGE_EXE%" (
    echo Downloading Miniforge...
    curl.exe -L --fail --retry 5 --retry-delay 3 -o "%MINIFORGE_EXE%" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
    if errorlevel 1 goto :fail
  )
  start /wait "" "%MINIFORGE_EXE%" /S /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /NoRegistry=1 /D=%BOOT%
  if errorlevel 1 goto :fail
)

if not exist "%CONDA%" (
  echo [ERROR] Miniforge did not install correctly: %CONDA%
  goto :fail
)

if not exist "%ENV_PY%" (
  echo [2/6] Creating isolated Python 3.10 environment...
  "%CONDA%" create -y -p "%ENV%" python=3.10 pip
  if errorlevel 1 goto :fail
) else (
  echo [2/6] HYPIR environment already exists - reusing it.
)

echo [3/6] Updating pip tools...
"%ENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo [4/6] Installing CUDA PyTorch 2.6.0 / torchvision 0.21.0...
"%ENV_PY%" -m pip install --upgrade torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto :fail

echo [5/6] Installing official HYPIR dependencies...
"%ENV_PY%" -m pip install accelerate==1.4.0 diffusers==0.32.2 gradio==5.21.0 lpips==0.1.4 open_clip_torch==2.31.0 openai==1.96.1 polars==1.24.0 tenacity==9.1.2 tensorboard==2.19.0 vision-aided-loss==0.1.0 omegaconf==2.3.0 python-dotenv==1.1.1 transformers==4.49.0 peft==0.14.0 einops==0.8.1 pydantic==2.10.6 opencv-python-headless==4.11.0.86 timm==1.0.15 huggingface_hub
if errorlevel 1 goto :fail

echo [6/6] Downloading HYPIR repo, HYPIR weights, and local SD2.1 base model...
"%ENV_PY%" "%INSTALL_PY%"
if errorlevel 1 goto :fail

echo.
echo Verifying CUDA...
"%ENV_PY%" -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
if errorlevel 1 goto :fail

echo.
echo ============================================================================
echo [OK] HYPIR installation complete.
echo Environment: %ENV%
echo Repository:  %EXTRA%hypir_src\HYPIR
echo Models:      %ROOT%\models\hypir
echo ============================================================================
echo You can close this window and use HYPIR in FrameVision.
pause
exit /b 0

:fail
echo.
echo ============================================================================
echo [ERROR] HYPIR installation failed.
echo Scroll up to the first error message. Existing downloads are kept so the
rem next run can resume/reuse them.
echo next run can resume/reuse them.
echo ============================================================================
pause
exit /b 1
