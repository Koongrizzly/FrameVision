@echo off
setlocal EnableExtensions
chcp 65001 >nul

rem ============================================================
rem LTX-2.5 DISTILLED - FrameVision UV installer
rem Layout:
rem   helpers\ltx25_helper.py
rem   models\ltx-2.5\LTX-2
rem   environments\ltx25
rem   models\ltx-2.5
rem   models\ltx-2.5\cache
rem   presets\setsave\ltx25.json
rem   logs\
rem ============================================================

cd /d "%~dp0"
for %%I in ("%CD%\..\..") do set "ROOT=%%~fI"
set "EXTRA=%ROOT%\presets\extra_env"
set "ENV=%ROOT%\environments\ltx25"
set "ENVS=%ROOT%\environments"
set "MODELS=%ROOT%\models\ltx-2.5"
set "REPO=%MODELS%\LTX-2"
set "CACHE=%MODELS%\cache"
set "TOOLS=%EXTRA%\tools"
set "UVDIR=%TOOLS%\uv"
set "UVEXE=%UVDIR%\uv.exe"
set "DOWNLOADS=%CACHE%\downloads"
set "UVZIP=%DOWNLOADS%\uv_windows.zip"
set "REPOZIP=%DOWNLOADS%\ltx2_main.zip"

rem UV environment and all persistent caches live outside the source repo.
set "UV_PROJECT_ENVIRONMENT=%ENV%"
set "UV_PYTHON_INSTALL_DIR=%ENVS%\uv-python"
set "UV_CACHE_DIR=%CACHE%\uv"
set "PIP_CACHE_DIR=%CACHE%\pip"
set "HF_HOME=%CACHE%\huggingface"
set "HF_HUB_CACHE=%CACHE%\huggingface\hub"
set "HF_XET_CACHE=%CACHE%\huggingface\xet"
set "TORCH_EXTENSIONS_DIR=%CACHE%\torch_extensions"
set "TRITON_CACHE_DIR=%CACHE%\triton"
set "XDG_CACHE_HOME=%CACHE%"
set "UV_NO_PROGRESS=0"
set "HF_XET_HIGH_PERFORMANCE=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

if not exist "%EXTRA%" mkdir "%EXTRA%"
if not exist "%ENVS%" mkdir "%ENVS%"
if not exist "%MODELS%" mkdir "%MODELS%"
if not exist "%CACHE%" mkdir "%CACHE%"
if not exist "%DOWNLOADS%" mkdir "%DOWNLOADS%"
if not exist "%TOOLS%" mkdir "%TOOLS%"
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"
if not exist "%ROOT%\presets\setsave" mkdir "%ROOT%\presets\setsave"

call :banner

where curl.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Windows curl.exe was not found.
    goto :fail
)
where tar.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Windows tar.exe was not found.
    goto :fail
)

rem ------------------------------------------------------------
rem 1. Bootstrap UV.
rem ------------------------------------------------------------
if exist "%UVEXE%" (
    echo [OK] UV already present: "%UVEXE%"
) else (
    echo.
    echo [1/6] Downloading UV...
    if not exist "%UVDIR%" mkdir "%UVDIR%"
    del /q "%UVZIP%" >nul 2>&1
    curl.exe -L --fail --retry 8 --retry-delay 3 --connect-timeout 30 ^
      -o "%UVZIP%" ^
      "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    if errorlevel 1 goto :uv_download_fail
    tar.exe -xf "%UVZIP%" -C "%UVDIR%"
    if errorlevel 1 goto :uv_extract_fail
    del /q "%UVZIP%" >nul 2>&1
    if not exist "%UVEXE%" goto :uv_extract_fail
    echo [OK] UV installed.
)
"%UVEXE%" --version
if errorlevel 1 goto :fail

rem ------------------------------------------------------------
rem 2. Official LTX-2 source under presets\extra_env.
rem ------------------------------------------------------------
if exist "%REPO%\pyproject.toml" (
    echo.
    echo [2/6] LTX-2 source already present - keeping local checkout.
) else (
    echo.
    echo [2/6] Downloading official Lightricks LTX-2 source...
    if exist "%REPO%" rmdir /s /q "%REPO%"
    if exist "%MODELS%\LTX-2-main" rmdir /s /q "%MODELS%\LTX-2-main"
    del /q "%REPOZIP%" >nul 2>&1
    curl.exe -L --fail --retry 8 --retry-delay 3 --connect-timeout 30 ^
      -o "%REPOZIP%" ^
      "https://github.com/Lightricks/LTX-2/archive/refs/heads/main.zip"
    if errorlevel 1 goto :repo_download_fail
    tar.exe -xf "%REPOZIP%" -C "%MODELS%"
    if errorlevel 1 goto :repo_extract_fail
    del /q "%REPOZIP%" >nul 2>&1
    if not exist "%MODELS%\LTX-2-main\pyproject.toml" goto :repo_extract_fail
    move "%MODELS%\LTX-2-main" "%REPO%" >nul
    if errorlevel 1 goto :repo_extract_fail
    echo [OK] LTX-2 source installed.
)

rem ------------------------------------------------------------
rem 3. UV-managed Python and environment under environments\.
rem ------------------------------------------------------------
echo.
echo [3/6] Preparing UV-managed Python 3.12...
"%UVEXE%" python install 3.12
if errorlevel 1 goto :python_fail

echo.
echo [4/6] Installing LTX-2 dependencies into "%ENV%"...
pushd "%REPO%"
"%UVEXE%" sync --python 3.12
if errorlevel 1 (
    popd
    goto :sync_fail
)
popd

rem ------------------------------------------------------------
rem 4. Windows Triton + SageAttention + PySide6.
rem ------------------------------------------------------------
echo.
echo [5/6] Installing/checking Triton-Windows, SageAttention and PySide6...

"%ENV%\Scripts\python.exe" -c "import triton" >nul 2>&1
if errorlevel 1 (
    echo Installing Triton-Windows...
    "%UVEXE%" pip install --python "%ENV%\Scripts\python.exe" "triton-windows<3.7"
    if errorlevel 1 goto :triton_fail
) else (
    echo [OK] Triton already installed.
)

"%ENV%\Scripts\python.exe" -c "import sageattention" >nul 2>&1
if errorlevel 1 (
    echo Installing SageAttention 2.2.0 post6...
    "%UVEXE%" pip install --python "%ENV%\Scripts\python.exe" ^
      "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl"
    if errorlevel 1 goto :sage_fail
) else (
    echo [OK] SageAttention already installed.
)

"%ENV%\Scripts\python.exe" -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo Installing PySide6 for the helper GUI...
    "%UVEXE%" pip install --python "%ENV%\Scripts\python.exe" PySide6
    if errorlevel 1 goto :pyside_fail
) else (
    echo [OK] PySide6 already installed.
)

"%ENV%\Scripts\python.exe" -c "import torch, triton, sageattention; from importlib.metadata import version; print('[OK] Torch:', torch.__version__); print('[OK] CUDA:', torch.version.cuda); print('[OK] Triton:', triton.__version__); print('[OK] SageAttention:', version('sageattention'))"
if errorlevel 1 goto :sage_verify_fail

rem ------------------------------------------------------------
rem 5. Hugging Face authentication.
rem ------------------------------------------------------------
echo.
echo [6/6] Checking Hugging Face access...
"%ENV%\Scripts\python.exe" -m huggingface_hub.commands.huggingface_cli whoami >nul 2>&1
if errorlevel 1 (
    "%ENV%\Scripts\hf.exe" auth whoami >nul 2>&1
)
if errorlevel 1 (
    echo Hugging Face login is required for gated LTX-2.5 files.
    echo You must already have accepted the model terms.
    "%ENV%\Scripts\hf.exe" auth login
    if errorlevel 1 goto :hf_auth_fail
) else (
    echo [OK] Existing Hugging Face login found.
)

rem ------------------------------------------------------------
rem 6. Distilled runtime weights only.
rem ------------------------------------------------------------
echo.
echo ============================================================
echo Downloading LTX-2.5 DISTILLED runtime files only
echo Existing files are reused and interrupted downloads resume.
echo Cache: "%CACHE%"
echo ============================================================
echo.

"%ENV%\Scripts\hf.exe" download Lightricks/LTX-2.5 ^
  diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors ^
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors ^
  vae/ltx-2.5-video-vae-bf16.safetensors ^
  vae/ltx-2.5-audio-vae-bf16.safetensors ^
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors ^
  --local-dir "%MODELS%"
if errorlevel 1 goto :model_download_fail

call :verify_models
if errorlevel 1 goto :fail
call :write_paths

echo.
echo ============================================================
echo [DONE] LTX-2.5 distilled installation completed successfully.
echo Repo   : "%REPO%"
echo Env    : "%ENV%"
echo Models : "%MODELS%"
echo Cache  : "%CACHE%"
echo ============================================================
echo.
goto :success

:verify_models
set "MISSING=0"
call :check_file "%MODELS%\diffusion_models\ltx-2.5-22b-distilled-transformer-bf16.safetensors"
call :check_file "%MODELS%\text_encoders\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
call :check_file "%MODELS%\vae\ltx-2.5-video-vae-bf16.safetensors"
call :check_file "%MODELS%\vae\ltx-2.5-audio-vae-bf16.safetensors"
call :check_file "%MODELS%\latent_upscale_models\ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
if "%MISSING%"=="1" exit /b 1
exit /b 0

:check_file
if not exist "%~1" (
    echo [ERROR] Missing required file: %~1
    set "MISSING=1"
) else (
    echo [OK] %~nx1
)
exit /b 0

:write_paths
> "%ROOT%\presets\setsave\ltx_2_5_paths.txt" echo LTX_REPO=%REPO%
>>"%ROOT%\presets\setsave\ltx_2_5_paths.txt" echo LTX_MODELS=%MODELS%
>>"%ROOT%\presets\setsave\ltx_2_5_paths.txt" echo LTX_VENV=%ENV%
>>"%ROOT%\presets\setsave\ltx_2_5_paths.txt" echo LTX_CACHE=%CACHE%
>>"%ROOT%\presets\setsave\ltx_2_5_paths.txt" echo TRANSFORMER=%MODELS%\diffusion_models\ltx-2.5-22b-distilled-transformer-bf16.safetensors
>>"%ROOT%\presets\setsave\ltx_2_5_paths.txt" echo TEXT_ENCODER=%MODELS%\text_encoders\gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
>>"%ROOT%\presets\setsave\ltx_2_5_paths.txt" echo VIDEO_VAE=%MODELS%\vae\ltx-2.5-video-vae-bf16.safetensors
>>"%ROOT%\presets\setsave\ltx_2_5_paths.txt" echo AUDIO_VAE=%MODELS%\vae\ltx-2.5-audio-vae-bf16.safetensors
>>"%ROOT%\presets\setsave\ltx_2_5_paths.txt" echo SPATIAL_UPSAMPLER=%MODELS%\latent_upscale_models\ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
exit /b 0

:banner
echo ============================================================
echo LTX-2.5 DISTILLED - FrameVision UV installer
echo ============================================================
echo Root   : "%ROOT%"
echo Repo   : "%REPO%"
echo Env    : "%ENV%"
echo Models : "%MODELS%"
echo Cache  : "%CACHE%"
echo ============================================================
exit /b 0

:uv_download_fail
echo [ERROR] UV download failed.& goto :fail
:uv_extract_fail
echo [ERROR] UV extraction failed.& goto :fail
:repo_download_fail
echo [ERROR] LTX-2 source download failed.& goto :fail
:repo_extract_fail
echo [ERROR] LTX-2 source extraction failed.& goto :fail
:python_fail
echo [ERROR] UV Python installation failed.& goto :fail
:sync_fail
echo [ERROR] LTX dependency installation failed.& goto :fail
:triton_fail
echo [ERROR] Triton-Windows installation failed.& goto :fail
:sage_fail
echo [ERROR] SageAttention installation failed.& goto :fail
:pyside_fail
echo [ERROR] PySide6 installation failed.& goto :fail
:sage_verify_fail
echo [ERROR] Triton/SageAttention verification failed.& goto :fail
:hf_auth_fail
echo [ERROR] Hugging Face authentication failed.& goto :fail
:model_download_fail
echo [ERROR] LTX model download failed.& goto :fail

:fail
echo.
echo Installation failed. See the messages above.
if /i "%~1"=="--no-pause" exit /b 1
pause
exit /b 1

:success
if /i "%~1"=="--no-pause" exit /b 0
pause
exit /b 0
