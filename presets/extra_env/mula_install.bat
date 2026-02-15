@echo off
setlocal

REM HeartMuLa one-click installer (CUDA + pinned deps)
REM PATCH v2: model download step enabled + portable HF cache

cd /d "%~dp0\..\.."
set "ROOT=%cd%"
set "ENV=%ROOT%\.mula_env"

echo.
echo [HeartMuLa] Root: %ROOT%
echo [HeartMuLa] Installer: mula_install.bat (PATCH v2)

REM Track which CUDA wheel tag we successfully installed (cu121 default, fallback cu118)
set "CUDA_TAG=cu121"

REM Create venv (prefer py -3.10 if available)
if not exist "%ENV%\Scripts\python.exe" (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.10 -m venv "%ENV%" 2>nul
  )
)
if not exist "%ENV%\Scripts\python.exe" (
  python -m venv "%ENV%"
)

set "PY=%ENV%\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] Failed to create virtual environment at %ENV%
  pause
  exit /b 1
)

"%PY%" -m pip install --upgrade pip setuptools wheel

echo.
echo [HeartMuLa] Installing CUDA PyTorch pinned to heartlib...
"%PY%" -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
"%PY%" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121
if errorlevel 1 (
  echo [WARN] cu121 wheels failed. Trying cu118...
  set "CUDA_TAG=cu118"
  "%PY%" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118
)

if errorlevel 1 (
  echo [ERROR] PyTorch CUDA install failed.
  pause
  exit /b 1
)

echo.
echo [HeartMuLa] Installing remaining dependencies...
REM IMPORTANT: do NOT use --upgrade here. It can replace CUDA torch with a newer CPU wheel.
REM Also ensure nothing can pull a different torch build.
set "CONSTRAINTS=%ROOT%\presets\extra_env\_mula_constraints.txt"
(
  echo torch==2.4.1+%CUDA_TAG%
  echo torchvision==0.19.1+%CUDA_TAG%
  echo torchaudio==2.4.1+%CUDA_TAG%
) > "%CONSTRAINTS%"

"%PY%" -m pip install --no-cache-dir -r "%ROOT%\presets\extra_env\mula_req.txt" -c "%CONSTRAINTS%"
if errorlevel 1 (
  echo [ERROR] Dependency install failed.
  pause
  exit /b 1
)

echo.
echo [HeartMuLa] Downloading models (this can take a while)...
set "MODELS_DIR=%ROOT%\models\HeartMuLa"
set "HF_HOME=%MODELS_DIR%\_hf_home"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "TRANSFORMERS_CACHE=%HF_HOME%\transformers"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

REM Optional: if Xet causes issues on some networks, you can force HTTP fallback.
REM set "HF_HUB_DISABLE_XET=1"

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "%ROOT%\presets\extra_env\mula_download_models.py" (
  echo [ERROR] Missing download script: %ROOT%\presets\extra_env\mula_download_models.py
  pause
  exit /b 1
)

echo   HF_HOME: %HF_HOME%
if not "%HTTPS_PROXY%"=="" echo   HTTPS_PROXY: %HTTPS_PROXY%
if not "%HTTP_PROXY%"=="" echo   HTTP_PROXY: %HTTP_PROXY%

"%PY%" "%ROOT%\presets\extra_env\mula_download_models.py" "%MODELS_DIR%"
if errorlevel 1 (
  echo [ERROR] Model download failed.
  pause
  exit /b 1
)

echo.
echo [HeartMuLa] Ensuring heartlib source is present...
set "HEARTLIB_DIR=%MODELS_DIR%\_heartlib_src"

REM The HF repos contain weights/configs, but the example script lives in the GitHub heartlib repo.
REM We fetch it on-demand so the app can run examples without manual steps.
if not exist "%HEARTLIB_DIR%\examples\run_music_generation.py" (
  echo   heartlib missing -> fetching...
  if exist "%HEARTLIB_DIR%" rmdir /s /q "%HEARTLIB_DIR%" 2>nul
  mkdir "%HEARTLIB_DIR%" 2>nul

  where git >nul 2>nul
  if not errorlevel 1 (
    echo   using git clone...
    git clone --depth 1 https://github.com/HeartMuLa/heartlib "%HEARTLIB_DIR%"
  ) else (
    echo   git not found -> using PowerShell zip download...
    set "ZIP=%TEMP%\heartlib_main.zip"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; iwr -UseBasicParsing https://github.com/HeartMuLa/heartlib/archive/refs/heads/main.zip -OutFile '%TEMP%\\heartlib_main.zip'; Expand-Archive -Force '%TEMP%\\heartlib_main.zip' '%TEMP%\\heartlib_zip'; Copy-Item -Force -Recurse '%TEMP%\\heartlib_zip\\heartlib-main\\*' '%HEARTLIB_DIR%'; Remove-Item -Force '%TEMP%\\heartlib_main.zip'; Remove-Item -Force -Recurse '%TEMP%\\heartlib_zip'; exit 0 } catch { Write-Host $_; exit 1 }"
  )

  if errorlevel 1 (
    echo [ERROR] Failed to fetch heartlib source.
    pause
    exit /b 1
  )
)

echo.
echo [HeartMuLa] Installing heartlib (editable) if source exists...
if exist "%ROOT%\models\HeartMuLa\_heartlib_src\pyproject.toml" (
  REM heartlib pins torch==2.4.1 (CPU on PyPI). We already installed CUDA torch.
  REM Install heartlib without dependencies to avoid downgrading torch.
  "%PY%" -m pip install --no-cache-dir -e "%ROOT%\models\HeartMuLa\_heartlib_src" --no-deps
)

echo.
echo [HeartMuLa] Verifying CUDA is enabled in torch...
"%PY%" -c "import torch; print('torch:', torch.__version__); print('torch.cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"

echo.
echo [OK] Done.
echo Env: %ENV%
echo Models: %MODELS_DIR%
echo UI: %PY% %ROOT%\helpers\heartmula.py
echo.
pause
exit /b 0
