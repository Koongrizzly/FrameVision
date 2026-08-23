@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
for %%I in ("%~dp0..\..") do set "FV_ROOT=%%~fI"
set "BIN_DIR=%FV_ROOT%\presets\bin"
set "UV_DIR=%BIN_DIR%\uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "LOG_DIR=%FV_ROOT%\logs"
set "LOG_FILE=%LOG_DIR%\ltx25_install.log"
set "NO_PAUSE=0"
set "ACTION="

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"
if /I "%~1"=="--w4a8" set "ACTION=install-w4a8"
if /I "%~1"=="--int4" set "ACTION=install-int4"
if /I "%~1"=="--fp16" set "ACTION=install-fp16"
if /I "%~1"=="--remove" set "ACTION=remove"
shift
goto parse

:parsed
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>nul
if not exist "%UV_DIR%" mkdir "%UV_DIR%" >nul 2>nul

if not defined ACTION (
  echo ============================================================
  echo FrameVision - LTX 2.5 Installer
  echo ============================================================
  echo.
  echo  1. W4A8 ConvRot  ^(recommended^)
  echo  2. INT4 ConvRot
  echo  3. Full FP16 / BF16 distilled
  echo  4. Remove an install
  echo  5. Cancel
  echo.
  set /p "CHOICE=Choose [1-5]: "
  if "!CHOICE!"=="5" exit /b 0
  if "!CHOICE!"=="1" set "ACTION=install-w4a8"
  if "!CHOICE!"=="2" set "ACTION=install-int4"
  if "!CHOICE!"=="3" set "ACTION=install-fp16"
  if "!CHOICE!"=="4" set "ACTION=remove"
)
if not defined ACTION (
  echo Invalid choice.
  if "%NO_PAUSE%"=="0" pause
  exit /b 2
)

if not exist "%UV_EXE%" (
  echo [SETUP] Installing portable uv into presets\bin\uv ...
  set "UV_ZIP=%TEMP%\framevision_uv.zip"
  del /q "!UV_ZIP!" >nul 2>nul
  curl.exe -L --fail --retry 5 --retry-delay 2 -o "!UV_ZIP!" "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" >>"%LOG_FILE%" 2>&1
  if errorlevel 1 goto fail
  tar.exe -xf "!UV_ZIP!" -C "%UV_DIR%" >>"%LOG_FILE%" 2>&1
  del /q "!UV_ZIP!" >nul 2>nul
  if not exist "%UV_EXE%" goto fail
)

for /f "delims=" %%P in ('"%UV_EXE%" python find 3.12 2^>nul') do set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE (
  echo [SETUP] Installing managed Python 3.12 ...
  "%UV_EXE%" python install 3.12 >>"%LOG_FILE%" 2>&1
  if errorlevel 1 goto fail
  for /f "delims=" %%P in ('"%UV_EXE%" python find 3.12 2^>nul') do set "PYTHON_EXE=%%P"
)
if not defined PYTHON_EXE goto fail

"%PYTHON_EXE%" "%~dp0ltx25_installer.py" --action "%ACTION%" --root "%FV_ROOT%" --uv "%UV_EXE%"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto failcode

echo.
echo [OK] Finished successfully.
if "%NO_PAUSE%"=="0" pause
exit /b 0

:fail
echo.
echo [ERROR] Installer setup failed. See:
echo %LOG_FILE%
if "%NO_PAUSE%"=="0" pause
exit /b 1

:failcode
echo.
echo [ERROR] Installer returned code %RC%. See:
echo %LOG_FILE%
if "%NO_PAUSE%"=="0" pause
exit /b %RC%
