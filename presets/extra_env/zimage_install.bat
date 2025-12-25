@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo =========================================
echo   Z-Image Optional Installer (Base)
echo -----------------------------------------
echo This installer installs the environment and all
echo needed dependencies
echo Pick one of the Z-Image model options in
echo the Optional Downloads menu list :
echo  - Full 16 FP model (safetensors)
echo  - Z-Image GGUF variants
echo -----------------------------------------
echo Windows prerequisite:
echo  - Microsoft Visual C++ Runtime (x64)
echo    will be installed automatically if
echo    missing, to avoid common DLL errors.
echo =========================================
echo.

REM Go to project root (this script lives in presets\extra_env)
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%\..\.."

if not exist "models" mkdir "models"

REM -----------------------------------------
REM Prereq: Visual C++ Redistributable (x64)
REM -----------------------------------------
call :ensure_vcredist_x64
if errorlevel 1 (
  echo.
  echo [WARN] Visual C++ Runtime install did not complete.
  echo       Z-Image GGUF/Turbo may fail to launch on this PC.
  echo.
)

REM -----------------------------------------
REM Extra guard: detect debug-built sd-cli.exe
REM (requires *D.dll like MSVCP140D.dll / UCRTBASED.dll)
REM Installing the regular VC++ redist will NOT fix that.
REM The right fix is shipping a Release build of sd-cli.exe.
REM -----------------------------------------
call :warn_if_sdcli_debug

echo.
echo [OK] Base step finished.
exit /b 0

:ensure_vcredist_x64
set "VCREDIST_OK=0"

REM Most common runtime files for MSVC 14.x (2015-2022)
if exist "%SystemRoot%\System32\vcruntime140.dll" set "VCREDIST_OK=1"
if exist "%SystemRoot%\System32\msvcp140.dll" set "VCREDIST_OK=1"

if "%VCREDIST_OK%"=="1" (
  REM Avoid parentheses in echo inside code blocks (CMD can mis-parse them)
  echo [OK] Visual C++ Runtime x64 already present.
  exit /b 0
)

echo [INFO] Visual C++ Runtime (x64) not found. Attempting install...

REM Requires admin rights on most systems
net session >nul 2>&1
if errorlevel 1 (
  echo [WARN] Not running as Administrator. The runtime installer may fail.
)

set "VCREDIST_URL=https://aka.ms/vs/17/release/vc_redist.x64.exe"
set "VCREDIST_EXE=%TEMP%\framevision_prereqs\vc_redist.x64.exe"
if not exist "%TEMP%\framevision_prereqs" mkdir "%TEMP%\framevision_prereqs" >nul 2>&1

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12;" ^
  "Invoke-WebRequest -Uri '%VCREDIST_URL%' -OutFile '%VCREDIST_EXE%';"
if errorlevel 1 (
  echo [ERROR] Could not download Visual C++ Runtime.
  echo         Please install it manually from:
  echo         https://aka.ms/vs/17/release/vc_redist.x64.exe
  exit /b 1
)

"%VCREDIST_EXE%" /install /quiet /norestart
set "RC=%ERRORLEVEL%"

if "%RC%"=="0" (
  echo [OK] Visual C++ Runtime installed.
  exit /b 0
) else (
  echo [ERROR] Visual C++ Runtime installer returned error code %RC%.
  echo         Please install it manually from:
  echo         https://aka.ms/vs/17/release/vc_redist.x64.exe
  exit /b 1
)

:warn_if_sdcli_debug
set "SDCLI_PATH="

REM Try a few common locations first (fast)
for %%P in (
  "sd-cli.exe"
  "bin\sd-cli.exe"
  "tools\sd-cli.exe"
  "tools\zimage\sd-cli.exe"
  "zimage\sd-cli.exe"
  "z-image\sd-cli.exe"
  "engines\sd-cli.exe"
) do (
  if not defined SDCLI_PATH if exist "%%~P" set "SDCLI_PATH=%%~P"
)

REM If not found, do a shallow-ish PowerShell search (skip huge folders where possible)
if not defined SDCLI_PATH (
  for /f "usebackq delims=" %%F in (`powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$root=(Get-Location).Path;" ^
    "$skip=@('venv','.venv','env','__pycache__','node_modules');" ^
    "$items=Get-ChildItem -Path $root -Directory -Force -ErrorAction SilentlyContinue | Where-Object { $skip -notcontains $_.Name };" ^
    "$found=$null;" ^
    "foreach($d in $items){ try{ $f=Get-ChildItem -Path $d.FullName -Filter 'sd-cli.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1; if($f){$found=$f.FullName; break} } catch{} }" ^
    "if(-not $found){ try{ $f=Get-ChildItem -Path $root -Filter 'sd-cli.exe' -ErrorAction SilentlyContinue | Select-Object -First 1; if($f){$found=$f.FullName} } catch{} }" ^
    "if($found){ $rel = Resolve-Path -LiteralPath $found; Write-Output $rel }"` ) do (
    set "SDCLI_PATH=%%F"
  )
)

if not defined SDCLI_PATH (
  REM Nothing to warn about if sd-cli isn't present yet
  exit /b 0
)

REM Check the binary for debug CRT dependency strings without running it
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p='%SDCLI_PATH%';" ^
  "if(-not (Test-Path -LiteralPath $p)){ exit 0 }" ^
  "$b=[IO.File]::ReadAllBytes($p);" ^
  "$s=[Text.Encoding]::ASCII.GetString($b);" ^
  "if($s -match 'MSVCP140D\.dll' -or $s -match 'VCRUNTIME140D\.dll' -or $s -match 'VCRUNTIME140_1D\.dll' -or $s -match 'UCRTBASED\.dll'){ exit 2 } else { exit 0 }"
set "DBGRC=%ERRORLEVEL%"

if "%DBGRC%"=="2" (
  echo.
  echo [IMPORTANT] Detected sd-cli.exe that depends on DEBUG MSVC DLLs.
  echo             Example errors: MSVCP140D.dll / VCRUNTIME140D.dll / UCRTBASED.dll missing.
  echo.
  echo             Installing Visual Studio is NOT a reliable fix for end users,
  echo             because debug runtimes are not redistributed system-wide.
  echo.
  echo             Recommended fix:
  echo             - Ship a RELEASE build of sd-cli.exe (no *D.dll dependencies),
  echo               then the VC++ Runtime (installed above) is sufficient.
  echo.
  echo             If you still want to install Visual Studio Build Tools, start here:
  echo             https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio
  echo.
)
exit /b 0
