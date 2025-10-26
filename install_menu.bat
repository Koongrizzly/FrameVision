@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"

:menu
REM ===== [FRAMEVISION TOOL LINK - DO NOT REMOVE] =====
REM Create a 'bin' junction that points to 'presets\bin' so legacy code
REM looking for ROOT\bin\ffmpeg.exe etc. still works without duplicating files.

set "TOOLREAL=%ROOT%presets\bin"
set "TOOLLINK=%ROOT%bin"

if exist "%TOOLREAL%" (
    REM If there's already a real folder at bin (like leftover copied EXEs), leave it.
    REM If it does NOT exist but a link is needed, create a junction.
    if not exist "%TOOLLINK%" (
        REM mklink /J requires admin (or Developer Mode). It creates a directory junction.
        mklink /J "%TOOLLINK%" "%TOOLREAL%"
    )
)
REM ===== [END FRAMEVISION TOOL LINK] =====
rem --- initialize ANSI colors (works on Windows 10+ / Windows Terminal) ---
for /F "delims=#" %%A in ('"prompt #$E# & for %%B in (1) do rem"') do set "ESC=%%A"
set "RST=%ESC%[0m"
set "BOLD=%ESC%[1m"
set "DIM=%ESC%[2m"
set "TITLE=%ESC%[95m"
set "ITEM=%ESC%[96m"
set "NOTE=%ESC%[93m"
set "GRAY=%ESC%[90m"
set "BLUE=%ESC%[34m"
set "GREEN=%ESC%[32m"

if defined _SKIPCLS (set "_SKIPCLS=") else (cls)
echo.
echo %TITLE%==============================================%RST%
echo %BOLD%            FrameVision Installer%RST%
echo %TITLE%==============================================%RST%
echo.
echo %ITEM% 1^)  Check requirements and disk space
echo %RST%      Runs preflight checks (Python, Git, FFmpeg, GPU/driver)
echo %RST%      and estimates storage needed. 
echo %BLUE%      Can create Venv and installs Python on path
echo.
echo %ITEM% 2^)  Core install %GRAY%(app only)%RST%
echo %RST%      Creates/updates .venv and installs FrameVision without
echo %RST%      ML dependencies or model downloads — fastest setup.
echo %BLUE%      Good for a quick repair when the app no longer starts.
echo. 
echo %ITEM% 3^)  Full install %GRAY%(CPU / non-CUDA)
echo %RST%      Installs the app plus CPU-only ML dependencies.
echo %BLUE%      Works on any machine; use this up to 6gig of Vram.
echo.
echo %ITEM% 4^)  Full install %GRAY%(CUDA GPU)
echo %RST%      Installs the app plus CUDA-enabled ML dependencies
echo %BLUE%      for NVIDIA GPUs. 8 gig Vram or more is advised 
echo.
echo %ITEM% 5^)  Exit
echo. %RST%

choice /C 12345 /N /M "Choose an option [1-5]: "%RST%
set "CHOICE=%ERRORLEVEL%"
echo.

if "%CHOICE%"=="1" goto check
if "%CHOICE%"=="2" goto core
if "%CHOICE%"=="3" goto cpu
if "%CHOICE%"=="4" goto cuda
goto end

:ensure_python
rem Ensure a REAL Python is available (not MS Store alias). Auto-install if missing.
set "PYTHON="

rem Prefer Python Launcher with explicit versions (verify it actually runs)
for %%V in (3.12 3.11 3.10) do if not defined PYTHON (
  py -%%V -c "import sys;print(1)" >nul 2>nul && set "PYTHON=py -%%V"
)

rem Fallback to python.exe on PATH, but reject the Microsoft Store alias under WindowsApps
if not defined PYTHON (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    echo %%P | find /I "\WindowsApps\python.exe" >nul
    if errorlevel 1 (
      rem Not the Windows Store alias; verify it runs
      python -c "import sys;print(1)" >nul 2>nul && set "PYTHON=python"
    )
  )
)

if defined PYTHON exit /b 0

echo Python not found or alias detected. Attempting automatic install of Python 3.11...
call :_install_python_311

rem Re-detect after install
set "PYTHON="
for %%V in (3.12 3.11 3.10) do if not defined PYTHON (
  py -%%V -c "import sys;print(1)" >nul 2>nul && set "PYTHON=py -%%V"
)
if not defined PYTHON (
  for /f "delims=" %%P in ('where python 2^>nul') do (
    echo %%P | find /I "\WindowsApps\python.exe" >nul
    if errorlevel 1 (
      python -c "import sys;print(1)" >nul 2>nul && set "PYTHON=python"
    )
  )
)

if defined PYTHON (
  echo Python is now available.
  exit /b 0
) else (
  echo Failed to install Python automatically. Please install Python 3.11+ and re-run.
  exit /b 1
)
:_install_python_311
rem Decide arch: AMD64, ARM64, or x86
set "ARCH=%PROCESSOR_ARCHITECTURE%"
if /I "%ARCH%"=="AMD64" (set "PY_FILE=python-3.11.9-amd64.exe") ^
else if /I "%ARCH%"=="ARM64" (set "PY_FILE=python-3.11.9-arm64.exe") ^
else (set "PY_FILE=python-3.11.9.exe")
set "PY_URL=https://www.python.org/ftp/python/3.11.9/%PY_FILE%"
set "PY_TMP=%TEMP%\%PY_FILE%"

rem Try winget first
winget --version >nul 2>nul
if not errorlevel 1 (
  echo Installing via winget...
  winget install -e --id Python.Python.3.11 --silent --accept-source-agreements --accept-package-agreements
  if not errorlevel 1 (
    echo winget install initiated.
    rem Give a moment then return
    timeout /t 3 >nul
    exit /b 0
  )
)

rem Fallback: download from python.org and run silent installer (requires elevation for AllUsers)
echo Downloading %PY_FILE% from python.org ...
powershell -NoProfile -Command "try{Invoke-WebRequest -UseBasicParsing '%PY_URL%' -OutFile '%PY_TMP%';exit 0}catch{Write-Host 'Download failed';exit 1}"
if errorlevel 1 (
  echo Download failed.
  exit /b 1
)

echo Running Python installer (silent)...
set "PY_ARGS=/quiet InstallAllUsers=1 PrependPath=1 Include_test=0 Include_launcher=1 SimpleInstall=1"
powershell -NoProfile -Command "Start-Process -FilePath '%PY_TMP%' -ArgumentList '%PY_ARGS%' -Verb runAs -Wait"
del /q "%PY_TMP%" >nul 2>nul
exit /b 0

:ensure_venv
if exist ".venv\Scripts\python.exe" goto :eof
echo Creating virtual environment...
call :ensure_python || exit /b 1
%PYTHON% -m venv .venv
if errorlevel 1 goto venv_fail
echo Venv created.
goto :eof
:venv_fail
echo Failed to create virtual environment.
pause
exit /b 1

:pip_upgrade
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --upgrade pip
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --prefer-binary pygame==2.6.* || goto pip_fail
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade mutagen
REM --- Qt/graph/MPV bindings ---
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --prefer-binary "PyQt5==5.15.*" || goto pip_fail
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --prefer-binary "pyqtgraph>=0.13,<1.0" || goto pip_fail
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --prefer-binary "python-mpv>=1.0.6" || goto pip_fail
rem ignore failures
exit /b 0

:install_psutil
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade "psutil>=5.9"
rem ignore failures
exit /b 0
:install_mutagen
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade mutagen
rem ignore failures
echo installing some extra packages

exit /b 0
:check
REM === [AUTO PATCH] Ensure Python is installed before venv creation (Option 1) ===
call :ensure_python
REM Prefer the interpreter chosen by :ensure_python
if defined PYTHON (
  set "PY_CMD=%PYTHON%"
) else (
  REM Fallback detection (should not happen if ensure_python succeeded)
  for %%V in (3.12 3.11 3.10) do if not defined PY_CMD (
    py -%%V -V >nul 2>nul && set "PY_CMD=py -%%V"
  )
  if not defined PY_CMD (
    for /f "delims=" %%P in ('where python 2^>nul') do (
      echo %%P | find /I "\WindowsApps\python.exe" >nul
      if errorlevel 1 set "PY_CMD=python"
    )
  )
)
REM === [AUTO PATCH END] ===

setlocal EnableExtensions EnableDelayedExpansion

echo(
echo %TITLE% ===============================%RST%
echo(  Requirements and Disk Report
echo %TITLE% ===============================%RST%
echo(

rem --- Python ---
set "PY_CMD="
for %%V in (3.11 3.10) do if not defined PY_CMD (
  py -%%V -V >nul 2>nul && set "PY_CMD=py -%%V"
)
if not defined PY_CMD (
  where python >nul 2>nul && set "PY_CMD=python"
)
if defined PY_CMD (
  for /f "tokens=2 delims= " %%P in ('%PY_CMD% -V 2^>nul') do set "PY_VER=%%P"
  echo( Python: OK ^(!PY_VER!^)
) else (
  echo( Python: NOT FOUND  ^(please install Python 3.10 or 3.11^)
)

echo(

rem --- Virtual environment (.venv) ---
if exist ".venv\Scripts\python.exe" (
  echo( .venv: OK  ^(present^)
  call ".venv\Scripts\activate.bat" >nul 2>nul

REM --- Auto-heal pip junk: remove dash-prefixed broken dists that cause 'Ignoring invalid distribution' warnings
set "_FV_VENV=%CD%\.venv"
set "_FV_SITEPKG=%_FV_VENV%\Lib\site-packages"
if exist "%_FV_SITEPKG%" (
  for /d %%D in ("%_FV_SITEPKG%\-*") do rmdir /s /q "%%~fD" 2>nul
  del /q "%_FV_SITEPKG%\-*.pth" 2>nul
  for /d %%I in ("%_FV_SITEPKG%\-*.dist-info") do rmdir /s /q "%%~fI" 2>nul
)

) else (
  echo( .venv: NOT PRESENT
  choice /C YN /N /M "Create .venv now? [Y/N]: "
  if errorlevel 2 (
    echo( Skipping venv creation.
  ) else (
    if defined PY_CMD (
      %PY_CMD% -m venv .venv
      if errorlevel 1 (
        echo( .venv: CREATE FAILED
      ) else (
        call ".venv\Scripts\activate.bat" >nul 2>nul
        echo( .venv: OK  ^(created^)
      )
    ) else (
      echo( .venv: CANNOT CREATE ^(Python not available^)
    )
  )
)

echo %GREENB%  Loading system info, This can take a minute %WHITE% 


rem --- Single dxdiag pass for OS/BIOS/CPU/RAM + GPU details ---
set "DXTXT=%TEMP%\dxdiag_framevision.txt"
dxdiag /whql:off /t "%DXTXT%" >nul 2>nul

rem -- OS / BIOS / CPU / RAM --
for /f "usebackq delims=" %%L in (`findstr /C:"Operating System:" /C:"BIOS:" /C:"Processor:" /C:"Memory:" /C:"Available OS Memory:" "%DXTXT%"`) do echo( %%L

echo(
rem -- Tab statuses --
for /f "usebackq delims=" %%L in (`findstr /C:"Display Tab 1:" /C:"Sound Tab 1:" /C:"Input Tab:" "%DXTXT%"`) do echo( %%L

echo(
rem -- GPU block (grouped) --
set "GPU_NAME="
for /f "tokens=1* delims=:" %%A in ('findstr /C:"Card name:" "%DXTXT%"') do if not defined GPU_NAME set "GPU_NAME=%%B"
if defined GPU_NAME (set "GPU_NAME=!GPU_NAME:~1!")
echo( !GPU_NAME!
for /f "usebackq delims=" %%L in (`findstr /R /C:"^ *Manufacturer:" /C:"^ *Chip type:" /C:"^ *DAC type:" /C:"^ *Dedicated Memory:" /C:"^ *Shared Memory:" /C:"^ *Current Mode:" /C:"^ *HDR Support:" "%DXTXT%"`) do echo( %%L

rem VRAM (GiB) from Dedicated Memory
set "VRAM_GB="
for /f "usebackq delims=" %%V in (`powershell -NoProfile -Command "$d=Get-Content -Raw '%DXTXT%' -ErrorAction SilentlyContinue; $m=[regex]::Match($d,'Dedicated Memory:\s*(\d+)\s*MB','IgnoreCase'); if($m.Success){ [math]::Round([double]$m.Groups[1].Value/1024,0).ToString([Globalization.CultureInfo]::InvariantCulture) }"`) do set "VRAM_GB=%%V"
if defined VRAM_GB echo( VRAM: !VRAM_GB! GiB

rem CUDA flag based on vendor in GPU_NAME or Manufacturer
set "CUDA_CAPABLE=NO"
echo(!GPU_NAME!| find /I "NVIDIA" >nul && set "CUDA_CAPABLE=YES"
for /f "tokens=1* delims=:" %%A in ('findstr /R /C:"^ *Manufacturer:" "%DXTXT%"') do (
  echo(%%B| find /I "NVIDIA" >nul && set "CUDA_CAPABLE=YES"
)
echo( CUDA-capable: !CUDA_CAPABLE!

echo(
rem --- Disk space ---
for /f "usebackq tokens=*" %%F in (`powershell -NoProfile -Command "[math]::Round(([IO.DriveInfo]'%CD:~0,3%').AvailableFreeSpace/1GB,0).ToString([System.Globalization.CultureInfo]::InvariantCulture)"`) do set "FREE_GB=%%F"
if not defined FREE_GB set "FREE_GB=0"
echo( Free space on drive %CD:~0,3% : !FREE_GB! GiB

echo(
rem --- Estimated required space (edit if needed) ---
set "BASE_REQ_GB=5"
set "MODELS_REQ_GB=20"
for /f "usebackq delims=" %%R in (`powershell -NoProfile -Command "[math]::Round([double](%BASE_REQ_GB% + %MODELS_REQ_GB%),1).ToString([System.Globalization.CultureInfo]::InvariantCulture)"`) do set "REQ_GB=%%R"
if not defined REQ_GB set "REQ_GB=%BASE_REQ_GB%"
echo( Estimated required: ~!REQ_GB! GiB  ^(Base app+venv ~!BASE_REQ_GB! GiB  + Models/Zips ~!MODELS_REQ_GB! GiB^)

echo(
rem --- Recommendation ---
set "RECO=CPU install (option 3)"
if /I "!CUDA_CAPABLE!"=="YES" set "RECO=GPU install (option 4)"
echo( Suggested: !RECO!

echo(
pause
endlocal & set "_SKIPCLS=1" & goto menu

:_CheckNvidia
rem helper: sets CUDA_CAPABLE=YES if %CAP% contains NVIDIA (case-insensitive), avoids 'ECHO is off.'
setlocal EnableDelayedExpansion
set "S=!CAP:NVIDIA=!"
if not "!S!"=="!CAP!" (
  endlocal & set "CUDA_CAPABLE=YES" & exit /b 0
) else (
  endlocal & exit /b 0
)
:py_found

if defined PY_CMD (

  for /f "tokens=2 delims= " %%P in ('%PY_CMD% -V 2^>nul') do set "PY_VER=%%P"

  echo( Python: !PY_CMD!  (version !PY_VER!)

) else (

 echo( Python: NOT FOUND on PATH. Please install Python 3.10 or 3.11.

)

rem --- Virtual environment (.venv) status & optional create/activate ---

if exist ".venv\Scripts\python.exe" (

  echo( .venv: PRESENT

  call ".venv\Scripts\activate.bat" >nul 2>nul

  set "VENV_ACTIVE=1"

) else (

  echo( .venv: MISSING

  if defined PY_CMD (

    choice /C YN /N /M "Create .venv now? [Y/N]: "

    if errorlevel 2 (

      echo( You chose NO. Returning to menu...

      endlocal & pause & goto menu

    )

    echo( Creating virtual environment...

    %PY_CMD% -m venv .venv

    if errorlevel 1 (

      echo( Failed to create virtual environment. You can retry later.

    ) else (

      echo( Venv created.

      call ".venv\Scripts\activate.bat" >nul 2>nul

      set "VENV_ACTIVE=1"

    )

  )

)

rem --- Pip install target (site-packages) ---

if exist ".venv\Scripts\python.exe" (

  set "SITE_PKGS=%CD%\.venv\Lib\site-packages"

  echo( pip target: !SITE_PKGS!

) else (

  echo( pip target: (venv not created yet)

)

echo.

rem --- GPU detection ---

set "CUDA_CAPABLE=NO"

set "GPU_VENDOR=Other"

set "GPU_NAME="

set "GPU_DRV="

set "GPU_VRAM_GB="

set "GPU_VRAM_RAW="


  set "GPU_NAME=%%~A"

  set "GPU_VRAM_RAW=%%~B"

)

if defined GPU_NAME (



  rem Convert e.g. "8192 MiB" -> GiB via PowerShell

  for /f "delims=" %%G in ('powershell -NoProfile -Command "$m=\"%GPU_VRAM_RAW%\"; if($m -match '\d+'){[math]::Round([double]($m -replace '\D','')/1024,1)}"') do set "GPU_VRAM_GB=%%G"

  set "CUDA_CAPABLE=YES"

  set "GPU_VENDOR=NVIDIA"

  echo( GPU: !GPU_VENDOR!  Name: !GPU_NAME!  Driver: !GPU_DRV!  VRAM: !GPU_VRAM_GB! GiB

  echo( %GREEN%  CUDA-capable: !CUDA_CAPABLE!%GRAY% 

) else (

  for /f "tokens=1,2 delims=|" %%I in ('powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController | Select-Object -First 1 Name,AdapterRAM) | % { '{0}|{1}' -f $_.Name, $_.AdapterRAM }"') do (

    set "GPU_NAME=%%~I"

    set "GPU_RAM_BYTES=%%~J"

  )

  if defined GPU_NAME (

    echo( GPU: !GPU_NAME!

    echo( CUDA-capable: NO

    for /f "delims=" %%V in ('powershell -NoProfile -Command "$b=%GPU_RAM_BYTES%; if($b -and $b -gt 0){[math]::Round([double]$b/1GB,1)}"') do set "GPU_VRAM_GB=%%V"

    if defined GPU_VRAM_GB echo( VRAM: !GPU_VRAM_GB! GiB

    for /f "delims=" %%V in ('powershell -NoProfile -Command "$n='%GPU_NAME%'; if($n -match '(?i)nvidia'){ 'Vendor: NVIDIA' } elseif($n -match '(?i)amd|radeon'){ 'Vendor: AMD' } elseif($n -match '(?i)intel'){ 'Vendor: Intel' } else { 'Vendor: Other' }"') do set "GPU_VENDOR_LINE=%%V"

    if defined GPU_VENDOR_LINE echo( !GPU_VENDOR_LINE!

  ) else (

    echo( GPU: Unable to detect.

    echo( CUDA-capable: NO

  )

)

echo.

rem --- Disk space on install drive ---

for /f "delims=" %%F in ('powershell -NoProfile -Command "$drive=(Get-Location).Path.Substring(0,1); [int]((Get-PSDrive -Name $drive).Free/1GB)"') do set "FREE_GB=%%F"

if not defined FREE_GB set "FREE_GB=0"

echo( Free space on drive %CD:~0,3% : !FREE_GB! GiB

rem --- Estimate required space ---

set "BASE_REQ_GB=5"

set "MODELS_BYTES=0"

set "ZIP_BYTES=0"

if exist "models" (

  for /f "delims=" %%M in ('powershell -NoProfile -Command "if(Test-Path 'models'){ (Get-ChildItem -LiteralPath 'models' -Recurse -Force -File | Measure-Object -Sum Length).Sum }"') do set "MODELS_BYTES=%%M"

)

for /f "delims=" %%Z in ('powershell -NoProfile -Command "(Get-ChildItem -Path . -Filter *.zip -File -Force | Measure-Object -Sum Length).Sum"') do set "ZIP_BYTES=%%Z"

if not defined MODELS_BYTES set "MODELS_BYTES=0"

if not defined ZIP_BYTES set "ZIP_BYTES=0"

for /f "delims=" %%G in ('powershell -NoProfile -Command "$m=[double](%MODELS_BYTES% + %ZIP_BYTES%); [math]::Round($m/1GB,1)"') do set "PRESENT_MODELS_GB=%%G"

if not defined PRESENT_MODELS_GB set "PRESENT_MODELS_GB=0"

if "%PRESENT_MODELS_GB%"=="0" (

  set "EST_MODELS_GB=10"

) else (

  set "EST_MODELS_GB=%PRESENT_MODELS_GB%"

)

for /f "delims=" %%R in ('powershell -NoProfile -Command "[math]::Round([double](%BASE_REQ_GB% + %EST_MODELS_GB%),1)"') do set "REQ_GB=%%R"

if not defined REQ_GB set "REQ_GB=%BASE_REQ_GB%"

echo( Estimated required: ~!REQ_GB! GiB  (Base app+venv ~!BASE_REQ_GB! GiB  + Models/Zips ~!EST_MODELS_GB! GiB)

rem Warning if not enough

set "NEED_WARN="

for /f "delims=" %%C in ('powershell -NoProfile -Command "if([double]%FREE_GB% -lt [double]%REQ_GB%){'WARN'}"') do set "NEED_WARN=%%C"

if defined NEED_WARN (

  powershell -NoProfile -Command "Write-Host 'WARNING: Free space (!FREE_GB! GiB) is less than estimated required (!REQ_GB! GiB).' -ForegroundColor Yellow" 2>nul

  echo( WARNING: Free space (!FREE_GB! GiB) is less than estimated required (!REQ_GB! GiB).

)

echo.

rem --- Recommendation ---

set "RECO=CPU install (option 3)"

for /f "delims=" %%K in ('powershell -NoProfile -Command "$v='%GPU_VRAM_GB%'; if($v -and [double]$v -ge 4){'OK'}"') do set "VRAM_OK=%%K"

if /I "%CUDA_CAPABLE%"=="YES" if "%VRAM_OK%"=="OK" set "RECO=GPU install (option 4)"

echo( Suggested: !RECO!

echo.

pause

endlocal & goto menu

:py_found

if defined PY_CMD (

  for /f "delims=" %%P in ('%PY_CMD% -c "import sys; print(\"%s.%s.%s\"% (sys.version_info[0],sys.version_info[1],sys.version_info[2]))"') do set "PY_VER=%%P"

  echo Python: !PY_CMD!  (version !PY_VER!)

) else (

  echo Python: NOT FOUND on PATH. Please install Python 3.10 or 3.11.

)

rem --- Virtual environment (.venv) status & optional create/activate ---

if exist ".venv\Scripts\python.exe" (

  echo .venv: PRESENT

  call ".venv\Scripts\activate.bat" >nul 2>nul

  set "VENV_ACTIVE=1"

) else (

  echo .venv: MISSING

  if defined PY_CMD (

    choice /C YN /N /M "Create .venv now? [Y/N]: "

    if errorlevel 2 (

      echo You chose NO. Returning to menu...

      endlocal & pause & goto menu

    )

    echo Creating virtual environment...

    %PY_CMD% -m venv .venv

    if errorlevel 1 (

      echo Failed to create virtual environment. You can retry later.

    ) else (

      echo Venv created.

      call ".venv\Scripts\activate.bat" >nul 2>nul

      set "VENV_ACTIVE=1"

    )

  )

)

rem --- Pip install target (site-packages) ---

if exist ".venv\Scripts\python.exe" (

  for /f "delims=" %%S in ('".\.venv\Scripts\python.exe" -c "import sysconfig;print(sysconfig.get_paths().get(\"purelib\",\"\"))"') do set "SITE_PKGS=%%S"

  echo pip target: !SITE_PKGS!

) else (

  echo pip target: (venv not created yet)

)

echo.

rem --- GPU detection ---

set "CUDA_CAPABLE=NO"

set "GPU_VENDOR=Other"

set "GPU_NAME="

set "GPU_DRV="

set "GPU_VRAM_GB="

set "GPU_VRAM_RAW="


  set "GPU_NAME=%%~A"

  set "GPU_VRAM_RAW=%%~B"

)

if defined GPU_NAME (



  rem Convert e.g. "8192 MiB" -> GiB via PowerShell

  for /f "delims=" %%G in ('powershell -NoProfile -Command "$m=\"%GPU_VRAM_RAW%\"; if($m -match '\d+'){[math]::Round([double]($m -replace '\D','')/1024,1)}"') do set "GPU_VRAM_GB=%%G"

  set "CUDA_CAPABLE=YES"

  set "GPU_VENDOR=NVIDIA"

  echo GPU: !GPU_VENDOR!  Name: !GPU_NAME!  Driver: !GPU_DRV!  VRAM: !GPU_VRAM_GB! GiB

  echo CUDA-capable: !CUDA_CAPABLE!

) else (

  for /f "tokens=1,2 delims=|" %%I in ('powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController | Select-Object -First 1 Name,AdapterRAM) | % { '{0}|{1}' -f $_.Name, $_.AdapterRAM }"') do (

    set "GPU_NAME=%%~I"

    set "GPU_RAM_BYTES=%%~J"

  )

  if defined GPU_NAME (

    echo GPU: !GPU_NAME!

    echo CUDA-capable: NO

    for /f "delims=" %%V in ('powershell -NoProfile -Command "$b=%GPU_RAM_BYTES%; if($b -and $b -gt 0){[math]::Round([double]$b/1GB,1)}"') do set "GPU_VRAM_GB=%%V"

    if defined GPU_VRAM_GB echo VRAM: !GPU_VRAM_GB! GiB

    for /f "delims=" %%V in ('powershell -NoProfile -Command "$n='%GPU_NAME%'; if($n -match '(?i)nvidia'){ 'Vendor: NVIDIA' } elseif($n -match '(?i)amd|radeon'){ 'Vendor: AMD' } elseif($n -match '(?i)intel'){ 'Vendor: Intel' } else { 'Vendor: Other' }"') do set "GPU_VENDOR_LINE=%%V"

    if defined GPU_VENDOR_LINE echo !GPU_VENDOR_LINE!

  ) else (

    echo GPU: Unable to detect.

    echo CUDA-capable: NO

  )

)

echo.

rem --- Disk space on install drive ---

for /f "delims=" %%F in ('powershell -NoProfile -Command "$drive=(Get-Location).Path.Substring(0,1); [int]((Get-PSDrive -Name $drive).Free/1GB)"') do set "FREE_GB=%%F"

if not defined FREE_GB set "FREE_GB=0"

echo Free space on drive %CD:~0,3% : !FREE_GB! GiB

rem --- Estimate required space ---

set "BASE_REQ_GB=5"

set "MODELS_BYTES=0"

set "ZIP_BYTES=0"

if exist "models" (

  for /f "delims=" %%M in ('powershell -NoProfile -Command "if(Test-Path 'models'){ (Get-ChildItem -LiteralPath 'models' -Recurse -Force -File | Measure-Object -Sum Length).Sum }"') do set "MODELS_BYTES=%%M"

)

for /f "delims=" %%Z in ('powershell -NoProfile -Command "(Get-ChildItem -Path . -Filter *.zip -File -Force | Measure-Object -Sum Length).Sum"') do set "ZIP_BYTES=%%Z"

if not defined MODELS_BYTES set "MODELS_BYTES=0"

if not defined ZIP_BYTES set "ZIP_BYTES=0"

for /f "delims=" %%G in ('powershell -NoProfile -Command "$m=[double](%MODELS_BYTES% + %ZIP_BYTES%); [math]::Round($m/1GB,1)"') do set "PRESENT_MODELS_GB=%%G"

if not defined PRESENT_MODELS_GB set "PRESENT_MODELS_GB=0"

if "%PRESENT_MODELS_GB%"=="0" (

  set "EST_MODELS_GB=10"

) else (

  set "EST_MODELS_GB=%PRESENT_MODELS_GB%"

)

for /f "delims=" %%R in ('powershell -NoProfile -Command "[math]::Round([double](%BASE_REQ_GB% + %EST_MODELS_GB%),1)"') do set "REQ_GB=%%R"

if not defined REQ_GB set "REQ_GB=%BASE_REQ_GB%"

echo Estimated required: ~!REQ_GB! GiB  ^(Base app+venv ~!BASE_REQ_GB! GiB  + Models/Zips ~!EST_MODELS_GB! GiB^)

rem Warning if not enough

set "NEED_WARN="

for /f "delims=" %%C in ('powershell -NoProfile -Command "if([double]%FREE_GB% -lt [double]%REQ_GB%){'WARN'}"') do set "NEED_WARN=%%C"

if defined NEED_WARN (

  powershell -NoProfile -Command "Write-Host 'WARNING: Free space (!FREE_GB! GiB) is less than estimated required (!REQ_GB! GiB).' -ForegroundColor Yellow" 2>nul

  echo WARNING: Free space (!FREE_GB! GiB) is less than estimated required (!REQ_GB! GiB).

)

echo.

rem --- Recommendation ---

set "RECO=CPU install (option 3)"

for /f "delims=" %%K in ('powershell -NoProfile -Command "$v='%GPU_VRAM_GB%'; if($v -and [double]$v -ge 4){'OK'}"') do set "VRAM_OK=%%K"

if /I "%CUDA_CAPABLE%"=="YES" if "%VRAM_OK%"=="OK" set "RECO=GPU install (option 4)"

echo Suggested: !RECO!

echo.

pause

endlocal & goto menu

:py_found

if defined PY_CMD (

  for /f "usebackq delims=" %%P in (`%PY_CMD% -c "import sys; print('.'.join(map(str,sys.version_info[:3])))"`) do set "PY_VER=%%P"

  echo Python: !PY_CMD!  ^(version !PY_VER!^)

) else (

  echo Python: NOT FOUND on PATH. Please install Python 3.10 or 3.11.

)

rem --- Virtual environment (.venv) status & optional create/activate ---

if exist ".venv\Scripts\python.exe" (

  echo .venv: PRESENT

  call ".venv\Scripts\activate.bat" >nul 2>nul

  set "VENV_ACTIVE=1"

) else (

  echo .venv: MISSING

  if defined PY_CMD (

    choice /C YN /N /M "Create .venv now? [Y/N]: "

    if errorlevel 2 (

      echo You chose NO. Returning to menu...

      endlocal & pause & goto menu

    )

    echo Creating virtual environment...

    %PY_CMD% -m venv .venv

    if errorlevel 1 (

      echo Failed to create virtual environment. You can retry later.

    ) else (

      echo Venv created.

      call ".venv\Scripts\activate.bat" >nul 2>nul

      set "VENV_ACTIVE=1"

    )

  )

)

rem --- Pip install target (site-packages) ---

if exist ".venv\Scripts\python.exe" (

  for /f "usebackq delims=" %%S in (`".\.venv\Scripts\python.exe" -c "import sysconfig;print(sysconfig.get_paths()['purelib'])"`) do set "SITE_PKGS=%%S"

  echo pip target: !SITE_PKGS!

) else (

  echo pip target: (venv not created yet)

)

echo.

rem --- GPU detection ---

set "CUDA_CAPABLE=NO"

set "GPU_VENDOR=Other"

set "GPU_NAME="

set "GPU_DRV="

set "GPU_VRAM_GB="


  set "GPU_NAME=%%~A"

  set "GPU_DRV=%%~B"

  set "GPU_VRAM_RAW=%%~C"

)

if defined GPU_NAME (

  rem Convert e.g. "8192 MiB" -> GiB via PowerShell

  for /f "usebackq delims=" %%G in (`powershell -NoProfile -Command "$m='%GPU_VRAM_RAW%'; if($m -match '\\d+'){[math]::Round([double]($m -replace '\\D','')/1024,1)}"`) do set "GPU_VRAM_GB=%%G"

  set "CUDA_CAPABLE=YES"

  set "GPU_VENDOR=NVIDIA"

  echo GPU: !GPU_VENDOR!  Name: !GPU_NAME!  Driver: !GPU_DRV!  VRAM: !GPU_VRAM_GB! GiB

  echo CUDA-capable: !CUDA_CAPABLE!

) else (

  for /f "usebackq tokens=1,2 delims=|" %%I in (`powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController | Select-Object -First 1 Name,AdapterRAM) | %%{ '{0}|{1}' -f $_.Name, $_.AdapterRAM }"`) do (

    set "GPU_NAME=%%~I"

    set "GPU_RAM_BYTES=%%~J"

  )

  if defined GPU_NAME (

    echo GPU: !GPU_NAME!

    echo CUDA-capable: NO

    for /f "usebackq delims=" %%V in (`powershell -NoProfile -Command "$b=%GPU_RAM_BYTES%; if($b -and $b -gt 0){[math]::Round([double]$b/1GB,1)}"` ) do set "GPU_VRAM_GB=%%V"

    if defined GPU_VRAM_GB echo VRAM: !GPU_VRAM_GB! GiB

    echo.

    for /f "usebackq delims=" %%V in (`powershell -NoProfile -Command "$n='%GPU_NAME%'; if($n -match '(?i)nvidia'){ 'Vendor: NVIDIA' } elseif($n -match '(?i)amd|radeon'){ 'Vendor: AMD' } elseif($n -match '(?i)intel'){ 'Vendor: Intel' } else { 'Vendor: Other' }"`) do set "GPU_VENDOR_LINE=%%V"

    if defined GPU_VENDOR_LINE echo !GPU_VENDOR_LINE!

  ) else (

    echo GPU: Unable to detect.

    echo CUDA-capable: NO

  )

)

echo.

rem --- Disk space on install drive ---

for /f "usebackq tokens=*" %%F in (`powershell -NoProfile -Command "[int]([System.IO.DriveInfo]::GetDrives() ^| ?{\$_.Name -eq (Get-Location).Path.Substring(0,3)}).AvailableFreeSpace/1GB"` ) do set "FREE_GB=%%F"

echo Free space on drive %CD:~0,3% : !FREE_GB! GiB

rem --- Estimate required space ---

set "BASE_REQ_GB=5"

set "MODELS_BYTES=0"

if exist "models" (

  for /f "usebackq delims=" %%M in (`powershell -NoProfile -Command "if(Test-Path 'models'){ (Get-ChildItem -LiteralPath 'models' -Recurse -Force -File ^| Measure-Object -Sum Length).Sum }"` ) do set "MODELS_BYTES=%%M"

)

rem Also include top-level ZIP packs (if any)

for /f "usebackq delims=" %%Z in (`powershell -NoProfile -Command "(Get-ChildItem -Path . -Filter *.zip -File -Force | Measure-Object -Sum Length).Sum"` ) do set "ZIP_BYTES=%%Z"

if not defined ZIP_BYTES set "ZIP_BYTES=0"

for /f "usebackq delims=" %%G in (`powershell -NoProfile -Command "$m=[double](%MODELS_BYTES% + %ZIP_BYTES%); [math]::Round($m/1GB,1)"`) do set "PRESENT_MODELS_GB=%%G"

set "EST_MODELS_GB="

if "%PRESENT_MODELS_GB%"=="0" (

  set "EST_MODELS_GB=10"

) else (

  set "EST_MODELS_GB=%PRESENT_MODELS_GB%"

)

for /f "usebackq delims=" %%R in (`powershell -NoProfile -Command "[math]::Round([double](%BASE_REQ_GB% + %EST_MODELS_GB%),1)"`) do set "REQ_GB=%%R"

echo Estimated required: ~!REQ_GB! GiB  ^(Base app+venv ~!BASE_REQ_GB! GiB  + Models/Zips ~!EST_MODELS_GB! GiB^)

rem Warning if not enough

set "NEED_WARN="

for /f "usebackq delims=" %%C in (`powershell -NoProfile -Command "if([double]%FREE_GB% -lt [double]%REQ_GB%){'WARN'}"` ) do set "NEED_WARN=%%C"

if defined NEED_WARN (

  powershell -NoProfile -Command "Write-Host 'WARNING: Free space (!FREE_GB! GiB) is less than estimated required (!REQ_GB! GiB).' -ForegroundColor Yellow" 2>nul

  echo WARNING: Free space (!FREE_GB! GiB) is less than estimated required (!REQ_GB! GiB).

)

echo.

rem --- Recommendation ---

set "RECO=CPU install (option 3)"

for /f "usebackq delims=" %%K in (`powershell -NoProfile -Command "$v='%GPU_VRAM_GB%'; if($v -and [double]$v -ge 4){'OK'}"` ) do set "VRAM_OK=%%K"

if /I "%CUDA_CAPABLE%"=="YES" if "%VRAM_OK%"=="OK" set "RECO=GPU install (option 4)"

echo Suggested: !RECO!

echo.

pause

endlocal & goto menu

:core
call :ensure_python || exit /b 1
call :ensure_venv   || exit /b 1
call :pip_upgrade
echo Installing core requirements...
if exist requirements-core.txt (
if exist ".venv\Scripts\python.exe"   call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade -r requirements-core.txt
  if errorlevel 1 goto pip_fail
)
call :install_psutil
if not exist "bin" mkdir "bin"
echo > ".installed_core"
echo Core install complete.
if exist "framevision_run.py" (
  echo Launching FrameVision...
  start "" "start.bat"
  goto end
) else (
  echo framevision_run.py was not found here; returning to menu.
  pause
  goto menu
)

:cpu
call :ensure_python || exit /b 1
call :ensure_venv   || exit /b 1
call :pip_upgrade
echo Installing CPU stack...
if exist requirements-core.txt (
if exist ".venv\Scripts\python.exe"   call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade -r requirements-core.txt
  if errorlevel 1 goto pip_fail
)
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --force-reinstall torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade torchvision==0.18.1+cpu torchaudio==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto pip_fail
if exist requirements-cpu.txt (
if exist ".venv\Scripts\python.exe"   call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade -r requirements-cpu.txt
  if errorlevel 1 goto pip_fail
)
rem === Download tools & models for CPU (non-CUDA) ===
if errorlevel 1 echo( (warning) tool download had issues; continuing...
call ".venv\Scripts\python.exe" -u scripts\download_externals.py --all

IF %ERRORLEVEL% NEQ 0 (
  echo download_externals failed with exit code %ERRORLEVEL%.
  echo Not launching the app. Press any key to continue . . .
  pause >nul
  goto :eof
)

echo === VERSION SUMMARY ===
call ".venv\Scripts\python.exe" -c "import sys;import pkgutil;import importlib;mods=[\'diffusers\',\'transformers\',\'huggingface_hub\',\'torch\'];print(\'Python:\',sys.version.split()[0]);print(\'CUDA torch:\',__import__(\'torch\').version.cuda);print(\'\'.join([m+\' \'+__import__(m).__version__+\'\n\' for m in mods if pkgutil.find_loader(m)]))"
if errorlevel 1 echo( (warning) model download had issues; continuing...

call :install_psutil
call :install_mutagen

echo > ".installed_cpu"
if exist "framevision_run.py" (
  echo CPU install complete. Launching FrameVision...
  start "" "start.bat"
  goto end
) else (
  echo CPU install complete, but framevision_run.py is missing; returning to menu.
  pause
  goto menu
)

:cuda
call :ensure_python
call :ensure_venv
call :pip_upgrade
if exist scripts\external_downloader.py (
  call ".venv\Scripts\python.exe" scripts\external_downloader.py >nul 2>nul
)
echo Installing CUDA stack...
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --force-reinstall torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto pip_fail
if exist requirements-core.txt (
if exist ".venv\Scripts\python.exe"   call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade -r requirements-core.txt
  if errorlevel 1 goto pip_fail
)
echo Installing Diffusers for TXT->IMG...
if errorlevel 1 goto pip_fail
if exist requirements-gpu.txt (
if exist ".venv\Scripts\python.exe"   call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade diffusers>=0.30.0
rem call :maybe_install_qwen_t2i

if exist ".venv\Scripts\python.exe"   call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade -r requirements-gpu.txt
  if errorlevel 1 goto pip_fail
)
rem === Download tools & models for GPU (CUDA) ===
if errorlevel 1 echo( (warning) tool download had issues; continuing...
rem Ensure Transformers for Qwen
call ".venv\Scripts\python.exe" -u scripts\download_externals.py --all
if errorlevel 1 echo( (warning) model download had issues; continuing...
rem Validate local Qwen20B model if present
rem  call ".venv\Scripts\python.exe" -u scripts\qwen20b_validate.py ".\models\Qwen20B"
)
if errorlevel 1 echo( (warning) model download had issues; continuing...


call :install_mutagen
echo > ".installed_gpu"
if exist "framevision_run.py" (
  echo CUDA install complete. Launching FrameVision...
  start "" "start.bat"
  goto end
) else (
  echo CUDA install complete, but framevision_run.py is missing; returning to menu.
  pause
  goto menu
)

:pip_fail
echo.
echo One or more install steps failed.
echo Close all Python apps and retry if DLLs were locked.
pause
goto menu


:end
exit /b 0
rem --- Extra: Install TeaCache and SageAttention ---
if exist ".venv\Scripts\python.exe" (
  rem Try possible package name variants for TeaCache
  call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade teacache ^
    || call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade tea-cache ^
    || call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade tea_cache
  rem Try possible package name variants for SageAttention
  call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade sageattention ^
    || call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade sage-attention ^
    || call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade SageAttention
) else (
  echo( [skip] .venv not found; skipping TeaCache / SageAttention install
)

call :install_mutagen
echo > ".installed_gpu"
if exist "framevision_run.py" (
  echo CUDA install complete. Launching FrameVision...
  start "" "start.bat"
  goto end
) else (
  echo CUDA install complete, but framevision_run.py is missing; returning to menu.
  pause
  goto menu
)

:pip_fail
echo.
echo One or more install steps failed.
echo Close all Python apps and retry if DLLs were locked.
pause
goto menu


:end
exit /b 0



call start.bat

:accel_install_sage2
REM === Optional: SageAttention v2/2++ from source (best-effort). Not called by default. ===
setlocal
set "ACCEL_DIR=%~dp0accel_pkgs"
if not exist "%ACCEL_DIR%" mkdir "%ACCEL_DIR%"
where git >nul 2>&1
if errorlevel 1 (
  echo [WARN] Git not found in PATH. Skipping SageAttention v2.
  endlocal & goto :eof
)

if not exist "%ACCEL_DIR%\SageAttention" (
  git clone --depth=1 https://github.com/thu-ml/SageAttention "%ACCEL_DIR%\SageAttention"
) else (
  pushd "%ACCEL_DIR%\SageAttention" && git pull && popd
)
if exist "%ACCEL_DIR%\SageAttention" (
  pushd "%ACCEL_DIR%\SageAttention"
  ".\.venv\Scripts\python" -m pip install -U ninja packaging setuptools wheel
  ".\.venv\Scripts\python" -m pip install -U -r requirements.txt 2>nul
  ".\.venv\Scripts\python" -m pip install -U .
  if errorlevel 1 (
    echo [WARN] SageAttention v2 build failed. Continuing...
  ) else (
    echo [OK] SageAttention v2 installed.
  )
  popd
)

echo.
echo [PAUSE] SageAttention v2 step finished. Press any key to continue...
pause >nu


call start.bat