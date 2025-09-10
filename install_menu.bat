@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"

:menu
if defined _SKIPCLS (set "_SKIPCLS=") else (cls)
echo.
echo ===============================
echo      FrameVision Installer
echo ===============================
echo.
echo 1^) Check requirements and disk space
echo 2^) Core install  (app only, no ML deps/models)
echo 3^) Full install  (CPU / non-CUDA)
echo 4^) Full install  (CUDA GPU)
echo 5^) Exit
echo.

choice /C 12345 /N /M "Choose an option [1-5]: "
set "CHOICE=%ERRORLEVEL%"
echo.

if "%CHOICE%"=="1" goto check
if "%CHOICE%"=="2" goto core
if "%CHOICE%"=="3" goto cpu
if "%CHOICE%"=="4" goto cuda
goto end
:ensure_python
rem Robust Python resolver: prefer py -3.11, then py -3.10, then python on PATH
set "PYTHON="
for %%V in (3.11 3.10) do if not defined PYTHON (
  py -%%V -V >nul 2>nul && set "PYTHON=py -%%V"
)
if not defined PYTHON (
  where python >nul 2>nul && set "PYTHON=python"
)
if not defined PYTHON (
  echo Python was not found. Please install Python 3.10 or 3.11.
  exit /b 1
)
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
rem ignore failures
exit /b 0

:install_psutil
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade "psutil>=5.9"
rem ignore failures
exit /b 0
:check
setlocal EnableExtensions EnableDelayedExpansion

echo(
echo ===============================
echo(  Requirements ^& Disk Report
echo ===============================
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

echo  wait a second


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
set "MODELS_REQ_GB=10"
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

  echo( CUDA-capable: !CUDA_CAPABLE!

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
rem Ensure transformers for Qwen on CPU
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade transformers>=4.51.3,<5 "safetensors>=0.4.3"
call ".venv\Scripts\python.exe" -u scripts\download_externals.py --all

IF %ERRORLEVEL% NEQ 0 (
  echo download_externals failed with exit code %ERRORLEVEL%.
  echo Not launching the app. Press any key to continue . . .
  pause >nul
  goto :eof
)
if errorlevel 1 echo( (warning) model download had issues; continuing...
echo Validating local Qwen20B model folder (if present)...
  call ".venv\Scripts\python.exe" -u scripts\qwen20b_validate.py ".\models\Qwen20B"
  if errorlevel 1 (
    echo [QWEN VALIDATION FAILED] — see messages above.
    pause
  ) else (
    echo [QWEN VALIDATION OK]
  )
)
echo === VERSION SUMMARY ===
call ".venv\Scripts\python.exe" -c "import sys;import pkgutil;import importlib;mods=[\'diffusers\',\'transformers\',\'huggingface_hub\',\'torch\'];print(\'Python:\',sys.version.split()[0]);print(\'CUDA torch:\',__import__(\'torch\').version.cuda);print(\'\'.join([m+\' \'+__import__(m).__version__+\'\n\' for m in mods if pkgutil.find_loader(m)]))"
if errorlevel 1 echo( (warning) model download had issues; continuing...

call :install_psutil

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
  call ".venv\Scripts\python.exe" scripts\external_downloader.py --component qwen_t2i_20b >nul 2>nul
)
if not exist .\models\QwenT2I20B md .\models\QwenT2I20B 2>nul
break > ".qwen_t2i_enabled"
echo Installing CUDA stack...
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --force-reinstall torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto pip_fail
if exist requirements-core.txt (
if exist ".venv\Scripts\python.exe"   call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade -r requirements-core.txt
  if errorlevel 1 goto pip_fail
)
echo Installing Diffusers & friends for TXT->IMG...
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade "diffusers==0.35.1" "transformers>=4.51.3,<5" "huggingface_hub==0.34.4" "safetensors>=0.4.3" "pillow==11.3.0"
if errorlevel 1 goto pip_fail
if exist requirements-gpu.txt (
if exist ".venv\Scripts\python.exe"   call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade diffusers>=0.30.0
call :maybe_install_qwen_t2i

if exist ".venv\Scripts\python.exe"   call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade -r requirements-gpu.txt
  if errorlevel 1 goto pip_fail
)
rem === Download tools & models for GPU (CUDA) ===
if errorlevel 1 echo( (warning) tool download had issues; continuing...
rem Ensure Transformers for Qwen
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade transformers>=4.51.3,<5 "safetensors>=0.4.3"
call ".venv\Scripts\python.exe" -u scripts\download_externals.py --all
if errorlevel 1 echo( (warning) model download had issues; continuing...
rem Validate local Qwen20B model if present
  call ".venv\Scripts\python.exe" -u scripts\qwen20b_validate.py ".\models\Qwen20B"
)
if errorlevel 1 echo( (warning) model download had issues; continuing...

rem ---- Optional Qwen Text-to-Image 20B enablement (post-CUDA stack) ----
if not "%QWEN_T2I_WANT%"=="1" goto :after_qwen_t2i_post
echo [info] Checking VRAM suitability for Qwen T2I 20B...
  echo [info] Sorry, this option is not available for your GPU right now.
  echo [info] We plan a lighter 6 GB variant later in the roadmap.
  pause
  goto :after_qwen_t2i_post
)
  echo [warning] Your GPU has less than 12 GB VRAM. You may experience instability with Qwen T2I 20B.
  choice /c YN /n /m "Continue anyway? [Y/N]: "
  if errorlevel 2 goto :after_qwen_t2i_post
)
echo [info] Enabling Qwen T2I 20B...
call :ensure_python || goto :after_qwen_t2i_post
call :ensure_venv   || goto :after_qwen_t2i_post
call :pip_upgrade
if exist ".venv\Scripts\python.exe" call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade pip
call ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade "huggingface_hub>=0.34.4" "diffusers==0.35.1" "transformers>=4.51.3,<5" "safetensors>=0.4.4" "pillow==10.4.0"
if exist scripts\external_downloader.py (
  echo [info] Fetching Qwen T2I 20B assets via external_downloader.py
  call ".venv\Scripts\python.exe" scripts\external_downloader.py --component qwen_t2i_20b || echo [warning] downloader returned a non-zero code
) else (
  echo [info] external_downloader.py not found. Assuming models are pre-copied.
)
if not exist .\models\QwenT2I20B md .\models\QwenT2I20B 2>nul
break > ".qwen_t2i_enabled"
echo [info] Qwen T2I 20B enabled. You can disable by deleting .qwen_t2i_enabled
pause
:after_qwen_t2i_post
call :install_psutil
REM >>> FRAMEVISION_QWEN_BEGIN
call :QWEN_T2I_MENU
REM <<< FRAMEVISION_QWEN_END

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

:maybe_install_qwen_t2i
echo.
echo [prompt] Also install Qwen Text-to-Image 20B now? (Y/N)
choice /C YN /N /M "Install Qwen T2I 20B now? [Y/N]"
if errorlevel 2 goto :t2i_skip
rem t2i_yes
echo [info] Preparing Qwen T2I 20B...
if not exist "models" md "models"
if not exist "models\QwenT2I20B" md "models\QwenT2I20B"
set "T2I_FOUND="
for %%F in ("models\QwenT2I20B\*.safetensors") do set "T2I_FOUND=1"
if not defined T2I_FOUND if exist "models\QwenT2I20B\model_index.json" set "T2I_FOUND=1"
if defined T2I_FOUND goto :t2i_enable
if exist "scripts\external_downloader.py" if exist ".\.venv\Scripts\python.exe" (
  call ".\.venv\Scripts\python.exe" "scripts\external_downloader.py" --component qwen_t2i_20b >nul 2>nul
)
:t2i_enable
break > ".qwen_t2i_enabled"
echo [ok] Qwen T2I 20B enabled.
goto :after_qwen_t2i_post
:t2i_skip
echo [info] Skipping Qwen T2I 20B.
goto :after_qwen_t2i_post
:t2i_yes
echo [info] Preparing Qwen T2I 20B...
if not exist "models\QwenT2I20B" mkdir "models\QwenT2I20B"

rem Detect pre-copied model files
set "T2I_FOUND="
for %%F in ("models\QwenT2I20B\*.safetensors") do set "T2I_FOUND=1"
if not defined T2I_FOUND if exist "models\QwenT2I20B\model_index.json" set "T2I_FOUND=1"

if defined T2I_FOUND (
  echo [ok] Found pre-copied Qwen T2I model files in models\QwenT2I20B.
) else (
  if exist "scripts\external_downloader.py" if exist ".\.venv\Scripts\python.exe" (
    echo [info] Attempting model fetch via scripts\external_downloader.py (guarded).
    echo [info] If your downloader expects args, it may be a no-op; this call is non-fatal.
    call ".\.venv\Scripts\python.exe" "scripts\external_downloader.py" --model qwen-t2i-20b --dest "models\QwenT2I20B" || echo [warn] Downloader did not complete (this is OK if you pre-copied files).
  ) else (
    echo [warn] Skipping auto-download (downloader or venv not present).
  )
)

rem Validate again after possible download
set "T2I_READY="
for %%F in ("models\QwenT2I20B\*.safetensors") do set "T2I_READY=1"
if not defined T2I_READY if exist "models\QwenT2I20B\model_index.json" set "T2I_READY=1"

if defined T2I_READY (
  echo [ok] Qwen T2I 20B assets detected. Enabling integration...
  break > ".qwen_t2i_enabled"
  echo [ok] Wrote .qwen_t2i_enabled flag.
) else (
  echo [warn] Qwen T2I 20B not detected. You can add files into models\QwenT2I20B and re-run option 4 to enable.
)
goto :eof

:t2i_skip
echo [info] Skipping Qwen T2I 20B.
goto :eof
:end
exit /b 0

REM >>> FRAMEVISION_QWEN_BEGIN

:QWEN_T2I_SKIP
goto QWEN_T2I_DONE

:QWEN_T2I_DONE
REM return to caller menu
exit /b 0
REM <<< FRAMEVISION_QWEN_END
IF %ERRORLEVEL% NEQ 0 (
  echo download_externals failed with %ERRORLEVEL%.
  pause
  goto :eof
)
call start.bat
