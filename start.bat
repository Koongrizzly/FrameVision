@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"

set FV_REALSR_GPU_ID=0
set FV_REALSR_JOBS=1:2:2


rem --- Ensure legacy .\bin points to .\presets\bin (ffmpeg etc.) ---
if exist "presets\bin" (
  if not exist "bin" (
    mklink /J "bin" "presets\bin" >nul 2>nul
  )
)

rem --- Only run app if venv + install flag + entry exist ---
set "HAVE_VENV="
if exist .venv\Scripts\python.exe set "HAVE_VENV=1"

set "HAVE_INSTALL_FLAG="
if exist .installed_cpu  set "HAVE_INSTALL_FLAG=1"
if exist .installed_gpu  set "HAVE_INSTALL_FLAG=1"
if exist .installed_core set "HAVE_INSTALL_FLAG=1"

set "HAVE_ENTRY="
if exist framevision_run.py set "HAVE_ENTRY=1"

if not defined HAVE_VENV goto do_install
if not defined HAVE_INSTALL_FLAG goto do_install
if not defined HAVE_ENTRY goto do_install

rem --- Self-heal: ensure psutil is present in the venv ---
".venv\Scripts\python.exe" -c "import psutil" 1>nul 2>nul
if errorlevel 1 (
  echo Missing 'psutil' in venv. Installing...
  ".venv\Scripts\python.exe" -m pip install --no-cache-dir --upgrade psutil>=5.9
)

rem --- Silent source tidy (lives in scripts\). No files created in root.
if exist "scripts\post_install_fixes.py" (
  ".venv\Scripts\python.exe" "scripts\post_install_fixes.py" 1>nul 2>nul
)

rem --- Start worker (optional)
if not exist logs mkdir logs
set "WORKER="
if exist framevision_worker.py set "WORKER=framevision_worker.py"
if not defined WORKER if exist worker\framevision_worker.py set "WORKER=worker\framevision_worker.py"
if not defined WORKER if exist helpers\worker.py set "WORKER=helpers\worker.py"
if not defined WORKER if exist scripts\framevision_worker.py set "WORKER=scripts\framevision_worker.py"
if defined WORKER start "" /MIN ".venv\Scripts\python.exe" "%WORKER%" 1>>"logs\worker.log" 2>&1

echo Starting FrameVision...
".venv\Scripts\python.exe" framevision_run.py
goto :eof

:do_install
if exist install_menu.bat (
  call install_menu.bat
  goto :eof
)
echo Installer not found (install_menu.bat missing).
echo Place start.bat and install_menu.bat in your FrameVision folder.
pause
goto :eof
