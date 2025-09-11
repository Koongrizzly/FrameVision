@echo off
setlocal EnableExtensions EnableDelayedExpansion
title FrameVision Updater (Strict Mirror)

REM ====== CONFIG ======
set "REPO_URL=https://github.com/Koongrizzly/FrameVision.git"
set "BRANCH=main"
set "LOG=update.log"
REM Exclusions (space separated) - not touched in ZIP mirror mode
set "EXCLUDES=.git .venv venv ENV outputs models .hf_cache __pycache__"
REM =====================

echo.
echo ==============================
echo   FrameVision - STRICT UPDATE
echo ==============================
echo   This will restore deleted files.
echo   (ZIP mode mirrors and may delete local-only files *outside* excludes)
echo ==============================
echo.

cd /d "%~dp0"
echo [%DATE% %TIME%] Start strict update >> "%LOG%"

where git >nul 2>nul
if %ERRORLEVEL%==0 (
    if exist ".git" (
        echo Using Git strict reset...
        git fetch origin %BRANCH% >> "%LOG%" 2>&1
        if errorlevel 1 goto :zip
        git reset --hard origin/%BRANCH% >> "%LOG%" 2>&1
        if errorlevel 1 goto :zip
        echo Restored files to match origin/%BRANCH%.
        goto :done
    ) else (
        echo Bootstrapping a git checkout...
        git init >> "%LOG%" 2>&1
        git remote add origin "%REPO_URL%" >> "%LOG%" 2>&1
        git fetch origin %BRANCH% >> "%LOG%" 2>&1
        if errorlevel 1 goto :zip
        git checkout -b %BRANCH% --track origin/%BRANCH% >> "%LOG%" 2>&1
        if errorlevel 1 goto :zip
        echo Initial checkout complete.
        goto :done
    )
)

:zip
echo.
echo Git unavailable or failed. Using ZIP mirror...
echo [%DATE% %TIME%] ZIP MIRROR mode >> "%LOG%"

set "PS_EX=%EXCLUDES%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Set-StrictMode -Version Latest; $ErrorActionPreference='Stop'; ^
   $tmp = Join-Path $env:TEMP ('fv_update_' + (Get-Date -Format yyyyMMdd_HHmmss)); ^
   New-Item -ItemType Directory -Path $tmp | Out-Null; ^
   $zip = Join-Path $tmp 'repo.zip'; ^
   $repoZip = 'https://github.com/Koongrizzly/FrameVision/archive/refs/heads/%BRANCH%.zip'; ^
   Invoke-WebRequest -Uri $repoZip -OutFile $zip; ^
   $dest = Join-Path $tmp 'unzipped'; ^
   Expand-Archive -Path $zip -DestinationPath $dest -Force; ^
   $root = Get-ChildItem $dest | Where-Object { $_.PSIsContainer } | Select-Object -First 1; ^
   $src = $root.FullName; ^
   $target = (Get-Location).Path; ^
   $exDirs = '%PS_EX%'.Split(' '); ^
   $exArgs = @(); foreach($d in $exDirs){ if($d){ $exArgs += @('/XD', $d) } }; ^
   $args = @($src, $target, '/MIR', '/NFL', '/NDL', '/NP', '/NJH', '/NJS') + $exArgs; ^
   $p = Start-Process -FilePath robocopy -ArgumentList $args -Wait -PassThru; ^
   if ($p.ExitCode -gt 7){ throw 'Robocopy failed with exit code ' + $p.ExitCode }"

if errorlevel 1 (
  echo ZIP mirror failed. See %LOG%.
  goto :fail
)

:done
echo.
echo Update complete. Deleted files restored. ✅
echo [%DATE% %TIME%] Done >> "%LOG%"
pause >nul
exit /b 0

:fail
echo Update failed. Check "%LOG%".
pause >nul
exit /b 1
