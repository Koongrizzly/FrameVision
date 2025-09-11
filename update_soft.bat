@echo off
setlocal EnableExtensions EnableDelayedExpansion
title FrameVision Updater

REM ====== CONFIG ======
set "REPO_URL=https://github.com/Koongrizzly/FrameVision.git"
set "BRANCH=main"
set "LOG=update.log"
REM =====================

echo.
echo ==============================
echo   FrameVision - UPDATE TOOL
echo ==============================
echo.

cd /d "%~dp0"
echo [%DATE% %TIME%] Starting update... >> "%LOG%"

where git >nul 2>nul
set "GIT_PRESENT=%ERRORLEVEL%"

if exist ".git" (
    if %GIT_PRESENT%==0 (
        echo Using Git: repository detected.
        echo [%DATE% %TIME%] Using git pull. >> "%LOG%"
        git rev-parse --is-inside-work-tree >nul 2>&1 || goto :fallback_zip

        git fetch origin %BRANCH% >> "%LOG%" 2>&1 || goto :fallback_zip

        git merge --ff-only origin/%BRANCH% >> "%LOG%" 2>&1
        if errorlevel 1 (
            echo Fast-forward failed. Trying 'git reset --hard origin/%BRANCH%'...
            git reset --hard origin/%BRANCH% >> "%LOG%" 2>&1
        )
        echo Done via git.
        goto :end_ok
    ) else (
        echo Git repo found but Git is not installed. Falling back to ZIP method.
        goto :fallback_zip
    )
)

if %GIT_PRESENT%==0 (
    echo No .git found. Initializing a lightweight git checkout...
    echo [%DATE% %TIME%] Bootstrapping git repo. >> "%LOG%"
    git init >> "%LOG%" 2>&1
    git remote add origin "%REPO_URL%" >> "%LOG%" 2>&1
    git fetch origin %BRANCH% >> "%LOG%" 2>&1 || goto :fallback_zip
    git checkout -b %BRANCH% --track origin/%BRANCH% >> "%LOG%" 2>&1 || goto :fallback_zip
    echo Done via git bootstrap.
    goto :end_ok
)

:fallback_zip
echo.
echo --- ZIP fallback path ---
echo Downloading latest %BRANCH% as ZIP from GitHub...
echo (This path is used when Git isn't available or pulls fail.)
echo [%DATE% %TIME%] Using ZIP fallback. >> "%LOG%"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Set-StrictMode -Version Latest; $ErrorActionPreference='Stop'; ^
   $tmp = Join-Path $env:TEMP ('fv_update_' + (Get-Date -Format yyyyMMdd_HHmmss)); ^
   New-Item -ItemType Directory -Path $tmp | Out-Null; ^
   $zip = Join-Path $tmp 'repo.zip'; ^
   $branch = '%BRANCH%'; ^
   $repoZip = 'https://github.com/Koongrizzly/FrameVision/archive/refs/heads/' + $branch + '.zip'; ^
   Invoke-WebRequest -Uri $repoZip -OutFile $zip; ^
   $dest = Join-Path $tmp 'unzipped'; ^
   Expand-Archive -Path $zip -DestinationPath $dest -Force; ^
   $root = Get-ChildItem $dest | Where-Object { $_.PSIsContainer } | Select-Object -First 1; ^
   $src = $root.FullName; ^
   Write-Host 'Copying files from' $src; ^
   $exDirs = @('.git','.venv','venv','ENV','outputs','models','.hf_cache','__pycache__'); ^
   $exArgs = @(); foreach($d in $exDirs){ $exArgs += @('/XD', $d) }; ^
   $target = (Get-Location).Path; ^
   $robolog = Join-Path $target '%LOG%'; ^
   $args = @($src, $target, '/E', '/XO', '/NFL', '/NDL', '/NP', '/NJH', '/NJS') + $exArgs; ^
   $p = Start-Process -FilePath robocopy -ArgumentList $args -Wait -PassThru; ^
   'ROBOCOPY EXIT CODE: ' + $p.ExitCode | Out-File -FilePath $robolog -Append; ^
   if ($p.ExitCode -gt 7){ throw 'Robocopy failed with exit code ' + $p.ExitCode }"

if errorlevel 1 goto :end_fail

echo ZIP update completed.
goto :end_ok

:end_ok
echo.
echo Update complete! ✅
echo [%DATE% %TIME%] Update completed. >> "%LOG%"
echo.
echo Press any key to close...
pause >nul
exit /b 0

:end_fail
echo.
echo Update failed. See "%LOG%" for details.
echo Press any key to close...
pause >nul
exit /b 1
