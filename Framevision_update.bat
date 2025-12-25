@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem FrameVision Updater - "Framevision_update.bat"
rem Repo: https://github.com/Koongrizzly/FrameVision (branch: main)
rem ============================================================

rem --- ensure we run from the script's own folder ---
pushd "%~dp0"

rem --- OPTIONAL: force UTF-8 (set to 1 to enable) ---
set "FORCE_UTF8="
if /I "%FORCE_UTF8%"=="1" (
  for /f "tokens=3 delims=: " %%C in ('chcp') do set "_OLDCP=%%C"
  chcp 65001 >nul
)

rem --- detect current codepage ---
for /f "tokens=3 delims=: " %%C in ('chcp') do set "CURCP=%%C"
set "UNICODE_OK=0"
if "%CURCP%"=="65001" set "UNICODE_OK=1"

rem --- initialize ANSI colors (Windows 10+ / Windows Terminal) ---
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
set "RED=%ESC%[31m"
set "BGRED=%ESC%[41m"

rem --- constants ---
set "REPO_OWNER=Koongrizzly"
set "REPO_NAME=FrameVision"
set "BRANCH=main"
set "ZIP_URL=https://github.com/%REPO_OWNER%/%REPO_NAME%/archive/refs/heads/%BRANCH%.zip"

rem --- paths (avoid nested single-quotes in for /f) ---
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%I"
set "WORKDIR=%CD%"
set "LOGDIR=%WORKDIR%\logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "LOGFILE=%LOGDIR%\update_%TS%.log"

rem --- build a unique temp folder ---
set "TMPROOT=%TEMP%\FV_UPD_%RANDOM%_%RANDOM%"
set "TMPZIP=%TMPROOT%\src.zip"
set "TMPSRC=%TMPROOT%\src"
set "SRCROOT=%TMPSRC%\%REPO_NAME%-%BRANCH%"

rem --- common Robocopy switches ---
set "RC_COMMON=/E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP /BYTES /FP"

call :banner
call :menu
goto :end

:banner
  cls
  echo.
  if "%UNICODE_OK%"=="1" goto :banner_unicode
  goto :banner_ascii

:banner_unicode
  set "ARW=→"
  echo %TITLE%╔═════════════════════════════════════════════════════════╗%RST%
  echo %TITLE%║%RST%                 %BOLD%FrameVision Updater%RST%                      %TITLE%║%RST%
  echo %TITLE%║%RST%              %DIM%GitHub !ARW! %REPO_OWNER%/%REPO_NAME% ^(%BRANCH%^) %RST%            %TITLE%║%RST%
  echo %TITLE%╚═════════════════════════════════════════════════════════╝%RST%
  echo.
  goto :eof

:banner_ascii
  rem Use escaped > to avoid redirection (ARW = -^>)
  set "ARW=-^>"
  echo %TITLE%===========================================================%RST%
  echo %BOLD%FrameVision Updater%RST%
  echo %DIM%GitHub !ARW! %REPO_OWNER%/%REPO_NAME% ^(%BRANCH%^) %RST%
  echo %TITLE%===========================================================%RST%
  echo.
  goto :eof

:menu
  echo %ITEM%[1]%RST% Update %BOLD%without overwriting%RST% (only add new files)
  echo %ITEM%[2]%RST% Update %BOLD%with overwrite%RST% (update changed + add new)
  echo %ITEM%[3]%RST% %BOLD%RESET (FULL)%RST% !ARW! remove everything and replace with latest main
  echo %ITEM%[4]%RST% %BOLD%RESET (SAFE)%RST% !ARW! restore app files from main but keep %DIM%output\%RST%, %DIM%presets\setsave\%RST%, and %DIM%models\%RST%
  echo %ITEM%[Q]%RST% Quit
  echo.
  choice /C 1234Q /N /M "%BOLD%Select an option (1-4, Q): %RST%"
  if errorlevel 5 goto :end
  if errorlevel 4 goto :reset_safe
  if errorlevel 3 goto :reset_full
  if errorlevel 2 goto :update_overwrite
  if errorlevel 1 goto :update_no_overwrite
  goto :menu

:prepare_temp
  where powershell >nul 2>&1
  if errorlevel 1 (
    echo %RED%PowerShell is required but not found. Please install PowerShell and try again.%RST%
    goto :cleanup_temp_and_menu
  )
  if exist "%TMPROOT%" rmdir /s /q "%TMPROOT%" >nul 2>&1
  mkdir "%TMPROOT%" >nul 2>&1
  mkdir "%TMPSRC%" >nul 2>&1
  rem download zip
  echo %NOTE%Downloading latest "%REPO_NAME% (%BRANCH%)"...%RST%
  powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri '%ZIP_URL%' -OutFile '%TMPZIP%' -ErrorAction Stop } catch { Write-Host 'DOWNLOAD_ERROR'; exit 1 }"
  if errorlevel 1 (
     echo %RED%Failed to download repository ZIP.%RST%
     goto :cleanup_temp_and_menu
  )
  rem extract zip
  echo %NOTE%Extracting archive...%RST%
  powershell -NoProfile -Command "Add-Type -A 'System.IO.Compression.FileSystem'; [IO.Compression.ZipFile]::ExtractToDirectory('%TMPZIP%','%TMPSRC%')"
  if errorlevel 1 (
     echo %RED%Failed to extract repository ZIP.%RST%
     goto :cleanup_temp_and_menu
  )
  if not exist "%SRCROOT%\" (
     echo %RED%Unexpected archive structure. Could not find "%SRCROOT%".%RST%
     goto :cleanup_temp_and_menu
  )
  goto :eof

:update_no_overwrite
  call :banner
  echo %BLUE%Mode:%RST% Safe update %DIM%(no overwrite)%RST%
  echo %DIM%Source:%RST% "%SRCROOT%"
  echo %DIM%Dest  :%RST% "%WORKDIR%"
  call :prepare_temp || goto :menu

  echo %NOTE%Planning changes (dry-run)...%RST%
  set "DRYLOG=%LOGDIR%\dryrun_safe_%TS%.txt"
  rem List files that would be copied if we only add new (no overwrites)
  robocopy "%SRCROOT%" "%WORKDIR%" /XC /XN /XO %RC_COMMON% /L > "%DRYLOG%"
  for /f %%C in ('type "%DRYLOG%" ^| find /c /v ""') do set "COUNT=%%C"
  echo %GREEN%New files to add:%RST% !COUNT!
  if "!COUNT!"=="0" (
      echo %GREEN%Nothing to do. You're already up to date ^(for new files^).%RST%
      goto :cleanup_temp_and_menu
  )

  echo %NOTE%Applying changes...%RST%
  robocopy "%SRCROOT%" "%WORKDIR%" /XC /XN /XO %RC_COMMON% >> "%LOGFILE%"
  echo %GREEN%Done.%RST%
  echo.
  echo %BOLD%Log:%RST% "%LOGFILE%"
  echo %DIM%^(This log contains the list of new files that were added.^)%RST%
  goto :cleanup_temp_and_menu

:update_overwrite
  call :banner
  echo %BLUE%Mode:%RST% Update with overwrite
  echo %DIM%Source:%RST% "%SRCROOT%"
  echo %DIM%Dest  :%RST% "%WORKDIR%"
  call :prepare_temp || goto :menu

  echo %NOTE%Planning changes (dry-run)...%RST%
  set "DRYLOG=%LOGDIR%\dryrun_overwrite_%TS%.txt"
  robocopy "%SRCROOT%" "%WORKDIR%" %RC_COMMON% /L > "%DRYLOG%"
  for /f %%C in ('type "%DRYLOG%" ^| find /c /v ""') do set "COUNT=%%C"
  echo %GREEN%Files to update/add:%RST% !COUNT!
  if "!COUNT!"=="0" (
      echo %GREEN%Nothing to do. You're already up to date.%RST%
      goto :cleanup_temp_and_menu
  )

  echo %NOTE%Applying changes...%RST%
  robocopy "%SRCROOT%" "%WORKDIR%" %RC_COMMON% >> "%LOGFILE%"
  echo %GREEN%Done.%RST%
  echo.
  echo %BOLD%Log:%RST% "%LOGFILE%"
  echo %DIM%^(This log contains the list of files that were updated or added.^)%RST%
  goto :cleanup_temp_and_menu

:reset_full
  call :banner
  echo %BGRED%%BOLD% WARNING %RST% %RED%This will DELETE everything in:%RST%
  echo     %BOLD%%WORKDIR%%RST%
  echo %RED%and replace it with the latest files from GitHub "%BRANCH%" branch.%RST%
  echo %NOTE%Before continuing, BACK UP your "%BOLD%output\%RST%%NOTE%" and "%BOLD%presets\setsave\%RST%%NOTE%" folders if you need them.%RST%
  echo.
  choice /C YN /M "Proceed with FULL RESET? (Y/N): "
  if errorlevel 2 goto :menu

  call :prepare_temp || goto :menu

  echo %NOTE%Resetting...%RST%
  rem Mirror source to destination; exclude this updater so it isn't deleted mid-run.
  robocopy "%SRCROOT%" "%WORKDIR%" /MIR %RC_COMMON% /XF "Framevision_update.bat" >> "%LOGFILE%"
  echo %GREEN%Full reset completed.%RST%
  echo.
  echo %BOLD%Log:%RST% "%LOGFILE%"
  goto :cleanup_temp_and_menu

:reset_safe
  call :banner
  echo %BLUE%Mode:%RST% Reset (safe) !ARW! restore app files but keep user data
  echo %NOTE%Keeping:%RST% "%BOLD%output\%RST%", "%BOLD%presets\setsave\%RST%", "%BOLD%models\%RST%"
  call :prepare_temp || goto :menu

  echo %NOTE%Resetting (preserving user folders)...%RST%
  robocopy "%SRCROOT%" "%WORKDIR%" /MIR %RC_COMMON% /XF "Framevision_update.bat" /XD "output" "presets\setsave" "models" >> "%LOGFILE%"
  echo %GREEN%Safe reset completed.%RST%
  echo.
  echo %BOLD%Log:%RST% "%LOGFILE%"
  goto :cleanup_temp_and_menu

:cleanup_temp_and_menu
  if exist "%TMPROOT%" rmdir /s /q "%TMPROOT%" >nul 2>&1
  echo.
  echo %GRAY%Press any key to return to menu...%RST%
  pause >nul
  call :banner
  goto :menu

:end
  rem restore original codepage if we forced UTF-8
  if defined _OLDCP chcp %_OLDCP% >nul
  rem return to the original dir and exit
  popd >nul 2>&1
  endlocal
  exit /b 0
