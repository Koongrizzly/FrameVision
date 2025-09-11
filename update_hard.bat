@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title FrameVision - STRICT UPDATE (Hard, Color)

REM ====== CONFIG ======
set "BRANCH=main"
set "REPO_URL=https://github.com/Koongrizzly/FrameVision.git"
set "SELF_DELETE=0"
REM ====================
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul
set "LOGFILE=%SCRIPT_DIR%update.log"

call :_init_ui

REM Prepare an empty hooks dir to disable any repo hooks during this run
set "NO_HOOKS=%TEMP%\framevision_no_hooks"
if not exist "%NO_HOOKS%" mkdir "%NO_HOOKS%" >nul 2>&1

call :_banner "FrameVision - STRICT UPDATE ^(Hard^)"
call :_note   "Logging to: ""%LOGFILE%"""
call :_note   "This will restore deleted files and mirror origin/%BRANCH%"
echo.

call :_info "Detecting Git..."
where git >nul 2>nul || (call :_err "Git not found in PATH." & goto _end_fail)

call :_info "Checking repository status..."
git rev-parse --is-inside-work-tree >nul 2>nul || (call :_err "Not a Git repository. Hint: git clone %REPO_URL%" & goto _end_fail)

for /f "usebackq delims=" %%R in (`git remote get-url origin 2^>nul`) do set "CURRENT_REMOTE=%%R"
if defined REPO_URL if /I not "%CURRENT_REMOTE%"=="%REPO_URL%" (
  call :_warn "Origin URL differs. Setting to %REPO_URL%"
  git remote set-url origin "%REPO_URL%" >nul 2>nul
)

call :_step "Fetching latest from origin/%BRANCH%..."
call :RUN_AND_TEE git -c core.hooksPath="%NO_HOOKS%" fetch origin %BRANCH%
if errorlevel 1 (call :_err "git fetch failed." & goto _end_fail)
set "TEMPZIP=%TEMP%\fv_mirror_%RANDOM%_%RANDOM%.zip"
call :_step "Creating mirror archive from origin/%BRANCH%..."
call :RUN_AND_TEE git -c core.hooksPath="%NO_HOOKS%" archive --format=zip -o "%TEMPZIP%" origin/%BRANCH%
if errorlevel 1 (call :_err "git archive failed." & goto _cleanup_zip_fail)

call :_step "Wiping working tree ^(keeps .git and this updater^)..."
call :RUN_AND_TEE git -c core.hooksPath="%NO_HOOKS%" clean -xfd -e "%~nx0" -e ".git" -e "%TEMPZIP%"
for /f "delims=" %%I in ('dir /b /a') do (
  if /I not "%%I"==".git" if /I not "%%I"=="%~nx0" if /I not "%%I"=="%TEMPZIP%" (
    if exist "%%I\NUL" (rmdir /s /q "%%I" 2>nul) else (del /f /q "%%I" 2>nul)
  )
)

call :_step "Expanding mirror into working tree..."
where tar >nul 2>nul
if errorlevel 1 goto :_use_powershell_extract
call :RUN_AND_TEE tar -xf "%TEMPZIP%"
goto :_after_extract
:_use_powershell_extract
call :RUN_AND_TEE powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%TEMPZIP%' -DestinationPath '.' -Force"
:_after_extract
if errorlevel 1 (call :_err "extract failed." & goto _cleanup_zip_fail)

del "%TEMPZIP%" >nul 2>&1
call :_ok "Strict update complete."
echo.
if "%SELF_DELETE%"=="1" (
  call :_note "This updater will remove itself now..."
  start "" cmd /d /c ping 127.0.0.1 -n 2 ^>nul ^& del /q "%~f0"
)

call :_note "Press any key to close..."
pause >nul
popd >nul
exit /b 0

:RUN_AND_TEE
setlocal
set "TMP=%TEMP%\runtee_%RANDOM%_%RANDOM%.log"
cmd /d /c %* >"%TMP%" 2>&1
set "_RC=%ERRORLEVEL%"
type "%TMP%"
>>"%LOGFILE%" type "%TMP%"
del "%TMP%" >nul 2>&1
endlocal
exit /b %_RC%

:_stamp
for /f "tokens=1-3 delims=/: " %%a in ("%date%") do set "D=%%a %%b %%c"
set "TS=[%DATE% %TIME%] "
exit /b 0

:_log
setlocal
call :_stamp
>>"%LOGFILE%" echo %TS%%~1
endlocal & exit /b 0

:_init_ui
for /f "delims=" %%A in ('echo prompt $E^| cmd') do set "ESC=%%A"
if not defined ESC goto :_noansi
set "CSI=%ESC%["
set "RESET=%ESC%[0m"
set "BOLD=%ESC%[1m"
set "DIM=%ESC%[2m"
set "FGOK=%ESC%[32m"
set "FGWARN=%ESC%[33m"
set "FGERR=%ESC%[31m"
set "FGINFO=%ESC%[36m"
set "FGNOTE=%ESC%[35m"
set "FGBAR=%ESC%[94m"
set "USE_ANSI=1"
goto :eof
:_noansi
set "USE_ANSI=0"
goto :eof

:_bar
if "%USE_ANSI%"=="1" goto :_bar_color
echo(======================================
goto :eof
:_bar_color
echo(%FGBAR%======================================%RESET%
goto :eof
:_banner
call :_bar
if "%USE_ANSI%"=="1" goto :_banner_color
echo( %~1 
goto :_banner_end
:_banner_color
echo(%BOLD%%FGINFO% %~1 %RESET%
:_banner_end
call :_bar
exit /b 0
:_info
if "%USE_ANSI%"=="1" goto :_info_color
echo( %~1 
goto :_info_end
:_info_color
echo(%FGINFO%%~1%RESET%
:_info_end
call :_log "%~1"
exit /b 0
:_note
if "%USE_ANSI%"=="1" goto :_note_color
echo( %~1 
goto :_note_end
:_note_color
echo(%FGNOTE%%~1%RESET%
:_note_end
call :_log "%~1"
exit /b 0
:_warn
if "%USE_ANSI%"=="1" goto :_warn_color
echo( WARN: %~1 
goto :_warn_end
:_warn_color
echo(%FGWARN%WARN: %~1%RESET%
:_warn_end
call :_log "WARN: %~1"
exit /b 0
:_err
if "%USE_ANSI%"=="1" goto :_err_color
echo( ERROR: %~1 
goto :_err_end
:_err_color
echo(%FGERR%ERROR: %~1%RESET%
:_err_end
call :_log "ERROR: %~1"
exit /b 0
:_ok
if "%USE_ANSI%"=="1" goto :_ok_color
echo( %~1 
goto :_ok_end
:_ok_color
echo(%FGOK%%~1%RESET%
:_ok_end
call :_log "%~1"
exit /b 0
:_step
if "%USE_ANSI%"=="1" goto :_step_color
echo( %~1 
goto :_step_end
:_step_color
echo(%BOLD%%FGINFO%%~1%RESET%
:_step_end
call :_log "%~1"
exit /b 0
:_cleanup_zip_fail
del "%TEMPZIP%" >nul 2>&1
goto :_end_fail

:_end_fail
echo.
call :_err "Update failed. See ""%LOGFILE%"" for details."
echo.
call :_note "Press any key to close..."
pause >nul
popd >nul
exit /b 1