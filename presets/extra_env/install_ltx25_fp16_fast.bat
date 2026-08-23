@echo off
call "%~dp0install_ltx25.bat" --fp16 --no-pause
exit /b %ERRORLEVEL%
