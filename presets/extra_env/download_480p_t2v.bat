@echo off
setlocal EnableExtensions

REM Convenience wrapper to download the 480p_t2v model.

call "%~dp0run_hunyuan15_cli.bat" download --model 480p_t2v
exit /b %errorlevel%
