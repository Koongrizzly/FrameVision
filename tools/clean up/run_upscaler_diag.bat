@echo off
setlocal
cd /d %~dp0
cd ..\..
if not exist .\.venv\Scripts\activate (
  echo [ERR] venv not found at .\.venv\Scripts\activate
  goto :eof
)
call .\.venv\Scripts\activate
python tools\diagnostics\upscaler_diag.py
