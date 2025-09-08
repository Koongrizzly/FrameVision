@echo off
setlocal
cd /d %~dp0
cd ..\..

REM Try to use venv if it exists, else fall back to system Python
set PYEXE=
if exist .\.venv\Scripts\python.exe set PYEXE=.\.venv\Scripts\python.exe
if "%PYEXE%"=="" (
  for %%P in (python.exe py.exe) do (
    where %%P >nul 2>nul && (
      set PYEXE=%%P
      goto :found
    )
  )
)
:found
if "%PYEXE%"=="" (
  echo [ERR] Could not find Python. Install Python or activate your venv, then run:
  echo     python tools\diagnostics\repo_sanity.py
  goto :eof
)

"%PYEXE%" tools\diagnostics\repo_sanity.py
