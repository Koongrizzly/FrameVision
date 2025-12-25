@echo off
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "GGUF_DIR=%ROOT%\models\ltx\calcuis_ltxv-gguf"
set "KEEP1=ltx-video-2b-v0.9-q5_k_m.gguf"
set "KEEP2=t5xxl_fp16-q4_0.gguf"

echo [LTX] This will delete ALL extra *.gguf files in:
echo     %GGUF_DIR%
echo [LTX] Keeping only:
echo     %KEEP1%
echo     %KEEP2%
echo.

if not exist "%GGUF_DIR%" (
  echo [LTX] Nothing to do (folder not found).
  exit /b 0
)

choice /M "Delete extra GGUF files now?"
if errorlevel 2 (
  echo [LTX] Cancelled.
  exit /b 0
)

for %%F in ("%GGUF_DIR%\*.gguf") do (
  if /I not "%%~nxF"=="%KEEP1%" if /I not "%%~nxF"=="%KEEP2%" (
    echo Deleting: %%~nxF
    del /f /q "%%F" >nul 2>nul
  )
)

echo.
echo [LTX] Done.
exit /b 0
