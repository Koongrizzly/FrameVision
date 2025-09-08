"%~dp0..\.\.venv\Scripts\activate" && cd /d "%~dp0.."
echo === Describer settings probe ===
python - <<PY
from PySide6.QtCore import QSettings
s=QSettings("FrameVision","FrameVision"); s.sync()
print("describe_detail_level =", s.value("describe_detail_level"))
print("describe_decode_style =", s.value("describe_decode_style"))
print("describe_promptify =", s.value("describe_promptify"))
print("describe_negative =", s.value("describe_negative"))
PY
pause
