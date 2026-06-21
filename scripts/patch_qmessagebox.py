
import sys, os, io, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
target = os.path.join(ROOT, "helpers", "tools_tab.py")

if not os.path.exists(target):
    print("[patch] helpers/tools_tab.py not found; nothing to do.")
    sys.exit(0)

with io.open(target, "r", encoding="utf-8") as f:
    src = f.read()

# Add QMessageBox import if missing
if "QMessageBox" not in src:
    # Try to extend an existing PySide6.QtWidgets import
    m = re.search(r'^(from\s+PySide6\.QtWidgets\s+import\s+[^\n]*?)\s*$', src, re.M)
    if m:
        line = m.group(1)
        if "QMessageBox" not in line:
            src = src[:m.start(1)] + (line + ", QMessageBox") + src[m.end(1):]
    else:
        # Insert a new import near the top
        lines = src.splitlines(True)
        insert_at = 0
        while insert_at < len(lines) and (lines[insert_at].startswith("#!") or "coding" in lines[insert_at]):
            insert_at += 1
        lines.insert(insert_at, "from PySide6.QtWidgets import QMessageBox\n")
        src = "".join(lines)

with io.open(target, "w", encoding="utf-8", newline="\n") as f:
    f.write(src)

print("[patch] QMessageBox import ensured in helpers/tools_tab.py")
