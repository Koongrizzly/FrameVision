
import os, io, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
target = os.path.join(ROOT, "helpers", "tools_tab.py")

if not os.path.exists(target):
    print("[fix] helpers/tools_tab.py not found here. Place this ZIP in your FrameVision root and run again.")
    sys.exit(1)

with io.open(target, "r", encoding="utf-8") as f:
    src = f.read()

orig = src

# Case A: parenthesized import followed by ', QMessageBox' OUTSIDE the parens
# e.g. from PySide6.QtWidgets import (A, B, C), QMessageBox
pat_a = re.compile(r'^(from\s+PySide6\.QtWidgets\s+import\s*\()([^)]*)(\)\s*,\s*QMessageBox\s*)$', re.M)
if pat_a.search(src):
    def _fix_a(m):
        before, inside, after = m.groups()
        items = [s.strip() for s in inside.split(",") if s.strip()]
        if "QMessageBox" not in items:
            items.append("QMessageBox")
        return f"{before}{', '.join(items)})"
    src = pat_a.sub(_fix_a, src)

# Case C: non-parenthesized import; append , QMessageBox if missing
pat_c = re.compile(r'^(from\s+PySide6\.QtWidgets\s+import\s+)([^\n()]+?)\s*$', re.M)
def _fix_c(m):
    head, tail = m.groups()
    parts = [p.strip() for p in tail.split(",")]
    if "QMessageBox" not in parts:
        parts.append("QMessageBox")
    return f"{head}{', '.join([p for p in parts if p])}"
src = pat_c.sub(_fix_c, src)

# Last resort: if no import from QtWidgets exists, insert a dedicated line at top
if "from PySide6.QtWidgets import" not in src:
    src = "from PySide6.QtWidgets import QMessageBox\n" + src

if src != orig:
    with io.open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print("[fix] Updated helpers/tools_tab.py import line for QMessageBox.")
else:
    print("[fix] No change needed (already correct).")
