
# scripts/post_install_fixes.py
# - Normalize helpers/tools_tab.py that may have been corrupted by earlier edits.
# - Ensure QMessageBox is imported correctly (inside parenthesized import or as its own line).
import os, io, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET = os.path.join(ROOT, "helpers", "tools_tab.py")

def normalize_tools_tab(path: str):
    if not os.path.exists(path):
        return
    # Read with utf-8-sig to strip BOM if present
    with io.open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        src = f.read()

    orig = src

    # Replace any literal backtick escape artifacts like `r`n with real newlines
    src = src.replace("`r`n", "\n").replace("`n", "\n").replace("`r", "\r")

    # Collapse any duplicate "from PySide6.QtWidgets import QMessageBox" lines at the very top
    lines = src.splitlines()
    cleaned = []
    seen_qmb_import = False
    for i, line in enumerate(lines):
        if line.strip() == "from PySide6.QtWidgets import QMessageBox":
            if seen_qmb_import:
                continue
            seen_qmb_import = True
        cleaned.append(line)
    src = "\n".join(cleaned)

    # Case 1: fix "…), QMessageBox" -> move inside parentheses
    src = re.sub(
        r"(from\s+PySide6\.QtWidgets\s+import\s*\()([^\)]*?)(\)\s*,\s*QMessageBox)",
        lambda m: m.group(1) + (m.group(2) + ", QMessageBox").replace(", ,", ", ").rstrip() + ")",
        src,
        flags=re.M,
    )

    # Case 2: parenthesized import without QMessageBox: inject
    def inject_inside_parens(match):
        head, inner, tail = match.groups()
        # Quick check to avoid duplicate
        if re.search(r"\bQMessageBox\b", inner):
            return match.group(0)
        inner2 = (inner.rstrip() + ", QMessageBox").replace(", ,", ", ")
        return f"{head}{inner2}{tail}"
    src = re.sub(
        r"(from\s+PySide6\.QtWidgets\s+import\s*\()([^\)]*?)(\))",
        inject_inside_parens,
        src,
        count=1,
        flags=re.M,
    )

    # Case 3: non-parenthesized import: append , QMessageBox if missing
    def append_qmb(match):
        head, tail = match.groups()
        items = [p.strip() for p in tail.split(",") if p.strip()]
        if "QMessageBox" not in items:
            items.append("QMessageBox")
        return head + ", ".join(items)
    src = re.sub(
        r"^(from\s+PySide6\.QtWidgets\s+import\s+)([^\n\(\)]+?)\s*$",
        append_qmb,
        src,
        flags=re.M,
    )

    # Last resort: ensure at least one QMessageBox import exists somewhere
    if "QMessageBox" not in src:
        src = "from PySide6.QtWidgets import QMessageBox\n" + src

    if src != orig:
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)

normalize_tools_tab(TARGET)
