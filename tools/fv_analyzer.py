
# tools/fv_analyzer.py
# FrameVision pre-flight analyzer: safe read-only checks
from __future__ import annotations
import os, sys, re, json, time, hashlib
from pathlib import Path
from datetime import datetime

KEYS = {
    "settings": ["helpers/settings_tab.py"],
    "sysmon": ["helpers/sysmon.py"],
    "app": ["helpers/framevision_app.py"],
    "intro": ["helpers/intro_data.py"],
    "ui_fixups": ["helpers/ui_fixups.py"],
}

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def find_project_root(start: Path) -> Path | None:
    # Heuristic: has helpers/ and framevision_run.py
    if (start / "helpers").is_dir() and (start / "framevision_run.py").exists():
        return start
    # walk up at most 2 levels
    cur = start
    for _ in range(2):
        cur = cur.parent
        if (cur / "helpers").is_dir() and (cur / "framevision_run.py").exists():
            return cur
    return None

def quick_compile(p: Path) -> tuple[bool, str]:
    import py_compile, traceback
    try:
        py_compile.compile(str(p), doraise=True)
        return True, "OK"
    except Exception as e:
        return False, str(e)

def scan_qss(root: Path) -> list[str]:
    problems = []
    qss_dir_candidates = [root/"helpers", root/"assets", root]
    seen = set()
    for base in qss_dir_candidates:
        if not base.exists():
            continue
        for p in base.rglob("*.qss"):
            if p in seen: continue
            seen.add(p)
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            # Simple checks: unbalanced /* */, unbalanced { }, stray "::" selectors lines
            if txt.count("/*") != txt.count("*/"):
                problems.append(f"{p}: Unbalanced comment tokens (/* */)")
            if txt.count("{") != txt.count("}"):
                problems.append(f"{p}: Unbalanced braces {{ }}")
            if "FvSettingsContent" in txt and ("{:" in txt):
                problems.append(f"{p}: Suspicious '{{:' near FvSettingsContent")
    return problems

def discover_siblings(root: Path) -> list[Path]:
    # Look for other installs one level up with a helpers/ dir
    siblings = []
    parent = root.parent
    if parent.exists():
        for item in parent.iterdir():
            if item.is_dir() and (item/"helpers").is_dir() and (item/"framevision_run.py").exists():
                if item != root:
                    siblings.append(item)
    return siblings

def file_summary(root: Path, label: str) -> dict:
    out = {"root": str(root), "files": {}, "compile": {}, "flags": {}}
    for key, rels in KEYS.items():
        for rel in rels:
            p = root / rel
            out["files"][rel] = {"exists": p.exists(), "sha256": sha256_file(p) if p.exists() else ""}
            if p.suffix == ".py" and p.exists():
                ok, msg = quick_compile(p)
                out["compile"][rel] = {"ok": ok, "msg": msg}
    # Feature detection (read-only)
    try:
        st = (root/"helpers/settings_tab.py").read_text(encoding="utf-8", errors="ignore")
        out["flags"]["temp_units_row"] = ("Temperature units" in st)
    except Exception:
        out["flags"]["temp_units_row"] = False
    try:
        sm = (root/"helpers/sysmon.py").read_text(encoding="utf-8", errors="ignore")
        out["flags"]["sysmon_uses_pushbuttons"] = ("QPushButton(\"GitHub\"" in sm) or ("QPushButton('GitHub'" in sm)
        out["flags"]["sysmon_temp_formatter"] = ("def _format_temp(" in sm) and ("extras.append(_format_temp(" in sm or "°C" not in sm)
    except Exception:
        out["flags"]["sysmon_uses_pushbuttons"] = False
        out["flags"]["sysmon_temp_formatter"] = False
    try:
        fa = (root/"helpers/framevision_app.py").read_text(encoding="utf-8", errors="ignore")
        out["flags"]["hud_temp_formatter"] = ("def _format_temp_units(" in fa) or ("{_format_temp_units(" in fa)
    except Exception:
        out["flags"]["hud_temp_formatter"] = False
    try:
        idt = (root/"helpers/intro_data.py").read_text(encoding="utf-8", errors="ignore")
        out["flags"]["intro_has_presets_startup"] = ("presets\" , \"startup" in idt) or ("presets\", \"startup" in idt) or ("presets/startup" in idt.replace("\\\\","/"))
        out["flags"]["intro_uses_qsettings_dir"] = ("intro_local_dir" in idt)
    except Exception:
        out["flags"]["intro_has_presets_startup"] = False
        out["flags"]["intro_uses_qsettings_dir"] = False
    out["qss_problems"] = scan_qss(root)
    return out

def main():
    start = Path.cwd()
    root = find_project_root(start) or start
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logs_dir = root/"logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    report_lines = []
    report_lines.append(f"FrameVision Analyzer — {ts}")
    report_lines.append("="*60)
    report_lines.append(f"Detected project root: {root}")

    # Summarize current root
    cur = file_summary(root, "current")
    report_lines.append("\n[Current root files]")
    for rel, info in cur["files"].items():
        report_lines.append(f"{rel}: {'OK' if info['exists'] else 'MISSING'}  {info['sha256'][:12]}")
    report_lines.append("\n[Compile checks]")
    for rel, res in cur["compile"].items():
        report_lines.append(f"{rel}: {'OK' if res['ok'] else 'FAIL'}  {res['msg']}")
    report_lines.append("\n[Feature flags]")
    for k,v in cur["flags"].items():
        report_lines.append(f"{k}: {v}")
    if cur["qss_problems"]:
        report_lines.append("\n[QSS issues]")
        report_lines.extend([f"- {p}" for p in cur["qss_problems"]])
    else:
        report_lines.append("\n[QSS issues] none detected")

    # Compare with siblings (if any)
    sibs = discover_siblings(root)
    if sibs:
        report_lines.append("\n[Sibling installs detected]")
        for sib in sibs:
            report_lines.append(f"- {sib}")
        report_lines.append("\n[Diff vs siblings] (sha256)")
        for sib in sibs:
            ssum = file_summary(sib, "sib")
            report_lines.append(f"\n-- Compare to: {sib}")
            for rel in KEYS["settings"]+KEYS["sysmon"]+KEYS["app"]+KEYS["intro"]+KEYS["ui_fixups"]:
                sha_cur = cur["files"].get(rel,{}).get("sha256","")
                sha_sib = ssum["files"].get(rel,{}).get("sha256","")
                status = "SAME" if sha_cur and sha_cur==sha_sib else "DIFF"
                report_lines.append(f"{rel}: {status}  cur={sha_cur[:10]}  sib={sha_sib[:10]}")
    else:
        report_lines.append("\n[Sibling installs detected] none")

    report = "\n".join(report_lines)

    # Write to file + print
    out = logs_dir / f"fv_analyzer_report_{ts}.txt"
    try:
        out.write_text(report, encoding="utf-8")
    except Exception:
        pass
    print(report)

    # Also write a short JSON for quick machine checks if needed later
    try:
        (logs_dir / f"fv_analyzer_report_{ts}.json").write_text(
            json.dumps({"current": cur}, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

    print("\nTip: Run this from EACH install you use (base + installer) to compare hashes quickly.")

if __name__ == "__main__":
    main()
