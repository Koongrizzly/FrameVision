# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, platform
from pathlib import Path
from PySide6.QtCore import QSettings

def run():
    info = {}
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    try:
        from PySide6 import QtCore
        info["qt"] = QtCore.qVersion()
    except Exception:
        info["qt"] = "unknown"

    # Active default format
    try:
        fmt = QSettings.defaultFormat()
        info["qsettings_default_format"] = {0:"Native",1:"Reg",2:"Ini"}.get(int(fmt), str(fmt))
    except Exception as e:
        info["qsettings_default_format"] = f"error: {e}"

    # Try writing a test key to the current app/org store
    ORG, APP = "FrameVision", "FrameVision"
    s = QSettings(ORG, APP)
    s.setValue("_diag/test_key", "ok")
    s.sync()
    got = s.value("_diag/test_key", None)
    info["qsettings_native_write_ok"] = (str(got) == "ok")

    # If INI is used, find its path; otherwise report where the backend would store values
    # We also check our local ./settings/interp.json mirror
    root = Path('.').resolve()
    info["cwd"] = str(root)
    ini_guess = root / "settings" / "framevision.ini"
    info["portable_ini_exists"] = ini_guess.exists()
    info["portable_ini_path"] = str(ini_guess)

    interp_json = root / "settings" / "interp.json"
    info["interp_json_exists"] = interp_json.exists()
    info["interp_json_path"] = str(interp_json)

    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    run()
