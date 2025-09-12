
# -*- coding: utf-8 -*-
"""
rife_core.py — Zero-config bootstrap + device pick for RIFE.

- ensure_bootstrap(root): run on tab open or app start.
  - Ensure bundled models are extracted via model_assets.ensure_rife_models.
  - Pick device using ONNX Runtime providers if available.
  - Persist choices to QSettings so the UI/worker can read them.
"""
from __future__ import annotations
from pathlib import Path
import os

def ensure_bootstrap(root: Path | str):
    root = Path(root).resolve()
    # Extract bundled models on first run
    try:
        from helpers.model_assets import ensure_rife_models
        mdir = ensure_rife_models(root)
    except Exception:
        mdir = None

    # Persist to QSettings if available (UI side)
    try:
        from PySide6.QtCore import QSettings
        s = QSettings("FrameVision","FrameVision")
        if mdir and str(mdir):
            s.setValue("rife/models_dir", str(mdir))
    except Exception:
        pass

    # Device pick via ONNX Runtime if present
    label = "CPU"
    try:
        import onnxruntime as ort
        providers = list(ort.get_available_providers())
        if "CUDAExecutionProvider" in providers:
            label = "CUDA"
        elif "DmlExecutionProvider" in providers:
            label = "DirectML"
        else:
            label = "CPU"
    except Exception:
        label = "CPU"

    try:
        from PySide6.QtCore import QSettings
        s = QSettings("FrameVision","FrameVision")
        s.setValue("rife/device_label", label)
    except Exception:
        pass

    return {"models_dir": str(mdir) if mdir else "", "device": label}
