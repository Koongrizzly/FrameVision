# helpers/qwen2512.py
# Qwen Image 2512 (GGUF) tool for FrameVision Tools tab.
# Uses sd-cli (stable-diffusion.cpp) under the hood.

from __future__ import annotations
import os, json, shlex, time, shutil
from pathlib import Path

import unicodedata

def _sanitize_prompt_ascii(p: str) -> str:
    """Make prompts safe for sd-cli builds that assert on non-ASCII bytes.

    Some Windows debug builds of stable-diffusion.cpp call ctype() on `char` without casting
    to unsigned char, which triggers a UCRT assert for bytes > 0x7F. We normalize common
    punctuation and strip remaining non-ASCII characters.
    """
    try:
        if not p:
            return ""
        # Normalize common “smart” punctuation to plain ASCII first
        p2 = (p.replace("\u00a0", " ")
                .replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'")
                .replace("—", "-").replace("–", "-")
                .replace("…", "..."))
        p2 = unicodedata.normalize("NFKD", p2)
        p2 = p2.encode("ascii", "ignore").decode("ascii", errors="ignore")
        # Collapse whitespace a bit
        p2 = " ".join(p2.split())
        return p2
    except Exception:
        return p


from PySide6.QtCore import Qt, QProcess, QObject, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QCheckBox,
    QFileDialog, QSizePolicy
)

class _ProcLogger(QObject):
    line = Signal(str)
    finished = Signal(int)

def _app_root() -> Path:
    # helpers/ is one level under the app root
    return Path(__file__).resolve().parent.parent

def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _open_folder(path: Path) -> None:
    try:
        os.startfile(str(path))  # type: ignore[attr-defined]
    except Exception:
        pass

def _settings_path(root: Path) -> Path:
    return root / "presets" / "setsave" / "qwen2512.json"

def _load_settings(root: Path) -> dict:
    d = {
        "width": 1024,
        "height": 1024,
        "steps": 40,
        "cfg": 2.5,
        "sampler": "euler",
        "flow_shift": 3,
        "seed": "",
        "offload_cpu": False,
        "prompt": "a cute creature made of clay, studio lighting, highly detailed",
        "lora_enabled": False,
        "lora_path": "",
        "lora_scale": 1.0
    }
    try:
        p = _settings_path(root)
        if p.exists():
            d.update(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        pass
    return d

def _save_settings(root: Path, d: dict) -> None:
    try:
        p = _settings_path(root)
        _ensure_dir(p.parent)
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass

def _find_sd_cli(root: Path) -> str:
    """Resolve sd-cli.exe.

    This tool can download its own sd-cli into /qwen2512/bin, but we also support reusing any existing sd-cli on the system.
    We try common locations first, then scan /models/* for sd-cli.exe, then PATH.
    """
    # Common bundled locations
    candidates = [
        root / ".qwen2512" / "bin" / "sd-cli.exe",
        root / "presets" / "bin" / "sd-cli.exe",
        root / "sd-cli.exe",
        root / "models" / "Z-Image-Turbo GGUF" / "bin" / "sd-cli.exe",
        root / "stable-diffusion.cpp" / "build" / "Release" / "sd-cli.exe",
    ]
    for c in candidates:
        try:
            if c.exists():
                return str(c)
        except Exception:
            pass

    # Scan existing model tool folders (fast: only one level deep under /models)
    models_dir = root / "models"
    try:
        if models_dir.exists():
            for child in models_dir.iterdir():
                if not child.is_dir():
                    continue
                for c in (child / "sd-cli.exe", child / "bin" / "sd-cli.exe"):
                    try:
                        if c.exists():
                            return str(c)
                    except Exception:
                        pass
    except Exception:
        pass

    # PATH lookup
    for name in ("sd-cli.exe", "sd-cli"):
        try:
            w = shutil.which(name)
            if w:
                return w
        except Exception:
            pass
    return "sd-cli"

def _pick_file(model_dir: Path, must_contain: list[str], suffix: str) -> Path | None:
    try:
        for p in sorted(model_dir.glob(f"*{suffix}")):
            name = p.name.lower()
            ok = True
            for s in must_contain:
                if s.lower() not in name:
                    ok = False
                    break
            if ok:
                return p
    except Exception:
        pass
    return None

def _model_paths(root: Path) -> tuple[Path | None, Path | None, Path | None]:
    md = root / "models" / "Qwen-Image-2512 GGUF"
    # Diffusion GGUF (Q4)
    diffusion = _pick_file(md, ["qwen", "image", "2512", "q4"], ".gguf")
    # LLM GGUF (Q4) — Unsloth suggests Qwen2.5-VL-7B-Instruct
    llm = _pick_file(md, ["qwen2.5", "vl", "7b", "instruct", "q4"], ".gguf")
    # VAE safetensors
    vae = md / "qwen_image_vae.safetensors"
    if not (vae.exists() if vae else False):
        vae = _pick_file(md, ["qwen", "vae"], ".safetensors")
    return diffusion, llm, vae if (vae and vae.exists()) else None

def _build_cmd(root: Path, prompt: str, w: int, h: int, steps: int, cfg: float,
               sampler: str, flow_shift: int, seed: str, offload_cpu: bool,
               lora_enabled: bool, lora_path: str, lora_scale: float) -> tuple[list[str], Path]:
    sd = _find_sd_cli(root)
    diffusion, llm, vae = _model_paths(root)
    out_dir = root / "output" / "qwen2512"
    _ensure_dir(out_dir)
    out_path = out_dir / f"qwen2512_{int(time.time())}.png"

    if diffusion is None or llm is None or vae is None:
        missing = []
        if diffusion is None: missing.append("diffusion GGUF (Q4)")
        if llm is None: missing.append("LLM GGUF (Qwen2.5-VL-7B-Instruct Q4)")
        if vae is None: missing.append("VAE safetensors")
        raise RuntimeError("Missing required model files: " + ", ".join(missing))

    # Optional LoRA (sd-cli parses <lora:NAME:W> tags into its internal loras map).
    final_prompt = prompt
    lora_dir = None
    if lora_enabled and str(lora_path).strip():
        lp = Path(str(lora_path).strip().strip('"').strip("'"))
        if not lp.is_absolute():
            # Allow relative names (or just a filename) under models/loras/
            lp = (root / "models" / "loras" / lp).resolve()
        if not lp.exists():
            raise RuntimeError(f"LoRA not found: {lp}")
        lora_dir = lp.parent
        lora_name = lp.stem
        try:
            wgt = float(lora_scale)
        except Exception:
            wgt = 1.0
        final_prompt = f"<lora:{lora_name}:{wgt}> {prompt}"

    # Qwen-Image models must be passed via --diffusion-model (NOT -m).
    # Using -m triggers SD "version detection" and fails with:
    #   get sd version from file failed
    # Reference: Unsloth's stable-diffusion.cpp tutorial for Qwen-Image-2512.
    args = [
        sd,
        "--diffusion-model", str(diffusion),
        "--vae", str(vae),
        "--llm", str(llm),
        "--sampling-method", str(sampler),
        "--cfg-scale", str(cfg),
        "--steps", str(int(steps)),
        "-W", str(int(w)),
        "-H", str(int(h)),
        "--diffusion-fa",
        "--flow-shift", str(int(flow_shift)),
        "-p", _sanitize_prompt_ascii(final_prompt),
        "-o", str(out_path),
    ]
    if lora_dir is not None:
        args += ["--lora-model-dir", str(lora_dir), "--lora-apply-mode", "at_runtime"]
    if seed.strip():
        args += ["--seed", seed.strip()]
    if offload_cpu:
        args += ["--offload-to-cpu"]
    return args, out_path

def install_qwen2512_tool(tools_pane, section) -> None:
    """
    Populate a CollapsibleSection with the WAN/Qwen2512 UI.
    tools_pane: ToolsPane instance (from helpers/tools_tab.py)
    section: CollapsibleSection
    """
    root = _app_root()
    s = _load_settings(root)

    wrap = QWidget()
    lay = QVBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)

    # Title / status row
    row0 = QHBoxLayout()
    lbl = QLabel("Run Qwen-Image-2512 (GGUF) via sd-cli (stable-diffusion.cpp).")
    lbl.setWordWrap(True)
    row0.addWidget(lbl, 1)

    btn_install = QPushButton("Install / Update…")
    btn_install.setToolTip("Runs presets/extra_env/qwen2512_install.bat to set up the environment and download Q4 models.")
    row0.addWidget(btn_install)

    btn_models = QPushButton("Open Models")
    btn_models.setToolTip("Open models/Qwen-Image-2512 GGUF/")
    row0.addWidget(btn_models)

    btn_out = QPushButton("Open Output")
    btn_out.setToolTip("Open output/qwen2512/")
    row0.addWidget(btn_out)

    lay.addLayout(row0)

    # Prompt
    prompt = QPlainTextEdit()
    prompt.setPlaceholderText("Prompt…")
    prompt.setPlainText(s.get("prompt", ""))
    prompt.setToolTip("Text prompt for the image generation.")
    prompt.setMinimumHeight(90)
    lay.addWidget(prompt)

    # LoRA (optional)
    row_lora = QHBoxLayout()
    cb_lora = QCheckBox("Use LoRA")
    cb_lora.setChecked(bool(s.get("lora_enabled", False)))
    cb_lora.setToolTip("Enable a LoRA for this run (sd-cli via <lora:NAME:W> + --lora-model-dir).")

    ed_lora = QLineEdit(str(s.get("lora_path", "")))
    ed_lora.setPlaceholderText("LoRA path (optional)…")
    ed_lora.setToolTip("Path to a .safetensors LoRA file. You can also paste a filename relative to models/loras/.")

    btn_lora = QPushButton("Browse…")
    btn_lora.setToolTip("Select a LoRA .safetensors file.")

    sp_lora = QDoubleSpinBox()
    sp_lora.setRange(0.0, 2.0)
    sp_lora.setDecimals(2)
    sp_lora.setSingleStep(0.05)
    sp_lora.setValue(float(s.get("lora_scale", 1.0)))
    sp_lora.setToolTip("LoRA strength (weight). 1.0 is typical; try 0.6–1.3.")

    row_lora.addWidget(cb_lora)
    row_lora.addWidget(ed_lora, 1)
    row_lora.addWidget(btn_lora)
    row_lora.addWidget(QLabel("Strength"))
    row_lora.addWidget(sp_lora)

    lay.addLayout(row_lora)


    # Settings row
    row1 = QHBoxLayout()

    sp_w = QSpinBox(); sp_w.setRange(64, 4096); sp_w.setValue(int(s.get("width", 1024)))
    sp_w.setToolTip("Width in pixels.")
    sp_h = QSpinBox(); sp_h.setRange(64, 4096); sp_h.setValue(int(s.get("height", 1024)))
    sp_h.setToolTip("Height in pixels.")
    sp_steps = QSpinBox(); sp_steps.setRange(1, 200); sp_steps.setValue(int(s.get("steps", 40)))
    sp_steps.setToolTip("Number of diffusion steps.")
    sp_cfg = QDoubleSpinBox(); sp_cfg.setRange(0.0, 20.0); sp_cfg.setDecimals(2); sp_cfg.setSingleStep(0.1); sp_cfg.setValue(float(s.get("cfg", 2.5)))
    sp_cfg.setToolTip("CFG scale (guidance). Qwen-Image often uses low CFG (around 2–3).")
    cmb_sampler = QComboBox()
    cmb_sampler.addItems(["euler", "euler_a", "heun", "dpm++2m", "dpm++2m_sde"])
    try:
        want = str(s.get("sampler", "euler"))
        ix = cmb_sampler.findText(want)
        if ix >= 0: cmb_sampler.setCurrentIndex(ix)
    except Exception:
        pass
    cmb_sampler.setToolTip("Sampling method (sampler).")

    sp_flow = QSpinBox(); sp_flow.setRange(0, 20); sp_flow.setValue(int(s.get("flow_shift", 3)))
    sp_flow.setToolTip("Qwen-Image flow-shift value (Unsloth example uses 3).")

    ed_seed = QLineEdit(str(s.get("seed", "")))
    ed_seed.setPlaceholderText("Seed (optional)")
    ed_seed.setToolTip("Optional random seed. Leave empty for random.")

    cb_offload = QCheckBox("Offload to CPU")
    cb_offload.setChecked(bool(s.get("offload_cpu", False)))
    cb_offload.setToolTip("Adds --offload-to-cpu to reduce VRAM usage (slower).")

    def _lblbox(t, wdg):
        box = QVBoxLayout()
        lb = QLabel(t); lb.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lb.setStyleSheet("opacity: 0.85;")
        box.addWidget(lb)
        box.addWidget(wdg)
        return box

    row1.addLayout(_lblbox("W", sp_w))
    row1.addLayout(_lblbox("H", sp_h))
    row1.addLayout(_lblbox("Steps", sp_steps))
    row1.addLayout(_lblbox("CFG", sp_cfg))
    row1.addLayout(_lblbox("Sampler", cmb_sampler))
    row1.addLayout(_lblbox("Flow", sp_flow))
    row1.addLayout(_lblbox("Seed", ed_seed))
    row1.addWidget(cb_offload)

    lay.addLayout(row1)

    # Run row
    row2 = QHBoxLayout()
    btn_run = QPushButton("Run")
    btn_run.setToolTip("Run sd-cli with the current settings.")
    btn_pick = QPushButton("Pick model folder…")
    btn_pick.setToolTip("Select a custom models folder (copies path into a quick note in the log).")
    btn_open_last = QPushButton("Open last image")
    btn_open_last.setToolTip("Open the last generated image (if any).")
    row2.addWidget(btn_run)
    row2.addWidget(btn_pick)
    row2.addWidget(btn_open_last)
    row2.addStretch(1)
    lay.addLayout(row2)

    log = QTextEdit()
    log.setReadOnly(True)
    log.setMinimumHeight(220)
    log.setToolTip("Logs from the installer and sd-cli.")
    lay.addWidget(log, 1)

    # internal state
    tools_pane._qwen2512_last_out = None
    tools_pane._qwen2512_proc = None

    def _append(msg: str) -> None:
        try:
            log.append(msg.rstrip("\n"))
        except Exception:
            pass

    def _snapshot_and_save() -> None:
        d = {
            "width": sp_w.value(),
            "height": sp_h.value(),
            "steps": sp_steps.value(),
            "cfg": float(sp_cfg.value()),
            "sampler": cmb_sampler.currentText(),
            "flow_shift": sp_flow.value(),
            "seed": ed_seed.text(),
            "offload_cpu": cb_offload.isChecked(),
            "prompt": prompt.toPlainText(),
            "lora_enabled": cb_lora.isChecked(),
            "lora_path": ed_lora.text(),
            "lora_scale": float(sp_lora.value()),
        }
        _save_settings(root, d)

    # Save on changes (cheap + safe)
    for wdg, sig in [
        (sp_w, sp_w.valueChanged),
        (sp_h, sp_h.valueChanged),
        (sp_steps, sp_steps.valueChanged),
        (sp_cfg, sp_cfg.valueChanged),
        (cmb_sampler, cmb_sampler.currentIndexChanged),
        (sp_flow, sp_flow.valueChanged),
        (ed_seed, ed_seed.textChanged),
        (cb_offload, cb_offload.toggled),
        (cb_lora, cb_lora.toggled),
        (ed_lora, ed_lora.textChanged),
        (sp_lora, sp_lora.valueChanged),
    ]:
        try:
            sig.connect(lambda *_: _snapshot_and_save())
        except Exception:
            pass
    try:
        prompt.textChanged.connect(lambda: _snapshot_and_save())
    except Exception:
        pass

    def _run_process(cmd: list[str], cwd: Path) -> QProcess:
        proc = QProcess(tools_pane)

        # Ensure DLLs next to sd-cli.exe can be found (stable-diffusion.dll etc.)
        wd = cwd
        try:
            exe_path = Path(cmd[0])
            if exe_path.is_absolute() and exe_path.exists():
                wd = exe_path.parent
        except Exception:
            pass
        proc.setWorkingDirectory(str(wd))

        # Merge stdout/stderr so we don't miss errors.
        proc.setProcessChannelMode(QProcess.MergedChannels)

        # Line-buffered reader (QProcess chunks can split lines; sd-cli also uses \r updates).
        buf = {"text": ""}   # mutable holder for closure
        saw_text = {"any": False}

        def _drain(force: bool = False):
            s = buf["text"]
            if force and s:
                # treat any remaining partial as a line
                line = s.split("\r")[-1].rstrip()
                if line:
                    saw_text["any"] = True
                    _append(line)
                buf["text"] = ""
                return

            while "\n" in s:
                line, s = s.split("\n", 1)
                line = line.split("\r")[-1].rstrip()
                if line:
                    saw_text["any"] = True
                    _append(line)
            buf["text"] = s

        def _on_ready():
            try:
                data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
            except Exception:
                data = ""
            if not data:
                return
            buf["text"] += data
            _drain(force=False)

        proc.readyReadStandardOutput.connect(_on_ready)

        def _on_finished(code, _status):
            _drain(force=True)
            _append(f"[done] exit code: {int(code)}")
            if int(code) != 0:
                _append("[hint] If the log is mostly empty or shows blank [INFO] lines, your sd-cli build may be too old for Qwen-Image flags like --llm/--flow-shift.")
                _append("[hint] Use Install / Update… to fetch a newer sd-cli into qwen2512/bin, then try again.")

        proc.finished.connect(_on_finished)

        def _on_error(_err):
            _append(f"[ERR] process error: {proc.errorString()}")
        try:
            proc.errorOccurred.connect(_on_error)
        except Exception:
            pass

        # Start
        proc.start(cmd[0], cmd[1:])
        try:
            if not proc.waitForStarted(3000):
                _append(f"[ERR] Failed to start: {proc.errorString()}")
        except Exception:
            pass
        return proc

    def do_install():
        bat = root / "presets" / "extra_env" / "qwen2512_install.bat"
        if not bat.exists():
            _append("[ERR] Missing installer: presets/extra_env/qwen2512_install.bat")
            return
        _append(f"[install] {bat}")
        tools_pane._qwen2512_proc = _run_process(["cmd.exe", "/c", str(bat)], root)

    def do_open_models():
        _open_folder(root / "models" / "Qwen-Image-2512 GGUF")

    def do_open_out():
        _open_folder(root / "output" / "qwen2512")

    def do_pick_models_folder():
        try:
            d = QFileDialog.getExistingDirectory(tools_pane, "Select models folder", str(root / "models"))
        except Exception:
            d = ""
        if d:
            _append(f"[info] Selected folder: {d}")
            _open_folder(Path(d))


    def do_pick_lora():
        try:
            f, _ = QFileDialog.getOpenFileName(tools_pane, "Select LoRA (.safetensors)", str(root / "models" / "loras"), "LoRA (*.safetensors)")
        except Exception:
            f = ""
        if f:
            ed_lora.setText(f)
            cb_lora.setChecked(True)
            _append(f"[info] LoRA: {f}")


    def do_open_last():
        p = getattr(tools_pane, "_qwen2512_last_out", None)
        if p:
            try:
                os.startfile(str(p))  # type: ignore[attr-defined]
            except Exception:
                _append("[ERR] Could not open last image.")
        else:
            _append("[info] No last image yet.")

    def do_run():
        _snapshot_and_save()
        prm = prompt.toPlainText().strip()
        if not prm:
            _append("[ERR] Prompt is empty.")
            return
        try:
            cmd, out_path = _build_cmd(
                root=root,
                prompt=prm,
                w=sp_w.value(),
                h=sp_h.value(),
                steps=sp_steps.value(),
                cfg=float(sp_cfg.value()),
                sampler=cmb_sampler.currentText(),
                flow_shift=sp_flow.value(),
                seed=ed_seed.text(),
                offload_cpu=cb_offload.isChecked(),
                lora_enabled=cb_lora.isChecked(),
                lora_path=ed_lora.text(),
                lora_scale=float(sp_lora.value()),
            )
        except Exception as e:

            _append(f"[ERR] {e}")
            _append("[hint] Make sure models are downloaded into models/Qwen-Image-2512 GGUF (Q4 diffusion, Q4 LLM, and qwen_image_vae.safetensors).")
            return

        tools_pane._qwen2512_last_out = out_path

        # Log which sd-cli we are using (helps debug mismatched builds).
        try:
            exe = cmd[0]
            _append(f"[info] sd-cli: {exe}")
            want = root / ".qwen2512" / "bin" / "sd-cli.exe"
            try:
                if str(exe).lower().endswith("sd-cli.exe") and Path(exe).exists():
                    if want.exists() and Path(exe).resolve() != want.resolve():
                        _append("[warn] You are NOT using qwen2512/bin/sd-cli.exe. This is fine for quick tests, but can break later if that folder disappears.")
                else:
                    if not want.exists():
                        _append("[warn] .qwen2512/bin/sd-cli.exe not found yet. Run Install / Update… to download a self-contained sd-cli for this tool.")
            except Exception:
                pass
        except Exception:
            pass

        _append("[run] " + " ".join(shlex.quote(c) for c in cmd))
        # If sd-cli is not found, QProcess can fail silently unless we log it.
        if cmd and (cmd[0] == "sd-cli" or cmd[0].endswith("sd-cli.exe")):
            try:
                if os.path.sep not in cmd[0] and not shutil.which(cmd[0]):
                    _append("[ERR] sd-cli not found. Install via the tool's installer, or put sd-cli.exe in /.qwen2512/bin, /bin, /presets/bin, the app root, or add it to PATH.")
            except Exception:
                pass
        tools_pane._qwen2512_proc = _run_process(cmd, root)

    btn_install.clicked.connect(do_install)
    btn_models.clicked.connect(do_open_models)
    btn_out.clicked.connect(do_open_out)
    btn_pick.clicked.connect(do_pick_models_folder)
    btn_lora.clicked.connect(do_pick_lora)
    btn_open_last.clicked.connect(do_open_last)
    btn_run.clicked.connect(do_run)

    section.setContentLayout(lay)
    try:
        section.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    except Exception:
        pass

# -----------------------------
# Standalone test runner
# -----------------------------
# You can run this file directly to test the Qwen2512 UI without the full FrameVision app:
#   python helpers/qwen2512.py
#
# Notes:
# - This only launches the UI. "Run" will still require sd-cli + the model files in models/Qwen-Image-2512 GGUF.
# - The tool uses the same on-disk settings file: presets/setsave/qwen2512.json (relative to app root).
#
if __name__ == "__main__":
    try:
        import sys
        from PySide6.QtWidgets import QApplication, QMainWindow, QScrollArea
    except Exception as e:
        raise SystemExit("PySide6 is required to run this standalone UI. Error: %r" % (e,))

    class _DummySection(QWidget):
        '''
        Minimal stand-in for FrameVision's CollapsibleSection.

        install_qwen2512_tool() expects:
        - section.setContentLayout(QLayout)
        - section.content (QWidget) (optional; used for size policy)
        '''
        def __init__(self, parent=None):
            super().__init__(parent)
            outer = QVBoxLayout(self)
            outer.setContentsMargins(8, 8, 8, 8)
            outer.setSpacing(8)
            self.content = QWidget(self)
            outer.addWidget(self.content, 1)

        def setContentLayout(self, layout):
            # Replace any existing layout cleanly
            try:
                old = self.content.layout()
                if old is not None:
                    QWidget().setLayout(old)  # detach
            except Exception:
                pass
            self.content.setLayout(layout)

    class _DummyToolsPane(QWidget):
        '''QWidget subclass is enough for QProcess parenting + attribute storage.'''
        pass

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Qwen-Image-2512 (GGUF) — Standalone Test")

    tools_pane = _DummyToolsPane()
    section = _DummySection()

    # Build the UI into the dummy section
    install_qwen2512_tool(tools_pane=tools_pane, section=section)

    # Optional scroll area (handy on small screens)
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setWidget(section)
    win.setCentralWidget(sc)

    win.resize(1100, 780)
    win.show()
    sys.exit(app.exec())
