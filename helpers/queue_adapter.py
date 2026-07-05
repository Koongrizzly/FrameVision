import os
import re
import time as _time
from pathlib import Path
def default_outdir(is_video: bool=False, purpose: str='upscale') -> str:
    base = _base_root()
    if purpose == 'rife':
        d = base / 'output' / 'video' / 'interpolated'
    elif purpose == 'wan22':
        d = base / 'output' / 'video' / 'wan22'
    elif purpose == 'chroma':
        d = base / 'output' / 'images' / 'chroma'
    else:
        d = base / 'output' / ('video' if is_video else 'photo') / 'upscaled'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
def _base_root():
    """Return a stable project root.
    We try to anchor relative paths to the repository/app root
    by inferring it from this file location when possible.
    """
    try:
        here = Path(__file__).resolve()
        parent = here.parent
        if parent.name.lower() == "helpers":
            return parent.parent
        return parent
    except Exception:
        return Path('.').resolve()
def _read_last_input_path():
    try:
        import json, os
        cfg = os.path.join(os.path.expanduser("~"), ".framelab", "upscaler_config.json")
        with open(cfg,"r",encoding="utf-8") as f:
            data = json.load(f)
        p = data.get("last_input_path")
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    return ""
def jobs_dirs():
    base = _base_root()
    d = {k: base/'jobs'/k for k in ('pending','running','done','failed')}
    for p in d.values(): p.mkdir(parents=True, exist_ok=True)
    return d
def _safe_int(x, default=4):
    try: return int(x)
    except Exception:
        try: return int(float(x))
        except Exception: return default
def _read_field(w, names, getter='text'):
    for n in names:
        obj = getattr(w, n, None)
        if obj is None: continue
        try:
            if getter=='text': return str(obj.text()).strip()
            if getter=='value': return int(obj.value())
            if getter=='currentText': return str(obj.currentText()).strip()
        except Exception: pass
    return ''
def infer_upscale_args(inner):
    # Input / output
    inp = _read_field(inner, ['edit_input','edit_input_path','line_input','input_path','edit_source'])
    outdir = _read_field(inner, ['edit_outdir','edit_output','line_output','output_dir'])
    # Scale
    scale = _read_field(inner, ['spin_scale','spin_factor'], getter='value')
    if not scale:
        scale = _safe_int(_read_field(inner, ['edit_scale','line_scale'], getter='text'), 4)
    # Model: pick the *active* model dropdown based on the selected engine.
    # (Many UIs keep other model combos instantiated; reading the wrong one yields "first item always".)
    engine = _read_field(inner, ['combo_engine','engine_combo','combo_upscale_engine'], getter='currentText').lower()
    model = ""
    # Best-case: Upscaler pane exposes a helper that returns the currently selected model text.
    try:
        fn = getattr(inner, "_fv_current_model_text", None)
        if callable(fn):
            model = str(fn() or "").strip()
    except Exception:
        model = ""
    # Engine-specific combo boxes (FrameVision Upscaler)
    if not model:
        if "waifu" in engine or "w2x" in engine:
            model = _read_field(inner, ['combo_model_w2x','combo_model_waifu','combo_model_waifu2x'], getter='currentText')
        elif "ultrasharp" in engine:
            model = _read_field(inner, ['combo_model_ultrasharp'], getter='currentText')
        elif "srmd" in engine:
            # Some builds use SRMD models via Real-ESRGAN naming; prefer that combo if present.
            model = _read_field(inner, ['combo_model_srmd_realsr','combo_model_srmd_realesr','combo_model_srmd_realesrgan'], getter='currentText')
            if not model:
                model = _read_field(inner, ['combo_model_srmd'], getter='currentText')
        elif "ncnn" in engine and "realsr" in engine:
            model = _read_field(inner, ['combo_model_realsr_ncnn','combo_model_realsrgan_ncnn'], getter='currentText')
        elif "realsr" in engine or "realesrgan" in engine or "real-esrgan" in engine or "esrgan" in engine:
            model = _read_field(inner, ['combo_model_realsr','combo_model_realesrgan','combo_model_esrgan'], getter='currentText')
    # Fallback: try any known model combo
    if not model:
        model = _read_field(inner, [
            'combo_model_realsr',
            'combo_model_realsr_ncnn',
            'combo_model_ultrasharp',
            'combo_model_srmd_realsr',
            'combo_model_srmd',
            'combo_model_w2x',
            'combo_model',
        ], getter='currentText')
    # Last resort: free-text fields
    if not model:
        model = _read_field(inner, ['edit_model','line_model'], getter='text')
    if not model:
        model = 'RealESRGAN-general-x4v3'
    inp = os.path.abspath(inp) if inp else ''
    outdir = os.path.abspath(outdir) if outdir else ''
    return inp, outdir, _safe_int(scale, 4), model
def enqueue(job_type, input_path=None, out_dir=None, factor=None, model=None, fmt='png'):
    """Create a queue job JSON.
    Backward-compatible:
      - enqueue(job_type, input_path, out_dir, factor, model, fmt='png')
      - enqueue(job_dict)  (external/seedvr2 jobs)
    """
    # Allow callers to pass a full job dict (e.g. SeedVR2 integration from upsc.py).
    if isinstance(job_type, dict) and input_path is None:
        job = job_type
        try:
            # Detect SeedVR2 jobs (prefer explicit engine)
            eng = str(job.get('engine') or '').lower().strip()
            cmd = job.get('cmd') or job.get('command') or job.get('ffmpeg_cmd')
            cmd_txt = ''
            try:
                if isinstance(cmd, (list, tuple)):
                    cmd_txt = ' '.join(map(str, cmd)).lower()
                elif isinstance(cmd, str):
                    cmd_txt = cmd.lower()
            except Exception:
                cmd_txt = ''
            if eng == 'seedvr2' or ('seedvr2' in cmd_txt) or ('inference_cli.py' in cmd_txt and 'seedvr2' in cmd_txt):
                return enqueue_seedvr2(job)
        except Exception:
            pass
        # Default: treat as external command job
        return enqueue_external(job)
    from helpers.job_helper import make_job_json
    d = jobs_dirs()
    # Keep factor consistent with model suffix when using fixed-scale Real-ESRGAN style models.
    try:
        _ms = re.search(r"(?i)-x(\d+)\s*$", str(model or ""))
        if _ms:
            factor = int(_ms.group(1))
    except Exception:
        pass
    args = {'factor': int(factor), 'model': model}
    try:
        _ms = re.search(r"(?i)-x(\d+)\s*$", str(model or ""))
        if _ms:
            args['model_scale'] = int(_ms.group(1))
    except Exception:
        pass
    if job_type == 'upscale_photo':
        args['format'] = fmt
    return make_job_json(job_type, input_path, out_dir, args, str(d['pending']), priority=500)
def enqueue_from_widget(inner, is_video: bool):
    inp, outdir, scale, model = infer_upscale_args(inner)
    if (not inp) or (not os.path.exists(inp)):
        inp = _read_last_input_path() or inp
    if not inp or not os.path.exists(inp): raise RuntimeError('No valid input selected.')
    if not outdir:
        outdir = default_outdir(is_video)
    jt = 'upscale_video' if is_video else 'upscale_photo'
    return enqueue(jt, inp, outdir, scale, model, fmt='png')
def _fv_fix_hunyuan15_conda_python_cmd(cmd, root_hint=None):
    """Pick a real Hunyuan Python for queued commands.
    This does not trust Scripts/python.exe. It scans known Hunyuan env folders
    and prefers the first interpreter that can import diffusers.
    """
    try:
        from pathlib import Path as _P
        import subprocess as _subprocess
        if not isinstance(cmd, (list, tuple)) or not cmd:
            return cmd, []
        fixed = list(cmd)
        joined = ' '.join(str(x) for x in fixed[:12]).replace('\\', '/').lower()
        if 'hunyuan15_cli.py' not in joined and 'hunyuan15' not in joined:
            return cmd, []
        root = _P(str(root_hint or _P.cwd())).resolve()
        env_dirs = [
            root / 'environments' / '.hunyuan15_official',
            root / 'environments' / '.hunyuan15',
            root / '.hunyuan15_env',
        ]
        cands = []
        try:
            _override = os.environ.get('FV_HUNYUAN15_PYTHON', '').strip()
            if _override:
                cands.append(_P(_override))
        except Exception:
            pass
        for env in env_dirs:
            cands.extend([env / 'python.exe', env / 'Scripts' / 'python.exe', env / 'bin' / 'python'])
        checked = []
        selected = None
        for py in cands:
            try:
                if not (py.exists() and py.is_file()):
                    continue
                ok = False
                try:
                    pr = _subprocess.run([str(py), '-c', 'import diffusers'], cwd=str(root), stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL, timeout=12, check=False)
                    ok = int(getattr(pr, 'returncode', 1)) == 0
                except Exception:
                    ok = False
                checked.append(f"{py} => {'diffusers OK' if ok else 'no diffusers'}")
                if ok:
                    selected = py
                    break
            except Exception:
                pass
        if selected is None:
            for py in cands:
                try:
                    norm = str(py).replace('\\', '/').lower()
                    if py.exists() and py.is_file() and norm.endswith('/python.exe') and '/scripts/' not in norm:
                        selected = py
                        checked.append(f"{py} => fallback root python")
                        break
                except Exception:
                    pass
        if selected is not None and str(selected) != str(fixed[0]):
            fixed[0] = str(selected)
            return fixed, checked
        return cmd, checked
    except Exception as e:
        return cmd, [f'queue adapter rewrite failed: {e}']
def _fv_fix_hunyuan15_args(args):
    try:
        if not isinstance(args, dict):
            return args
        engine = str(args.get('engine') or '').lower().strip()
        cmd = args.get('ffmpeg_cmd') or args.get('cmd')
        cmd_txt = ''
        try:
            cmd_txt = ' '.join(str(x) for x in cmd) if isinstance(cmd, (list, tuple)) else str(cmd or '')
        except Exception:
            cmd_txt = ''
        if engine == 'hunyuan15' or 'hunyuan15_cli.py' in cmd_txt:
            fixed, checked = _fv_fix_hunyuan15_conda_python_cmd(cmd, root_hint=args.get('cwd'))
            args = dict(args)
            args['hunyuan15_python_rewrite_checked'] = checked[:10]
            if fixed is not cmd:
                if args.get('ffmpeg_cmd') is not None:
                    args['ffmpeg_cmd'] = fixed
                elif args.get('cmd') is not None:
                    args['cmd'] = fixed
                args['python_path_fixed'] = 'hunyuan15_verified_python'
                args['hunyuan15_python_rewrite'] = str(fixed[0]) if fixed else ''
    except Exception as e:
        try:
            args = dict(args)
            args['hunyuan15_python_rewrite_error'] = str(e)
        except Exception:
            pass
    return args
def enqueue_tool_job(job_type: str, input_path: str, out_dir: str, args: dict, priority: int=600):
    from helpers.job_helper import make_job_json
    d = jobs_dirs()
    args = _fv_fix_hunyuan15_args(args or {})
    return make_job_json(job_type, input_path, out_dir, args or {}, str(d['pending']), priority=int(priority))

def enqueue_musicclip_tool_job(input_path: str, out_dir: str, args: dict, priority: int = 560):
    """Queue a Music Clip Creator helper command with metadata for footer mirroring.

    It still uses the proven tools_ffmpeg worker path, but marks the job so
    Music Clip Creator can find the matching queue JSON/log reliably and keep
    showing progress in its sticky footer.
    """
    data = dict(args or {})
    data.setdefault("queue_family", "musicclip")
    data.setdefault("engine", "musicclip")
    data.setdefault("progress_owner", "Music Clip Creator")
    return enqueue_tool_job("tools_ffmpeg", input_path, out_dir, data, priority=priority)

def enqueue_musicclip_ltx_tool_job(input_path: str, out_dir: str, args: dict, priority: int = 570):
    """Queue a Music Clip Creator LTX full-run helper with metadata for UI progress."""
    data = dict(args or {})
    data.setdefault("queue_family", "musicclip_ltx")
    data.setdefault("engine", "musicclip_ltx")
    data.setdefault("progress_owner", "Music Clip Creator LTX")
    return enqueue_tool_job("tools_ffmpeg", input_path, out_dir, data, priority=priority)
def _fv_first_existing_python(env_dir: Path) -> Path | None:
    """Return the first Python executable inside a portable FrameVision env."""
    try:
        candidates = [
            env_dir / "python.exe",              # Windows conda prefix
            env_dir / "Scripts" / "python.exe",  # Windows venv
            env_dir / "bin" / "python",          # Linux/macOS conda or venv
            env_dir / "python",
        ]
        for py in candidates:
            try:
                if py.exists() and py.is_file():
                    return py
            except Exception:
                pass
    except Exception:
        pass
    return None
def enqueue_chroma_generate(settings: dict, priority: int = 610):
    """Queue one SPARK.Chroma image generation job.
    Chroma must run inside FrameVision's shared image-model environment,
    not the main app environment.  The queued job is executed by worker.py via
    the generic tools_ffmpeg runner so it gets normal queue progress/log/cancel
    behavior while still calling helpers/chroma.py --generate.
    """
    root = _base_root()
    out_dir = root / "output" / "images" / "chroma"
    out_dir.mkdir(parents=True, exist_ok=True)
    env_dir = root / "environments" / ".images_models"
    py = _fv_first_existing_python(env_dir)
    if py is None:
        # Keep the expected portable path in the job for clearer queue errors.
        py = env_dir / ("python.exe" if os.name == "nt" else "bin/python")
    helper = root / "helpers" / "chroma.py"
    data = dict(settings or {})
    def _text(key: str, default: str = "") -> str:
        try:
            return str(data.get(key, default) or "")
        except Exception:
            return default
    def _int(key: str, default: int) -> int:
        try:
            return int(data.get(key, default))
        except Exception:
            try:
                return int(float(data.get(key, default)))
            except Exception:
                return int(default)
    def _float(key: str, default: float) -> float:
        try:
            return float(data.get(key, default))
        except Exception:
            return float(default)
    prompt = _text("prompt").strip()
    negative = _text("negative")
    width = _int("width", 1024)
    height = _int("height", 1024)
    steps = _int("steps", 35)
    guidance = _float("guidance", 3.0)
    seed = _int("seed", -1)
    max_sequence_length = _int("max_sequence_length", 512)
    offload_cpu = bool(data.get("offload_cpu", True))
    cmd = [
        str(py), str(helper), "--generate",
        "--prompt", prompt,
        "--negative", negative,
        "--width", str(width),
        "--height", str(height),
        "--steps", str(steps),
        "--guidance", str(guidance),
        "--seed", str(seed),
        "--max-sequence-length", str(max_sequence_length),
        "--offload-cpu" if offload_cpu else "--no-offload-cpu",
    ]
    label_prompt = prompt.replace("\n", " ").strip()
    if len(label_prompt) > 70:
        label_prompt = label_prompt[:67].rstrip() + "..."
    label = "Chroma image" + (f" — {label_prompt}" if label_prompt else "")
    args = {
        "label": label,
        "engine": "chroma",
        "ffmpeg_cmd": cmd,
        "cwd": str(root),
        "scan_dir": str(out_dir),
        "scan_ext": ".png",
        "prompt": prompt,
        "negative": negative,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance,
        "seed": seed,
        "max_sequence_length": max_sequence_length,
        "offload_cpu": offload_cpu,
        "env_python": str(py),
        "cli_py": str(helper),
        "assistant_origin": data.get("assistant_origin", ""),
        "assistant_chat_only": bool(data.get("assistant_chat_only", False)),
    }
    return enqueue_tool_job("chroma_generate", "", str(out_dir), args, priority=int(priority))
def default_ideogram4_outdir():
    base = _base_root()
    d = base / 'output' / 'image' / 'ideogram4'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
def enqueue_ideogram4_generate(settings: dict, priority: int = 610):
    """Queue one Ideogram 4 GGUF image generation job for the FrameVision worker."""
    root = _base_root()
    data = dict(settings or {})
    out_dir = Path(str(data.get('output_dir') or default_ideogram4_outdir())).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    def _text(key: str, default: str = '') -> str:
        try:
            return str(data.get(key, default) or '')
        except Exception:
            return default
    def _int(key: str, default: int) -> int:
        try:
            return int(data.get(key, default))
        except Exception:
            try:
                return int(float(data.get(key, default)))
            except Exception:
                return int(default)
    def _float(key: str, default: float) -> float:
        try:
            return float(data.get(key, default))
        except Exception:
            return float(default)
    prompt = _text('prompt').strip()
    if not prompt:
        raise RuntimeError('Prompt is empty.')
    negative = _text('negative')
    width = _int('width', 1024)
    height = _int('height', 1024)
    steps = _int('steps', 20)
    guidance = _float('guidance', 3.5)
    seed = _int('seed', -1)
    raw_prompt = bool(data.get('raw_prompt', False))
    gguf_stream_layers = bool(data.get('gguf_stream_layers', False))
    gguf_dir = _text('gguf_dir')
    gguf_diffusion_file = _text('gguf_diffusion_file')
    gguf_unconditional_file = _text('gguf_unconditional_file')
    gguf_llm_file = _text('gguf_llm_file')
    gguf_vae_file = _text('gguf_vae_file')
    sd_cli_path = _text('sd_cli_path')
    stamp = _time.strftime('%Y%m%d_%H%M%S') + f"_{int((_time.time() % 1.0) * 1000):03d}"
    out_file = str(out_dir / f'ideogram4_gguf_{stamp}.png')
    preview = prompt.replace('\n', ' ').strip()[:80] or 'Ideogram 4 GGUF'
    args = {
        'label': 'Ideogram 4 GGUF: ' + preview,
        'engine': 'ideogram4_gguf',
        'prompt': prompt,
        'negative': negative,
        'width': width,
        'height': height,
        'steps': steps,
        'guidance': guidance,
        'seed': seed,
        'raw_prompt': raw_prompt,
        'gguf_stream_layers': gguf_stream_layers,
        'gguf_dir': gguf_dir,
        'gguf_diffusion_file': gguf_diffusion_file,
        'gguf_unconditional_file': gguf_unconditional_file,
        'gguf_llm_file': gguf_llm_file,
        'gguf_vae_file': gguf_vae_file,
        'sd_cli_path': sd_cli_path,
        'output_dir': str(out_dir),
        'out_file': out_file,
        'outfile': out_file,
        'assistant_origin': str(getattr(inner, 'assistant_origin', '') or getattr(getattr(inner, 'cfg', None), 'assistant_origin', '') or ''),
        'assistant_chat_only': bool(getattr(inner, 'assistant_chat_only', False) or getattr(getattr(inner, 'cfg', None), 'assistant_chat_only', False)),
    }
    return enqueue_tool_job('ideogram4_generate', '', str(out_dir), args, priority=int(priority))

def default_krea2_outdir():
    base = _base_root()
    d = base / 'output' / 'images' / 'krea2'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def enqueue_krea2_generate(settings: dict, priority: int = 610):
    """Queue one Krea 2 GGUF image generation job for the FrameVision worker."""
    root = _base_root()
    data = dict(settings or {})

    def _text(key: str, default: str = '') -> str:
        try:
            return str(data.get(key, default) or '')
        except Exception:
            return default

    def _int(key: str, default: int) -> int:
        try:
            return int(data.get(key, default))
        except Exception:
            try:
                return int(float(data.get(key, default)))
            except Exception:
                return int(default)

    prompt = _text('prompt').strip()
    if not prompt:
        raise RuntimeError('Prompt is empty.')

    out_dir = Path(_text('output_dir') or default_krea2_outdir())
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = data.get('cmd') or data.get('ffmpeg_cmd')
    out_file = _text('out_file') or _text('outfile') or _text('output_path')
    if not out_file:
        stamp = _time.strftime('%Y%m%d_%H%M%S') + f"_{int((_time.time() % 1.0) * 1000):03d}"
        out_file = str(out_dir / f'krea2_gguf_{stamp}.png')

    if not cmd:
        sdcli = _text('sdcli') or str(root / 'presets' / 'bin' / ('sd-cli.exe' if os.name == 'nt' else 'sd-cli'))
        model = _text('model')
        llm = _text('llm')
        vae = _text('vae')
        def _abs(p: str) -> str:
            pp = Path(str(p).strip().strip('"'))
            if not pp.is_absolute():
                pp = root / pp
            return str(pp)
        cmd = [
            _abs(sdcli),
            '--diffusion-model', _abs(model),
            '--llm', _abs(llm),
            '--vae', _abs(vae),
            '-p', prompt,
            '--steps', str(_int('steps', 8)),
            '--cfg-scale', str(data.get('cfg', 1.0)),
            '--guidance', str(data.get('guidance', 3.5)),
            '--width', str(_int('width', 1024)),
            '--height', str(_int('height', 1024)),
            '--seed', str(_int('seed', -1)),
            '--batch-count', str(_int('batch', 1)),
            '--output', str(out_file),
        ]
        negative = _text('negative')
        if negative:
            cmd += ['--negative-prompt', negative]
        try:
            flow = float(data.get('flow_shift', 1.15))
            if flow >= 0:
                cmd += ['--flow-shift', str(flow)]
        except Exception:
            pass
        init_img = _text('init_img')
        if init_img:
            cmd += ['--init-img', _abs(init_img), '--strength', str(data.get('strength', 0.75))]
        if bool(data.get('diffusion_fa', True)):
            cmd.append('--diffusion-fa')
        if bool(data.get('offload', False)):
            cmd.append('--offload-to-cpu')
        if bool(data.get('vae_tiling', False)):
            cmd.append('--vae-tiling')
        if bool(data.get('disable_metadata', False)):
            cmd.append('--disable-image-metadata')
        if bool(data.get('verbose', True)):
            cmd.append('-v')
        backend = _text('backend')
        if backend:
            cmd += ['--backend', backend]
        params_backend = _text('params_backend')
        if params_backend:
            cmd += ['--params-backend', params_backend]
        sampler = _text('sampler')
        if sampler and sampler != 'auto':
            cmd += ['--sampling-method', sampler]
        scheduler = _text('scheduler')
        if scheduler and scheduler != 'auto':
            cmd += ['--scheduler', scheduler]
        extra = _text('extra_args')
        if extra:
            try:
                import shlex as _shlex
                cmd += _shlex.split(extra)
            except Exception:
                cmd += extra.split()

    preview = prompt.replace('\n', ' ').strip()[:80] or 'Krea 2 GGUF'
    label = _text('label') or ('Krea 2 GGUF: ' + preview)
    args = {
        'label': label,
        'engine': 'krea2_gguf',
        'ffmpeg_cmd': list(cmd) if isinstance(cmd, (list, tuple)) else cmd,
        'cmd': list(cmd) if isinstance(cmd, (list, tuple)) else cmd,
        'cwd': _text('cwd') or str(root),
        'outfile': str(out_file),
        'out_file': str(out_file),
        'output_path': str(out_file),
        'scan_dir': _text('scan_dir') or str(out_dir),
        'scan_ext': _text('scan_ext') or '.png',
        'prompt': prompt,
        'negative': _text('negative'),
        'width': _int('width', 1024),
        'height': _int('height', 1024),
        'steps': _int('steps', 8),
        'seed': _int('seed', -1),
        'batch': _int('batch', 1),
        'model': _text('model'),
        'llm': _text('llm'),
        'vae': _text('vae'),
    }
    return enqueue_tool_job('krea2_generate', '', str(out_dir), args, priority=int(priority))

def default_boogu_outdir(mode: str = 'normal'):
    base = _base_root()
    if str(mode or '').lower() == 'edit':
        d = base / 'output' / 'edits' / 'boogu'
    else:
        d = base / 'output' / 'images' / 'boogu'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def enqueue_boogu_generate(settings: dict, priority: int = 610):
    """Queue one Boogu Image sd-cli job for the FrameVision worker."""
    root = _base_root()
    data = dict(settings or {})

    def _text(key: str, default: str = '') -> str:
        try:
            return str(data.get(key, default) or '')
        except Exception:
            return default

    def _int(key: str, default: int) -> int:
        try:
            return int(data.get(key, default))
        except Exception:
            try:
                return int(float(data.get(key, default)))
            except Exception:
                return int(default)

    mode = _text('mode', 'normal').strip().lower() or 'normal'
    prompt = _text('prompt').strip()
    out_dir = Path(_text('output_dir') or default_boogu_outdir(mode))
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = data.get('cmd') or data.get('ffmpeg_cmd')
    if not cmd:
        raise RuntimeError('Boogu queue job is missing cmd/ffmpeg_cmd.')

    preview = prompt.replace('\n', ' ').strip()[:80] or ('Boogu Image ' + mode)
    label = _text('label') or ('Boogu Image: ' + preview)
    if label.lower() in ('boogu image create', 'boogu image edit') and preview:
        label = label + ' — ' + preview

    args = {
        'label': label,
        'engine': 'boogu_image',
        'mode': mode,
        'ffmpeg_cmd': list(cmd) if isinstance(cmd, (list, tuple)) else cmd,
        'cmd': list(cmd) if isinstance(cmd, (list, tuple)) else cmd,
        'cwd': _text('cwd') or str(root),
        'scan_dir': _text('scan_dir') or str(out_dir),
        'scan_ext': _text('scan_ext') or '.png',
        'prompt': prompt,
        'width': _int('width', 1024),
        'height': _int('height', 1024),
        'steps': _int('steps', 4 if mode == 'normal' else 20),
        'seed': _int('seed', -1),
    }
    return enqueue_tool_job('boogu_generate', '', str(out_dir), args, priority=int(priority))


def enqueue_ace_step15(cfg_path: str, out_dir: str, env_python: str, cli_py: str, project_root: str, label: str="Ace-Step 1.5", hide_console: bool=True, priority: int=620):
    """Convenience wrapper to enqueue an Ace-Step 1.5 job.
    The Ace-Step 1.5 UI writes a TOML config first, then enqueues the job so it is
    reproducible when the worker picks it up.
    """
    args = {
        "label": label,
        "env_python": env_python,
        "cli_py": cli_py,
        "project_root": project_root,
        "cfg_path": cfg_path,
        "hide_console": bool(hide_console),
    }
    return enqueue_tool_job("ace_step_15", "", out_dir, args, priority=int(priority))
def enqueue_rife_from_widget(inner):
    """Read fields from RifeWidget and enqueue a rife_interpolate job."""
    # Pull values with best-effort getattr
    def _read(name, default=""):
        try:
            obj = getattr(inner, name)
            if callable(obj): return obj()
        except Exception: pass
        # fallbacks to typical attribute names
        for cand in ('edit_input','input_path','line_input'):
            v = getattr(inner, cand, None)
            if v and hasattr(v, 'text'): 
                try: return str(v.text()).strip()
                except Exception: pass
        return default
    inp = _read('input_path', '')
    outdir = _read('output_dir', '')
    # defaults
    if not inp or not os.path.exists(inp): raise RuntimeError('No valid input selected.')
    if not outdir: outdir = default_outdir(True, 'rife')
    # numeric options
    def _iv(name, default=0):
        try: return int(getattr(inner, name)())
        except Exception: return int(default)
    factor = max(2, _iv('rife_factor', 2))
    target_fps = max(0, _iv('rife_target_fps', 0))
    threads = max(0, _iv('threads', 0))
    gpu = max(0, _iv('gpu_id', 0))
    streaming = bool(getattr(inner, 'streaming', lambda: False)())
    chunk = max(2, _iv('chunk_seconds', 8)) if streaming else 0
    fmt = str(getattr(inner, 'out_format', lambda: 'mp4')())
    args = {
        'factor': int(factor),
        'target_fps': int(target_fps),
        'threads': int(threads),
        'gpu': int(gpu),
        'streaming': bool(streaming),
        'chunk_seconds': int(chunk),
        'format': fmt
    }
    return enqueue_tool_job('rife_interpolate', inp, outdir, args, priority=550)
# >>> FRAMEVISION_QWEN_BEGIN
def enqueue_txt2img_qwen(job_args: dict):
    """Insert-only wrapper to enqueue a Qwen txt2img job."""
    try:
        from helpers.job_helper import make_job_json
        from helpers.queue_adapter import jobs_dirs
        d = jobs_dirs()
        out_dir = job_args.get('output') or default_txt2img_outdir()
        args = {k:v for k,v in job_args.items() if k not in ('output','run_now')}
        return make_job_json('txt2img_qwen', '', out_dir, args, str(d['pending']), priority=500)
    except Exception as e:
        print('[queue] enqueue_txt2img_qwen failed:', e)
        return False
def default_qwen2511_outdir():
    from pathlib import Path
    base = Path('.').resolve()
    d = base/'output'/'edits'/'qwen_2511'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
def enqueue_flux_klein_from_widget(inner):
    """Enqueue a Flux Klein GGUF image edit job from the Flux Klein editor UI."""
    import time as _time
    from pathlib import Path as _P
    try:
        from helpers.queue_adapter import jobs_dirs
    except Exception:
        from queue_adapter import jobs_dirs
    try:
        from helpers.job_helper import make_job_json
    except Exception:
        from job_helper import make_job_json
    d = jobs_dirs()
    def _txt(name, default=''):
        try:
            obj = getattr(inner, name, None)
            if obj is None:
                return default
            if hasattr(obj, 'text'):
                return str(obj.text()).strip()
            if hasattr(obj, 'toPlainText'):
                return str(obj.toPlainText()).strip()
        except Exception:
            pass
        return default
    def _val(name, default=None):
        try:
            obj = getattr(inner, name, None)
            if obj is None:
                return default
            if hasattr(obj, 'value'):
                return obj.value()
        except Exception:
            pass
        return default
    def _checked(name, default=False):
        try:
            obj = getattr(inner, name, None)
            if obj is None:
                return default
            if hasattr(obj, 'isChecked'):
                return bool(obj.isChecked())
        except Exception:
            pass
        return default
    def _combo_data(name):
        try:
            obj = getattr(inner, name, None)
            if obj is None:
                return ''
            if hasattr(obj, 'currentData'):
                return str(obj.currentData() or '').strip()
        except Exception:
            pass
        return ''
    def _combo_text(name, default=''):
        try:
            obj = getattr(inner, name, None)
            if obj is None:
                return default
            if hasattr(obj, 'currentText'):
                return str(obj.currentText() or '').strip()
        except Exception:
            pass
        return default
    sdcli_path = _txt('sdcli_edit')
    model_dir = _txt('modeldir_edit')
    diffusion_model = _combo_data('flux_combo')
    llm_model = _combo_data('llm_combo')
    vae_file = _combo_data('vae_combo')
    lora_file = _combo_data('lora_combo')
    prompt = _txt('prompt_edit')
    negative = _txt('neg_edit')
    width = int(_val('width_spin', 1024) or 1024)
    height = int(_val('height_spin', 1024) or 1024)
    steps = int(_val('steps_spin', 4) or 4)
    cfg_scale = float(_val('cfg_spin', 1.0) or 1.0)
    seed = int(_val('seed_spin', 0) or 0)
    random_seed = _checked('chk_rand_seed', True)
    sampling_method = _combo_text('sampling_combo', 'euler') or 'euler'
    diffusion_fa = _checked('chk_diffusion_fa', True)
    offload_to_cpu = _checked('chk_offload_cpu', False)
    vae_tiling = _checked('chk_vae_tiling', False)
    out_name = _txt('out_name_edit')
    lora_strength = float(_val('lora_strength_spin', 1.0) or 1.0)
    ref_images = []
    try:
        cfg = getattr(inner, 'cfg', None)
        ref_images = list(getattr(cfg, 'ref_images', []) or [])
    except Exception:
        ref_images = []
    ref_images = [str(x).strip() for x in ref_images if str(x).strip()]
    if not prompt:
        raise RuntimeError('Prompt is empty.')
    if not sdcli_path or not _P(sdcli_path).is_file():
        raise RuntimeError('sd-cli not found: ' + str(sdcli_path))
    if not diffusion_model or not _P(diffusion_model).is_file():
        raise RuntimeError('Flux GGUF not found: ' + str(diffusion_model))
    if not llm_model or not _P(llm_model).is_file():
        raise RuntimeError('Qwen GGUF not found: ' + str(llm_model))
    if not vae_file or not _P(vae_file).is_file():
        raise RuntimeError('VAE not found: ' + str(vae_file))
    if lora_file and not _P(lora_file).is_file():
        raise RuntimeError('LoRA not found: ' + str(lora_file))
    for rp in ref_images:
        if not _P(rp).is_file():
            raise RuntimeError('Reference image not found: ' + rp)
    try:
        out_dir = getattr(getattr(inner, 'paths', None), 'out_dir', '') or ''
    except Exception:
        out_dir = ''
    if not out_dir:
        try:
            root = getattr(getattr(inner, 'paths', None), 'root', '') or ''
            out_dir = str(_P(root) / 'output' / 'edits' / 'flux_klein') if root else str(_P('.') / 'output' / 'edits' / 'flux_klein')
        except Exception:
            out_dir = str(_P('.') / 'output' / 'edits' / 'flux_klein')
    _P(out_dir).mkdir(parents=True, exist_ok=True)
    if out_name:
        out_file = str(_P(out_dir) / out_name)
    else:
        out_file = str(_P(out_dir) / f"klein_{_time.strftime('%Y%m%d_%H%M%S')}.png")
    label = 'Flux Klein image edit'
    if prompt:
        label = 'Flux Klein: ' + prompt.replace('\n', ' ').strip()[:80]
    args = {
        'label': label,
        'sdcli_path': sdcli_path,
        'model_dir': model_dir,
        'diffusion_model': diffusion_model,
        'llm_model': llm_model,
        'vae_file': vae_file,
        'lora_file': lora_file,
        'lora_strength': lora_strength,
        'prompt': prompt,
        'negative': negative,
        'ref_images': ref_images,
        'width': width,
        'height': height,
        'steps': steps,
        'cfg_scale': cfg_scale,
        'seed': seed,
        'random_seed': random_seed,
        'sampling_method': sampling_method,
        'diffusion_fa': diffusion_fa,
        'offload_to_cpu': offload_to_cpu,
        'vae_tiling': vae_tiling,
        'out_file': out_file,
        'outfile': out_file,
    }
    input_path = ref_images[0] if ref_images else ''
    return make_job_json('flux_klein_image_edit', input_path, out_dir, args, str(d['pending']), priority=500)
def enqueue_firered_from_widget(inner):
    """Enqueue a FireRed image edit job from the FireRed editor UI."""
    import time as _time
    from pathlib import Path as _P
    try:
        from helpers.queue_adapter import jobs_dirs
    except Exception:
        from queue_adapter import jobs_dirs
    try:
        from helpers.job_helper import make_job_json
    except Exception:
        from job_helper import make_job_json
    d = jobs_dirs()
    def _txt(name, default=''):
        try:
            obj = getattr(inner, name, None)
            if obj is None:
                return default
            if hasattr(obj, 'text'):
                return str(obj.text()).strip()
            if hasattr(obj, 'toPlainText'):
                return str(obj.toPlainText()).strip()
            if hasattr(obj, 'currentText'):
                return str(obj.currentText() or '').strip()
        except Exception:
            pass
        return default
    def _val(name, default=None):
        try:
            obj = getattr(inner, name, None)
            if obj is None:
                return default
            if hasattr(obj, 'value'):
                return obj.value()
        except Exception:
            pass
        return default
    def _checked(name, default=False):
        try:
            obj = getattr(inner, name, None)
            if obj is None:
                return default
            if hasattr(obj, 'isChecked'):
                return bool(obj.isChecked())
        except Exception:
            pass
        return default
    try:
        images = list(getattr(inner, '_collect_images', lambda: [])() or [])
    except Exception:
        images = []
    images = [str(x).strip() for x in images if str(x).strip()]
    sdcli_path = _txt('sdcli_edit')
    model_path = _txt('model_edit')
    vae_path = _txt('vae_edit')
    llm_path = _txt('llm_edit')
    lora_path = _txt('lora_combo')
    prompt = _txt('prompt_edit')
    negative = _txt('negative_edit')
    width = int(_val('width_spin', 1024) or 1024)
    height = int(_val('height_spin', 1024) or 1024)
    steps = int(_val('steps_spin', 8) or 8)
    cfg_scale = int(_val('cfg_spin', 4) or 4)
    strength = float(_val('strength_spin', 0.75) or 0.75)
    seed = _txt('seed_edit', '-1') or '-1'
    sampler = _txt('sampler_combo', 'euler') or 'euler'
    batch = int(_val('batch_spin', 1) or 1)
    output_dir = _txt('output_dir_edit')
    prefix = _txt('prefix_edit', 'firered') or 'firered'
    fmt = _txt('format_combo', 'png') or 'png'
    offload_cpu = _checked('chk_offload_cpu', False)
    flash_attn = _checked('chk_flash_attn', False)
    vae_tiling = _checked('chk_vae_tiling', False)
    verbose = _checked('chk_verbose', True)
    if not prompt:
        raise RuntimeError('Prompt is empty.')
    if not images:
        raise RuntimeError('Add at least one input image. FireRed is an image edit model.')
    if not sdcli_path or not _P(sdcli_path).is_file():
        raise RuntimeError('sd-cli not found: ' + str(sdcli_path))
    if not model_path or not _P(model_path).is_file():
        raise RuntimeError('FireRed GGUF not found: ' + str(model_path))
    if not vae_path or not _P(vae_path).is_file():
        raise RuntimeError('VAE not found: ' + str(vae_path))
    if not llm_path or not _P(llm_path).is_file():
        raise RuntimeError('LLM not found: ' + str(llm_path))
    if lora_path and not _P(lora_path).is_file():
        raise RuntimeError('LoRA not found: ' + str(lora_path))
    for rp in images:
        if not _P(rp).is_file():
            raise RuntimeError('Input image not found: ' + rp)
    if not output_dir:
        output_dir = str(_P('.') / 'output' / 'edits' / 'firered')
    _P(output_dir).mkdir(parents=True, exist_ok=True)
    out_file = str(_P(output_dir) / f"{prefix}_{_time.strftime('%Y%m%d_%H%M%S')}.{fmt}")
    label = 'FireRed image edit'
    if prompt:
        label = 'FireRed: ' + prompt.replace('\n', ' ').strip()[:80]
    args = {
        'label': label,
        'sdcli_path': sdcli_path,
        'model_path': model_path,
        'vae_path': vae_path,
        'llm_path': llm_path,
        'lora_path': lora_path,
        'prompt': prompt,
        'negative': negative,
        'images': images,
        'width': width,
        'height': height,
        'steps': steps,
        'cfg_scale': cfg_scale,
        'strength': strength,
        'seed': str(seed),
        'sampler': sampler,
        'batch': batch,
        'offload_cpu': offload_cpu,
        'flash_attn': flash_attn,
        'vae_tiling': vae_tiling,
        'verbose': verbose,
        'out_file': out_file,
        'outfile': out_file,
    }
    input_path = images[0] if images else ''
    return make_job_json('firered_image_edit', input_path, output_dir, args, str(d['pending']), priority=500)
def enqueue_qwen2511_from_widget(inner):
    """Enqueue a Qwen2511 (image edit) job from the Qwen2511Pane UI.
    Uses output/edits/qwen_2511 to match the pane's current result folder.
    """
    import time as _time
    from pathlib import Path as _P
    try:
        from helpers.job_helper import make_job_json
        from helpers.queue_adapter import jobs_dirs
    except Exception:
        from job_helper import make_job_json
        from queue_adapter import jobs_dirs
    d = jobs_dirs()
    out_dir = default_qwen2511_outdir()
    # Required: input image
    init_img = str(getattr(getattr(inner, 'ed_initimg', None), 'text', lambda: '')()).strip()
    if not init_img:
        raise RuntimeError("Input image missing.")
    if not _P(init_img).is_file():
        raise RuntimeError("Input image not found: " + init_img)
    # Optional mask and refs
    mask_img = str(getattr(getattr(inner, 'ed_mask', None), 'text', lambda: '')()).strip()
    invert_mask = bool(getattr(getattr(inner, 'chk_invert_mask', None), 'isChecked', lambda: False)())
    ref_imgs = []
    for attr in ('ed_ref1', 'ed_ref2', 'ed_ref3'):
        try:
            p = str(getattr(getattr(inner, attr, None), 'text', lambda: '')()).strip()
        except Exception:
            p = ''
        if p:
            if not _P(p).is_file():
                raise RuntimeError("Reference image not found: " + p)
            ref_imgs.append(p)
    use_increase_ref_index = bool(getattr(getattr(inner, 'chk_ref_increase_index', None), 'isChecked', lambda: True)())
    disable_auto_resize_ref_images = bool(getattr(getattr(inner, 'chk_disable_ref_resize', None), 'isChecked', lambda: False)())
    # Models / paths
    sdcli_path = str(getattr(getattr(inner, 'ed_sdcli', None), 'text', lambda: '')()).strip()
    try:
        unet_path = getattr(getattr(inner, 'cb_unet', None), 'currentData', lambda: '')()
        llm_path = getattr(getattr(inner, 'cb_llm', None), 'currentData', lambda: '')()
        mmproj_path = getattr(getattr(inner, 'cb_mmproj', None), 'currentData', lambda: '')()
        vae_path = getattr(getattr(inner, 'cb_vae', None), 'currentData', lambda: '')()
    except Exception:
        unet_path = llm_path = mmproj_path = vae_path = ''
    # Prompts
    try:
        prompt = str(getattr(getattr(inner, 'ed_prompt', None), 'toPlainText', lambda: '')()).strip()
    except Exception:
        prompt = ''
    negative = str(getattr(getattr(inner, 'ed_neg', None), 'text', lambda: '')()).strip()
    # LoRA
    try:
        lora_dir = str(getattr(getattr(inner, 'ed_lora_dir', None), 'text', lambda: '')()).strip()
    except Exception:
        lora_dir = ''
    if not lora_dir:
        # Reasonable default matches Qwen2511 pane default.
        lora_dir = os.path.join(_P('.').resolve(), 'models', 'lora', 'qwen2511')
    try:
        lora_name = str(getattr(getattr(inner, 'cb_lora', None), 'currentText', lambda: '')()).strip()
    except Exception:
        lora_name = ''
    if str(lora_name).strip().lower() in ('(none)', 'none'):
        lora_name = ''
    try:
        lora_strength = float(getattr(getattr(inner, 'sp_lora_strength', None), 'value', lambda: 1.0)())
    except Exception:
        lora_strength = 1.0
    # Settings
    steps = int(getattr(getattr(inner, 'sp_steps', None), 'value', lambda: 28)())
    cfg = float(getattr(getattr(inner, 'sp_cfg', None), 'value', lambda: 4.5)())
    seed = int(getattr(getattr(inner, 'sp_seed', None), 'value', lambda: -1)())
    width = int(getattr(getattr(inner, 'sp_w', None), 'value', lambda: 1024)())
    height = int(getattr(getattr(inner, 'sp_h', None), 'value', lambda: 576)())
    # Strength (mirror direct-run behavior: read from UI and store into the queued job)
    strength = None
    try:
        strength = float(getattr(getattr(inner, 'sp_strength', None), 'value', lambda: None)())
    except Exception:
        strength = None
    if strength is None:
        try:
            strength = float(getattr(getattr(inner, 'lbl_strength', None), 'value', lambda: None)())
        except Exception:
            strength = None
    if strength is None:
        try:
            strength = float(getattr(getattr(inner, 'sl_strength', None), 'value', lambda: 100)()) / 100.0
        except Exception:
            strength = 1.0
    try:
        # NaN guard
        if strength != strength:
            strength = 1.0
    except Exception:
        strength = 1.0
    try:
        strength = max(0.0, min(1.0, float(strength)))
    except Exception:
        strength = 1.0
    sampling_method = str(getattr(getattr(inner, 'cb_sampling', None), 'currentText', lambda: 'euler_a')())
    shift = float(getattr(getattr(inner, 'sp_shift', None), 'value', lambda: 12.5)())
    # Low VRAM / perf
    use_vae_tiling = bool(getattr(getattr(inner, 'chk_vae_tiling', None), 'isChecked', lambda: False)())
    vae_tile_size = str(getattr(getattr(inner, 'cb_vae_tile_size', None), 'currentText', lambda: '256x256')()).strip()
    vae_tile_overlap = float(getattr(getattr(inner, 'sp_vae_tile_overlap', None), 'value', lambda: 0.50)())
    use_offload = bool(getattr(getattr(inner, 'chk_offload', None), 'isChecked', lambda: False)())
    use_mmap = bool(getattr(getattr(inner, 'chk_mmap', None), 'isChecked', lambda: False)())
    use_vae_on_cpu = bool(getattr(getattr(inner, 'chk_vae_on_cpu', None), 'isChecked', lambda: False)())
    use_clip_on_cpu = bool(getattr(getattr(inner, 'chk_clip_on_cpu', None), 'isChecked', lambda: False)())
    use_diffusion_fa = bool(getattr(getattr(inner, 'chk_diffusion_fa', None), 'isChecked', lambda: False)())
    # Output file (precomputed so the queue can display a deterministic target)
    ts = int(_time.time())
    out_file = str(_P(out_dir) / f"qwen_image_edit_2511_{ts}.png")
    label = "Qwen2511 image edit"
    if prompt:
        label = "Qwen2511: " + (prompt.replace("\n", " ").strip()[:80])
    args = {
        "label": label,
        "sdcli_path": sdcli_path,
        "init_img": init_img,
        "mask_img": mask_img,
        "invert_mask": invert_mask,
        "ref_images": ref_imgs,
        "use_increase_ref_index": use_increase_ref_index,
        "disable_auto_resize_ref_images": disable_auto_resize_ref_images,
        "prompt": prompt,
        "negative": negative,
        "lora_dir": lora_dir,
        "lora_name": lora_name,
        "lora_strength": lora_strength,
        "unet_path": unet_path,
        "llm_path": llm_path,
        "mmproj_path": mmproj_path,
        "vae_path": vae_path,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "width": width,
        "height": height,
        "strength": strength,
        "sampling_method": sampling_method,
        "shift": shift,
        "use_vae_tiling": use_vae_tiling,
        "vae_tile_size": vae_tile_size,
        "vae_tile_overlap": vae_tile_overlap,
        "use_offload": use_offload,
        "use_mmap": use_mmap,
        "use_vae_on_cpu": use_vae_on_cpu,
        "use_clip_on_cpu": use_clip_on_cpu,
        "use_diffusion_fa": use_diffusion_fa,
        "out_file": out_file,
        "outfile": out_file,
    }
    return make_job_json("qwen2511_image_edit", init_img, out_dir, args, str(d["pending"]), priority=500)
# --- FrameVision: queue helpers for Upsc & external commands ---
def _is_video_path(p: str) -> bool:
    try:
        ext = str(p).lower().rsplit('.', 1)[-1]
    except Exception:
        return False
    return ('.' + ext) in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
def enqueue_job(input_path: str, out_dir: str, factor: int, model: str):
    """Convenience wrapper so callers can enqueue without knowing job_type."""
    is_video = _is_video_path(input_path)
    if not out_dir:
        out_dir = default_outdir(is_video, 'upscale')
    jt = 'upscale_video' if is_video else 'upscale_photo'
    return enqueue(jt, input_path, out_dir, factor, model, fmt='png')
# Back-compat aliases so generic code finds us
def add_job(input_path: str, out_dir: str, factor: int, model: str):
    return enqueue_job(input_path, out_dir, factor, model)
def _infer_seedvr2_input_from_cmd(cmd):
    try:
        if isinstance(cmd, (list, tuple)):
            parts=list(cmd)
        elif isinstance(cmd, str):
            import shlex
            parts=shlex.split(cmd)
        else:
            return ''
        # heuristic: input is first argument after inference_cli.py
        for i, part in enumerate(parts):
            if isinstance(part, str) and part.lower().endswith('inference_cli.py'):
                if i+1 < len(parts):
                    return str(parts[i+1])
        # fallback: first existing file path in args
        import os
        for part in parts:
            if isinstance(part, str) and os.path.exists(part):
                return part
    except Exception:
        pass
    return ''
def enqueue_seedvr2(job: dict):
    """Enqueue a SeedVR2 job as a first-class queue entry (type='seedvr2').
    Accepts a dict with keys like: cmd, cwd, env, output/outfile, input, out_dir, engine.
    """
    from helpers.job_helper import make_job_json
    d = jobs_dirs()
    args = dict(job) if isinstance(job, dict) else {'job': job}
    cmd = args.get('cmd') or args.get('command')
    if not cmd:
        raise RuntimeError('enqueue_seedvr2: missing job["cmd"]')
    inp = str(job.get('input') or '').strip()
    if not inp:
        inp = _infer_seedvr2_input_from_cmd(cmd)
    out_file = str(job.get('output') or job.get('outfile') or job.get('out_file') or '').strip()
    out_dir = str(job.get('out_dir') or '').strip()
    if not out_dir:
        try:
            import os
            if out_file:
                out_dir = os.path.dirname(out_file)
        except Exception:
            out_dir = ''
    if not out_dir:
        # SeedVR2 is mostly video; default to video upscale folder
        out_dir = default_outdir(True, 'upscale')
    # Ensure UI and worker have the fields they expect
    args.setdefault('engine', 'seedvr2')
    args.setdefault('outfile', out_file)
    args.setdefault('output', out_file)
    args.setdefault('cwd', job.get('cwd'))
    if job.get('env') is not None:
        args['env'] = job.get('env')
    return make_job_json('seedvr2', inp, out_dir, args, str(d['pending']), priority=500)
def enqueue_external(job: dict):
    """Enqueue a single external command job via tools_ffmpeg.
    Expected job keys include:
      - cmd: list/str command to execute (required)
      - out_dir: optional output directory for the job
      - any additional metadata (prompt, label, outfile, etc.) will be stored
        in the job's "args" dict and is available to the queue UI.
    """
    import os as _os
    out_dir = job.get('out_dir') or default_outdir(False, 'tools')
    # Copy all fields into args so downstream code (thumbnails, output
    # resolver, etc.) can see things like outfile/label/prompt.
    args = dict(job) if isinstance(job, dict) else {'cmd': job}
    cmd = args.get('cmd') or args.get('ffmpeg_cmd')
    if not cmd:
        raise RuntimeError('enqueue_external: missing job["cmd"]')
    # --- Fix common Python -c one-liner issue (while after ';') ---
    # Some planner lock jobs pass python code like: "...;while cond: ..." which
    # is invalid syntax. Python requires 'while' to start a new statement.
    try:
        if isinstance(cmd, (list, tuple)):
            cmd = list(cmd)
            if '-c' in cmd:
                i = cmd.index('-c')
                if i + 1 < len(cmd) and isinstance(cmd[i + 1], str):
                    code = cmd[i + 1]
                    if ';while' in code:
                        code = code.replace(';while ', '\nwhile ').replace(';while\n', '\nwhile\n')
                        code = code.replace(';while not', '\nwhile not')
                        cmd[i + 1] = code
                        args['cmd'] = cmd
        elif isinstance(cmd, str):
            # nothing to do (string cmds are opaque)
            pass
    except Exception:
        pass
    # --- Make tools jobs visible in the Queue UI ---
    # Many queue UIs assume job['input'] points to an existing file (for
    # thumbnails / labels). tools_ffmpeg jobs often have empty input.
    # Create a tiny marker file in out_dir and use it as input.
    input_path = ''
    try:
        from pathlib import Path as _P
        od = _P(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        marker = od / '_tools_job.txt'
        if not marker.exists():
            label = ''
            try:
                label = str(args.get('label') or args.get('title') or 'Tools job')
            except Exception:
                label = 'Tools job'
            try:
                marker.write_text(label[:200], encoding='utf-8')
            except Exception:
                # best-effort only
                pass
        input_path = str(marker)
    except Exception:
        input_path = ''
    # out_dir is carried separately in the job JSON; don't duplicate it in args
    args.pop('out_dir', None)
    return enqueue_tool_job('tools_ffmpeg', input_path, out_dir, args, priority=550)
# <<< FRAMEVISION_QWEN_END
# --- LTX 2.3 queue integration ------------------------------------------------
def enqueue_ltx23_from_widget(inner):
    """Enqueue LTX 2.3 the same boring way other Tools-tab commands are queued.
    This intentionally uses the generic tools_ffmpeg command runner instead of a
    special LTX worker path. The LTX UI already knows how to build the correct
    command; Queue only stores and runs that command.
    """
    try:
        # Keep the same validation as the direct Run button.
        fn = getattr(inner, "validate_before_run", None)
        if callable(fn) and not bool(fn()):
            return False
    except Exception:
        # Validation is best-effort only; build_command will still fail loudly if
        # something important is missing.
        pass
    try:
        build = getattr(inner, "build_command", None)
        if not callable(build):
            raise RuntimeError("LTX UI widget does not expose build_command().")
        program, args, output_path, _extra = build(randomize_seed=True, prepare_video_inputs=True)
    except Exception as exc:
        raise RuntimeError(f"Could not build LTX command: {exc}")
    from pathlib import Path as _P
    cmd = [str(program)] + [str(x) for x in (args or [])]
    output_path = _P(str(output_path))
    out_dir = str(output_path.parent)
    def _txt_attr(name, default=""):
        try:
            obj = getattr(inner, name, None)
            if obj is None:
                return default
            if hasattr(obj, "toPlainText"):
                return str(obj.toPlainText()).strip()
            if hasattr(obj, "text"):
                return str(obj.text()).strip()
            if hasattr(obj, "currentText"):
                return str(obj.currentText()).strip()
        except Exception:
            pass
        return default
    prompt = _txt_attr("prompt_edit", "")
    label_prompt = (prompt or output_path.stem).replace("\n", " ").strip()
    if len(label_prompt) > 82:
        label_prompt = label_prompt[:82].rstrip()
    label = "LTX 2.3: " + (label_prompt or output_path.stem)
    try:
        cwd = _txt_attr("ltx_root_row", "")
    except Exception:
        cwd = ""
    if not cwd:
        try:
            cwd = str(output_path.parents[2])
        except Exception:
            cwd = ""
    # Optional remux metadata is kept for future Queue UI/worker support, but the
    # first goal is to run the exact LTX command through the same generic queue
    # path as Tools-tab jobs.
    remux_audio = False
    audio_path = ""
    try:
        should = getattr(inner, "_should_remux_audio", None)
        remux_audio = bool(should()) if callable(should) else False
    except Exception:
        remux_audio = False
    try:
        audio_path = _txt_attr("audio_row", "")
    except Exception:
        audio_path = ""
    def _checked_attr(name):
        try:
            obj = getattr(inner, name, None)
            return bool(obj and obj.isChecked())
        except Exception:
            return False
    start_video_path = ""
    end_video_path = ""
    ffmpeg_path = ""
    try:
        start_video_path = _txt_attr("start_video_row", "")
    except Exception:
        start_video_path = ""
    try:
        end_video_path = _txt_attr("end_video_row", "")
    except Exception:
        end_video_path = ""
    try:
        ffmpeg_path = _txt_attr("ffmpeg_row", "")
    except Exception:
        ffmpeg_path = ""
    prepared_video_frame_paths = []
    for _attr in ("_prepared_start_frame_path", "_prepared_end_frame_path"):
        try:
            _value = str(getattr(inner, _attr, "") or "").strip()
        except Exception:
            _value = ""
        if _value:
            prepared_video_frame_paths.append(_value)
    glue_videos = bool(_checked_attr("glue_input_videos_check") and (start_video_path or end_video_path))
    job = {
        "cmd": cmd,
        "cwd": cwd,
        "out_dir": out_dir,
        "outfile": str(output_path),
        "scan_dir": out_dir,
        "scan_ext": output_path.suffix or ".mp4",
        "label": label,
        "title": label,
        "engine": "ltx23",
        "prompt": prompt,
        "remux_audio": remux_audio,
        "audio_path": audio_path,
        "ltx_glue_input_videos": glue_videos,
        "ltx_start_video_path": start_video_path,
        "ltx_end_video_path": end_video_path,
        "ltx_temp_video_frame_paths": prepared_video_frame_paths,
        "ffmpeg_path": ffmpeg_path,
    }
    return enqueue_external(job)
def enqueue_ltx23_generate(job: dict):
    """Compatibility wrapper for callers that already prepared an LTX job dict."""
    args = dict(job or {})
    args.setdefault("engine", "ltx23")
    return enqueue_external(args)


def enqueue_planner_generate(job: dict, out_dir: str = '', title: str = '', slug: str = '', priority: int = 120):
    """Enqueue a real Planner pipeline job for the FrameVision worker.

    This is different from the old Planner "queue lock" job.  The worker owns
    the actual Planner pipeline now; the UI only writes the job payload into
    jobs/pending and can be closed/restarted without pretending to run locally.
    """
    try:
        from helpers.job_helper import make_job_json
    except Exception:
        from job_helper import make_job_json
    try:
        import json as _json
        import time as _time
        d = jobs_dirs()
        payload = dict(job or {})
        out_dir = str(out_dir or payload.get('out_dir') or default_outdir(True, 'planner')).strip()
        if not out_dir:
            out_dir = default_outdir(True, 'planner')
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        job_id = str(payload.get('job_id') or payload.get('id') or '').strip()
        title = str(title or payload.get('title') or '').strip()
        slug = str(slug or payload.get('slug') or '').strip()
        label_base = title or payload.get('prompt') or job_id or 'Planner job'
        label = 'Planner: ' + str(label_base).replace('\n', ' ').strip()[:90]

        marker = od / '_planner_queue_job.txt'
        try:
            marker.write_text(
                f"label={label}\n"
                f"planner_job_id={job_id}\n"
                f"created_at={_time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                encoding='utf-8',
            )
        except Exception:
            pass

        args = {
            'planner_job': payload,
            'out_dir': str(od),
            'title': title,
            'slug': slug,
            'label': label,
            'engine': 'planner',
            'worker_mode': 'real_planner_pipeline',
        }
        qid = make_job_json('planner_generate', str(marker), str(od), args, str(d['pending']), priority=int(priority))
        try:
            # Add the real FrameVision queue id to the marker so Planner can later
            # count pending/running/done status without using a fake done flag.
            prev = marker.read_text(encoding='utf-8', errors='replace') if marker.exists() else ''
            marker.write_text(
                prev
                + f"fv_job_id={qid}\n"
                + f"pending_dir={d.get('pending')}\n"
                + f"out_dir={od}\n",
                encoding='utf-8',
            )
        except Exception:
            pass
        return qid
    except Exception as e:
        try:
            print('[queue] enqueue_planner_generate failed:', e)
        except Exception:
            pass
        return False
# -----------------------------------------------------------------------------
def enqueue_resize_job(input_path: str, out_dir: str, cmd, outfile: str = '', label: str = 'Resize') -> bool:
    """Enqueue a resize/convert FFmpeg job using the generic external runner.
    Uses the real input path so the queue row stays visible and identifiable.
    """
    try:
        from helpers.job_helper import make_job_json
        from helpers.queue_adapter import jobs_dirs
    except Exception:
        from job_helper import make_job_json
        from queue_adapter import jobs_dirs
    try:
        d = jobs_dirs()
        args = {
            'cmd': list(cmd) if isinstance(cmd, (list, tuple)) else cmd,
            'label': str(label or 'Resize').strip() or 'Resize',
            'outfile': str(outfile or '').strip(),
        }
        inp = str(input_path or '').strip()
        if not inp:
            inp = str((Path(out_dir) / '_resize_job.txt').resolve())
            try:
                Path(inp).parent.mkdir(parents=True, exist_ok=True)
                if not Path(inp).exists():
                    Path(inp).write_text(args['label'][:200], encoding='utf-8')
            except Exception:
                pass
        return bool(make_job_json('tools_ffmpeg', inp, str(out_dir), args, str(d['pending']), priority=560))
    except Exception as e:
        try:
            print('[queue] enqueue_resize_job failed:', e)
        except Exception:
            pass
        return False
def default_txt2img_outdir():
    from pathlib import Path
    base = Path('.').resolve()
    d = base/'output'/'photo'/'txt2img'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
def enqueue_txt2img(job: dict) -> bool:
    """Enqueue txt2img; fan-out one row per seed. Always queue for batch>1."""
    try:
        from helpers.job_helper import make_job_json
        from helpers.queue_adapter import jobs_dirs, default_txt2img_outdir
        import time as _time, random as _random
        d = jobs_dirs()
        out_dir = job.get("output") or default_txt2img_outdir()
        # first line title = first 30 chars of prompt
        preview = ((job.get("prompt") or "").strip()[:30]) or "[txt2img]"
        # Keys to pass to worker
        keys = [
            "prompt","negative","seed","seed_policy","batch","cfg_scale",
            "width","height","steps","sampler","flow_shift","offload_cpu","model_path",
            "lora_path","lora_scale","lora2_path","lora2_scale",
            "attn_slicing","vae_device","gpu_index","threads",
            "format","filename_template","hires_helper","fit_check",
            "vram_profile","engine",
            # Z-Image GGUF selective edit / img2img
            "selective_edit","source_image","mask_image","init_image_enabled","init_image",
            "img2img_strength","strength","gguf_model_path","gguf_instruct_path","gguf_vae_path",
            "sd_cli_path","temp_job_dir","keep_temp","model_name","model","backend",
            "assistant_origin","assistant_chat_only"
        ]
        base = {k: job.get(k) for k in keys if k in job}
        base["label"] = preview
        # Avoid SDXL model name bleeding into Z-Image queue rows
        try:
            ek = str(job.get('engine') or '').lower().strip()
            if ek in ('zimage', 'zimage_gguf'):
                base.pop('model_path', None)
                if ek == 'zimage_gguf' and base.get('selective_edit'):
                    base.setdefault('model_name', 'Z-Image-Turbo GGUF Selective Edit')
                    base.setdefault('model', 'Z-Image-Turbo GGUF Selective Edit')
                else:
                    base.setdefault('model_name', 'Z-Image-Turbo')
                    base.setdefault('model', 'Z-Image-Turbo')
        except Exception:
            pass
        # seeds fan-out
        batch = int(job.get("batch") or 1)
        seed0 = int(job.get("seed") or 0)
        policy = (job.get("seed_policy") or "fixed").lower()
        if batch <= 1:
            items = [(seed0, base)]
        else:
            if policy == "increment":
                seeds = [seed0 + i for i in range(batch)]
            elif policy == "random":
                rng = _random.Random(seed0 if seed0 else int(_time.time()))
                seeds = [rng.randint(0, 2_147_483_647) for _ in range(batch)]
            else:
                seeds = [seed0 for _ in range(batch)]
            items = [(s, base) for s in seeds]
        prio = 550 if job.get("run_now") else 650
        ok_all = True
        for s, base_args in items:
            args = dict(base_args)
            args["seed"] = int(s)
            args["batch"] = 1  # per-row
            # enqueue with EMPTY input string (prompt is not a path)
            ok = make_job_json("txt2img", "", out_dir, args, str(d["pending"]), priority=prio)
            ok_all = bool(ok_all and ok)
        return bool(ok_all)
    except Exception as e:
        try: print("[queue] enqueue_txt2img failed:", e)
        except Exception: pass
        return False
def enqueue_hidream_from_widget(inner, mode: str = "create"):
    """Enqueue a HiDream BF16 job from the embedded HiDream UI.
    mode:
      - create
      - edit
      - multi_reference
    """
    import time as _time
    from pathlib import Path as _P
    try:
        from helpers.job_helper import make_job_json
        from helpers.queue_adapter import jobs_dirs
    except Exception:
        from job_helper import make_job_json
        from queue_adapter import jobs_dirs
    d = jobs_dirs()
    def _txt(attr: str, plain: bool = False) -> str:
        obj = getattr(inner, attr, None)
        if obj is None:
            return ""
        try:
            return str(obj.toPlainText()).strip() if plain else str(obj.text()).strip()
        except Exception:
            return ""
    def _model_key() -> str:
        try:
            fn = getattr(inner, 'current_model_key', None)
            if callable(fn):
                key = str(fn() or '').strip()
                if key:
                    return key
        except Exception:
            pass
        try:
            return str(inner.model_combo.currentData() or 'base').strip() or 'base'
        except Exception:
            return 'base'
    mode = str(mode or 'create').strip().lower()
    if mode not in ('create', 'edit', 'multi_reference', 'multi', 'multi-ref'):
        raise RuntimeError(f'Unknown HiDream queue mode: {mode}')
    if mode in ('multi', 'multi-ref'):
        mode = 'multi_reference'
    model_key = _model_key()
    out_dir = _txt('output_dir_edit')
    if not out_dir:
        try:
            out_dir = str(_base_root() / 'models' / 'hidream_bf16' / 'results')
        except Exception:
            out_dir = str(_P('.') / 'models' / 'hidream_bf16' / 'results')
    prompt = ''
    raw_prompt = ''
    refs = []
    ref_roles = []
    keep_original_aspect = False
    ui_key = 'main'
    negative = ''
    if mode == 'create':
        prompt = _txt('prompt_edit', plain=True)
        ui_key = 'main'
        negative = _txt('negative_prompt_edit', plain=True)
        if not prompt:
            raise RuntimeError('Enter a prompt first.')
    elif mode == 'edit':
        prompt = _txt('edit_prompt', plain=True)
        ui_key = 'edit'
        negative = _txt('edit_negative_prompt_edit', plain=True)
        try:
            refs = list(getattr(inner.ref_list, 'paths')())
        except Exception:
            refs = []
        try:
            keep_original_aspect = bool(inner.keep_aspect.isChecked())
        except Exception:
            keep_original_aspect = False
        if not prompt:
            raise RuntimeError('Enter an edit instruction first.')
        if not refs:
            raise RuntimeError('Add at least one reference image.')
    else:
        raw_prompt = _txt('multi_prompt', plain=True)
        ui_key = 'multi'
        negative = _txt('multi_negative_prompt_edit', plain=True)
        try:
            ref_roles = list(getattr(inner.multi_ref_list, 'references')())
        except Exception:
            ref_roles = []
        refs = [str(r.get('path') or '').strip() for r in ref_roles if str(r.get('path') or '').strip()]
        try:
            prompt = str(inner.multi_reference_prompt_with_roles(raw_prompt, ref_roles)).strip()
        except Exception:
            prompt = raw_prompt
        if not raw_prompt:
            raise RuntimeError('Enter a multi-reference instruction first.')
        if not refs:
            raise RuntimeError('Add at least one reference image.')
    try:
        settings = dict(getattr(inner, 'selected_generation_settings')(ui_key, negative))
    except Exception:
        settings = {}
    prefix = f'hidream_{model_key}_{mode}'
    try:
        out_path = str(getattr(inner, 'output_path')(prefix))
    except Exception:
        ts = int(_time.time())
        out_path = str(_P(out_dir) / f'{prefix}_{ts}.png')
    title_prompt = raw_prompt or prompt
    short = (title_prompt or '').replace('\n', ' ').strip()
    if len(short) > 80:
        short = short[:80]
    label = f'HiDream {mode.replace("_", " ")}'
    if short:
        label += ': ' + short
    # Keep both names so older/newer worker builds cannot accidentally drop references.
    # The worker primarily reads 'refs', while other FrameVision image-edit jobs use
    # 'ref_images'. Supplying both makes edit and multi-reference queue jobs robust.
    args = {
        'label': label,
        'mode': mode,
        'model_key': model_key,
        'prompt': prompt,
        'raw_prompt': raw_prompt,
        'refs': refs,
        'ref_images': refs,
        'ref_roles': ref_roles,
        'keep_original_aspect': keep_original_aspect,
        'settings': settings,
        'output_path': out_path,
        'out_file': out_path,
        'outfile': out_path,
        'out_dir': out_dir,
        'hidream_ref_count': len(refs),
        'assistant_origin': str(getattr(inner, 'assistant_origin', '') or ''),
        'assistant_chat_only': bool(getattr(inner, 'assistant_chat_only', False)),
    }
    input_path = refs[0] if refs else ''
    return make_job_json('hidream_generate', input_path, out_dir, args, str(d['pending']), priority=500)
def enqueue_wan22_from_widget(inner) -> bool:
    """Queue Wan 2.2 by using the exact command built by helpers/wan22.py.

    Older code tried to reconstruct all Wan settings in queue_adapter and then
    asked worker.py to rebuild the command again. That easily drifts from the
    direct-run path. Direct run works, so queue now stores and runs that same
    command via the generic tools_ffmpeg runner.
    """
    try:
        from pathlib import Path as _P
        import time as _time
        import os as _os

        if not hasattr(inner, "_build_command") or not callable(getattr(inner, "_build_command")):
            raise RuntimeError("Wan22 UI does not expose _build_command().")

        py, cmd_args, cwd = inner._build_command()
        cmd = [str(py)] + [str(x) for x in (cmd_args or [])]

        # Find the output path from the direct command. Normal Wan uses --save_file;
        # Turbo uses --output_path. Fall back to _last_run_out_path when available.
        def _arg_after(flag: str) -> str:
            try:
                parts = list(cmd_args or [])
                if flag in parts:
                    i = parts.index(flag)
                    if i + 1 < len(parts):
                        return str(parts[i + 1])
            except Exception:
                pass
            return ""

        out_file = _arg_after("--save_file") or _arg_after("--output_path")
        if not out_file:
            try:
                out_file = str(getattr(inner, "_last_run_out_path", "") or "")
            except Exception:
                out_file = ""
        out_path = _P(out_file).expanduser() if out_file else _P(default_outdir(True, "wan22")) / f"wan22_{int(_time.time())}.mp4"
        if not out_path.is_absolute():
            try:
                out_path = (_P(str(cwd)).resolve() / out_path).resolve()
            except Exception:
                out_path = out_path.resolve()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Input path for the queue row.
        input_path = ""
        try:
            mode_txt = str(inner.cmb_mode.currentText()).strip().lower()
        except Exception:
            mode_txt = ""
        try:
            img = str(inner.ed_image.text()).strip()
        except Exception:
            img = ""
        if "image" in mode_txt and img:
            input_path = img
        else:
            try:
                prompt = str(inner.ed_prompt.toPlainText()).strip()
            except Exception:
                prompt = ""
            sidecar = out_path.parent / f".wan22_prompt_{int(_time.time())}.txt"
            try:
                sidecar.write_text((prompt or "Wan2.2 text2video job")[:2000], encoding="utf-8")
                input_path = str(sidecar)
            except Exception:
                input_path = ""

        # Label + metadata for the Queue UI.
        try:
            prompt_preview = str(inner.ed_prompt.toPlainText()).replace("\n", " ").strip()
        except Exception:
            prompt_preview = ""
        try:
            turbo_enabled = bool(inner._wan_turbo_enabled())
        except Exception:
            turbo_enabled = False
        label = ("Wan2.2 Turbo: " if turbo_enabled else "Wan2.2: ") + (prompt_preview[:80] or ("image2video" if input_path and not input_path.endswith(".txt") else "text2video"))

        # Match direct QProcess environment flags.
        env = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        try:
            if hasattr(inner, "_wan_flash_attention_enabled") and (not bool(inner._wan_flash_attention_enabled())):
                env["FV_WAN_DISABLE_FLASH_ATTENTION"] = "1"
        except Exception:
            pass
        try:
            if hasattr(inner, "_wan_crawl_guard_enabled"):
                env["FV_WAN_SHARED_MEM_GUARD"] = "1" if bool(inner._wan_crawl_guard_enabled()) else "0"
        except Exception:
            pass

        args = {
            "label": label,
            "engine": "wan22_turbo" if turbo_enabled else "wan22",
            "cmd": cmd,
            "cwd": str(cwd),
            "outfile": str(out_path),
            "out_file": str(out_path),
            "env": env,
            "prompt": prompt_preview,
            "runner": "direct_command",
        }
        try:
            args["steps"] = int(getattr(inner, "spn_steps").value())
        except Exception:
            pass
        try:
            args["frames"] = int(getattr(inner, "spn_frames").value())
        except Exception:
            pass
        try:
            args["seed"] = int(getattr(inner, "spn_seed").value())
        except Exception:
            pass

        # Use generic command runner so the worker cannot drift from the direct Wan UI.
        return bool(enqueue_tool_job("tools_ffmpeg", str(input_path or ""), str(out_path.parent), args, priority=550))
    except Exception as e:
        try:
            print("[queue] enqueue_wan22_from_widget failed:", e)
        except Exception:
            pass
        return False

def default_ace_outdir() -> str:
    from pathlib import Path
    base = Path('.').resolve()
    d = base / 'output' / 'ace'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
def enqueue_ace_from_widget(inner) -> bool:
    """Read fields from AceUI and enqueue an ACE-Step text-to-music or audio-to-audio job."""
    try:
        from helpers.job_helper import make_job_json
        from helpers.queue_adapter import jobs_dirs
        from pathlib import Path as _Path
        d = jobs_dirs()
        def _val(obj, name, default=None):
            try:
                w = getattr(obj, name, None)
                if hasattr(w, 'value'):
                    return w.value()
            except Exception:
                pass
            return default
        prompt = ''
        try:
            if hasattr(inner, 'prompt_edit'):
                prompt = str(inner.prompt_edit.text()).strip()
        except Exception:
            prompt = ''
        negative = ''
        try:
            if hasattr(inner, 'negative_edit'):
                negative = str(inner.negative_edit.text()).strip()
        except Exception:
            negative = ''
        lyrics = ''
        try:
            if hasattr(inner, 'lyrics_edit'):
                lyrics = str(inner.lyrics_edit.toPlainText()).strip()
        except Exception:
            lyrics = ''
        duration = int(_val(inner, 'duration_spin', 60) or 60)
        steps = int(_val(inner, 'steps_spin', 60) or 60)
        try:
            guidance = float(getattr(inner, 'guidance_spin').value())
        except Exception:
            guidance = 15.0
        scheduler = 'euler'
        try:
            if hasattr(inner, 'scheduler_combo'):
                scheduler = str(inner.scheduler_combo.currentText()).strip()
        except Exception:
            pass
        cfg_type = 'apg'
        try:
            if hasattr(inner, 'cfg_combo'):
                cfg_type = str(inner.cfg_combo.currentText()).strip()
        except Exception:
            pass
        seed = int(_val(inner, 'seed_spin', 0) or 0)
        out_rel = ''
        try:
            out_rel = str(inner.output_edit.text()).strip()
        except Exception:
            out_rel = ''
        out_dir = out_rel or default_ace_outdir()
        ref_audio_input = ''
        try:
            if hasattr(inner, 'ref_audio_edit'):
                ref_audio_input = str(inner.ref_audio_edit.text()).strip()
        except Exception:
            ref_audio_input = ''
        try:
            if ref_audio_input:
                p = _Path(ref_audio_input)
                if not p.is_absolute():
                    p = _Path('.').resolve() / p
                ref_audio_input = str(p)
        except Exception:
            pass
        try:
            v = int(getattr(inner, 'ref_strength_slider').value())
            ref_strength = max(0, min(100, v)) / 100.0
        except Exception:
            ref_strength = 0.5
        job_type = 'ace_audio2audio' if ref_audio_input else 'ace_text2music'
        label = (prompt[:80] or ('ACE-Step ' + ('audio2audio' if ref_audio_input else 'text2music')))
        args = {
            'prompt': prompt,
            'negative_prompt': negative,
            'lyrics': lyrics,
            'audio_duration': float(duration),
            'infer_step': int(steps),
            'guidance_scale': float(guidance),
            'scheduler_type': scheduler or 'euler',
            'cfg_type': cfg_type or 'apg',
            'seed': int(seed),
            'output_rel': out_rel or 'output/ace',
            'ref_audio_input': ref_audio_input,
            'ref_audio_strength': float(ref_strength),
            'audio2audio_enable': bool(ref_audio_input),
            'label': label,
        }
        if ref_audio_input:
            input_path = ref_audio_input
        else:
            try:
                base_dir = _Path(out_dir) if out_dir else _Path(default_ace_outdir())
            except Exception:
                base_dir = _Path(default_ace_outdir())
            dummy = base_dir / 'ace_text_prompt.txt'
            try:
                if not dummy.exists():
                    dummy.parent.mkdir(parents=True, exist_ok=True)
                    snippet = (prompt or label or 'ACE text2music job')[:160]
                    dummy.write_text(snippet, encoding='utf-8')
                input_path = str(dummy)
            except Exception:
                input_path = ''
        pending_dir = d['pending']
        return make_job_json(job_type, input_path, out_dir, args, str(pending_dir))
    except Exception as e:
        try:
            print('[queue] enqueue_ace_from_widget failed:', e)
        except Exception:
            pass
        return False
def enqueue_hiar_from_widget(inner) -> bool:
    """Read fields from HiARPane and enqueue a queued HiAR video generation job."""
    try:
        from pathlib import Path as _P
        def _txt(name, default=""):
            try:
                obj = getattr(inner, name)
                if hasattr(obj, 'text'):
                    return str(obj.text()).strip()
            except Exception:
                pass
            return default
        def _plain(name, default=""):
            try:
                obj = getattr(inner, name)
                if hasattr(obj, 'toPlainText'):
                    return str(obj.toPlainText()).strip()
            except Exception:
                pass
            return default
        def _val(name, default=0):
            try:
                obj = getattr(inner, name)
                if hasattr(obj, 'value'):
                    return obj.value()
            except Exception:
                pass
            return default
        def _checked(name, default=False):
            try:
                obj = getattr(inner, name)
                if hasattr(obj, 'isChecked'):
                    return bool(obj.isChecked())
            except Exception:
                pass
            return bool(default)
        def _current(name, default=""):
            try:
                obj = getattr(inner, name)
                if hasattr(obj, 'currentText'):
                    return str(obj.currentText()).strip()
            except Exception:
                pass
            return default
        repo_root = _txt('repo_root_edit')
        python_path = _txt('python_edit')
        config_path = _txt('config_edit')
        checkpoint_path = _txt('checkpoint_edit')
        prompt_file = _txt('prompt_file_edit')
        extended_prompt_path = _txt('extended_prompt_edit')
        output_dir = _txt('output_edit')
        prompt_text = _plain('prompt_text')
        negative_prompt = _plain('negative_prompt_box')
        frames = int(_val('frames_spin', 66))
        seed = int(_val('seed_spin', 0))
        guidance = float(_val('guidance_spin', 3.0))
        samples = int(_val('samples_spin', 1))
        inference_method = _current('inference_method_combo', 'timestep_first') or 'timestep_first'
        frame_first_blocks = int(_val('frame_first_blocks_spin', 1))
        use_ema = _checked('use_ema_check', False)
        save_with_index = _checked('save_with_index_check', True)
        auto_open_output = _checked('auto_open_output_check', False)
        if not output_dir:
            try:
                output_dir = str(getattr(inner, 'default_output_dir'))
            except Exception:
                output_dir = str(_P('.') / 'output' / 'hiar')
        _P(output_dir).mkdir(parents=True, exist_ok=True)
        if not prompt_text and not prompt_file:
            raise RuntimeError('Provide prompt text or select a prompt file before queueing HiAR.')
        label_source = prompt_text or ''
        if not label_source and prompt_file:
            try:
                label_source = _P(prompt_file).read_text(encoding='utf-8', errors='replace')
            except Exception:
                label_source = _P(prompt_file).stem
        label_source = ' '.join(str(label_source).split())
        label = (label_source[:80] or 'HiAR video')
        dummy_input = ''
        if prompt_file and _P(prompt_file).exists():
            dummy_input = prompt_file
        else:
            dummy = _P(output_dir) / 'hiar_prompt.txt'
            try:
                if prompt_text:
                    dummy.write_text((prompt_text[:500] or 'HiAR prompt') + '\n', encoding='utf-8')
                elif not dummy.exists():
                    dummy.write_text('HiAR prompt\n', encoding='utf-8')
                dummy_input = str(dummy)
            except Exception:
                dummy_input = ''
        args = {
            'label': label,
            'repo_root': repo_root,
            'python_path': python_path,
            'config_path': config_path,
            'checkpoint_path': checkpoint_path,
            'prompt_file': prompt_file,
            'prompt_text': prompt_text,
            'extended_prompt_path': extended_prompt_path,
            'output_folder': output_dir,
            'num_output_frames': frames,
            'seed': seed,
            'guidance_scale': guidance,
            'negative_prompt': negative_prompt,
            'num_samples': samples,
            'inference_method': inference_method,
            'num_frame_first_blocks': frame_first_blocks,
            'use_ema': use_ema,
            'save_with_index': save_with_index,
            'auto_open_output': auto_open_output,
        }
        return bool(enqueue_tool_job('hiar_generate', dummy_input, output_dir, args, priority=550))
    except Exception as e:
        try:
            print('[queue] enqueue_hiar_from_widget failed:', e)
        except Exception:
            pass
        return False
def default_qwentts_outdir() -> str:
    base = _base_root()
    d = base / 'output' / 'audio' / 'qwen3tts'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)

def default_dotstts_outdir():
    from pathlib import Path
    base = _base_root()
    d = base / 'output' / 'dots_tts'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def enqueue_dotstts_from_widget(inner) -> str:
    """Read the FrameVision dots.tts helper and enqueue it via the generic tools_ffmpeg runner."""
    try:
        from pathlib import Path as _P
        import time as _time
        exe, args, env, workdir = inner._build_command()
        out_path = _P(str(inner._output_path())).resolve()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        text = ''
        try:
            text = str(inner.text_edit.toPlainText()).replace('\n', ' ').strip()
        except Exception:
            text = ''
        preview = text[:80].rstrip() + ('...' if len(text) > 80 else '')
        label = 'dots.tts' + (f' — {preview}' if preview else '')

        prompt_audio = ''
        try:
            prompt_audio = str(inner.prompt_audio_edit.text()).strip()
        except Exception:
            prompt_audio = ''

        cmd = [str(exe)] + [str(x) for x in (args or [])]
        job_args = {
            'engine': 'dots_tts',
            'label': label,
            'cmd': cmd,
            'cwd': str(workdir),
            'outfile': str(out_path),
            'out_file': str(out_path),
            'env': dict(env or {}),
            'prompt': text,
            'prompt_audio': prompt_audio,
            'steps': int(getattr(inner, 'steps_spin').value()) if hasattr(inner, 'steps_spin') else None,
        }
        # Use a real audio reference as queue input when available; otherwise create a tiny sidecar
        # so the queue row still has a stable input path.
        input_path = prompt_audio
        if not input_path:
            sidecar = out_path.parent / f".dots_tts_{int(_time.time())}.txt"
            try:
                sidecar.write_text(text[:1000], encoding='utf-8')
                input_path = str(sidecar)
            except Exception:
                input_path = ''
        return enqueue_tool_job('tools_ffmpeg', str(input_path or ''), str(out_path.parent), job_args, priority=610)
    except Exception as e:
        raise RuntimeError(f'Could not enqueue dots.tts job: {e}')


def enqueue_qwentts_from_widget(inner) -> bool:
    """Read fields from QwenTTSUI and enqueue a queued Qwen TTS generation job."""
    try:
        from pathlib import Path as _P
        try:
            from helpers.job_helper import make_job_json
            from helpers.queue_adapter import jobs_dirs
        except Exception:
            from job_helper import make_job_json
            from queue_adapter import jobs_dirs
        d = jobs_dirs()
        try:
            mode, payload = inner._collect_payload()
        except Exception as e:
            raise RuntimeError(f'Failed to collect Qwen TTS payload: {e}')
        model_path = str(payload.get('model_path') or '').strip()
        if not model_path or not _P(model_path).exists():
            raise RuntimeError('Qwen TTS model folder missing.')
        out_dir = default_qwentts_outdir()
        output_name = str(((payload.get('common') or {}).get('output_name')) or 'qwen_tts').strip() or 'qwen_tts'
        label_mode = {
            'custom': 'CustomVoice',
            'clone': 'Voice Clone',
            'design': 'Voice Design',
        }.get(str(mode).strip().lower(), 'Qwen TTS')
        label = f'Qwen TTS - {label_mode} - {output_name}'
        input_path = ''
        try:
            if str(mode).strip().lower() == 'clone':
                input_path = str(payload.get('ref_audio_path') or '').strip()
        except Exception:
            input_path = ''
        if not input_path:
            dummy = _P(out_dir) / 'qwentts_queue_job.txt'
            try:
                if not dummy.exists():
                    dummy.parent.mkdir(parents=True, exist_ok=True)
                    snippet = str(payload.get('text') or label)[:160]
                    dummy.write_text(snippet, encoding='utf-8')
                input_path = str(dummy)
            except Exception:
                input_path = model_path
        env_python = str(_P(__file__).resolve().parents[1] / 'environments' / '.qwen3tts' / 'Scripts' / 'python.exe')
        ui_script = str(_P(__file__).resolve().parent / 'qwentts_ui.py')
        args = {
            'mode': mode,
            'payload': payload,
            'env_python': env_python,
            'ui_script': ui_script,
            'label': label,
        }
        return make_job_json('qwentts_generate', input_path, out_dir, args, str(d['pending']))
    except Exception as e:
        try:
            print('[queue] enqueue_qwentts_from_widget failed:', e)
        except Exception:
            pass
        return False
def enqueue_splitglue_ffmpeg(label: str, cmd: list, output_file: str, out_dir: str, input_path: str = '', priority: int = 600):
    """Queue a Split/Glue ffmpeg command through the generic tools_ffmpeg worker path."""
    args = {
        'label': str(label or 'Split/Glue ffmpeg job'),
        'ffmpeg_cmd': [str(x) for x in (cmd or [])],
        'outfile': str(output_file or ''),
        'cwd': str(_base_root()),
    }
    return enqueue_tool_job('tools_ffmpeg', str(input_path or ''), str(out_dir or ''), args, priority=int(priority))
def default_lens_turbo_outdir():
    from pathlib import Path
    base = _base_root()
    d = base / 'output' / 'lens_turbo_u4'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
def enqueue_lens_turbo_u4(job: dict) -> bool:
    """Enqueue a Lens Turbo U4 text-to-image job for the FrameVision worker."""
    try:
        from helpers.job_helper import make_job_json
    except Exception:
        from job_helper import make_job_json
    try:
        d = jobs_dirs()
        params = dict(job.get('params') or job)
        prompt = str(params.get('prompt') or job.get('prompt') or '').strip()
        if not prompt:
            raise RuntimeError('Prompt is empty.')
        params['prompt'] = prompt
        out_dir = str(job.get('output_dir') or params.get('output_dir') or default_lens_turbo_outdir())
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        preview = prompt.replace('\n', ' ').strip()[:80] or 'Lens Turbo U4'
        args = dict(params)
        args['label'] = 'Lens Turbo U4: ' + preview
        args['engine'] = 'lens_turbo_u4'
        prio = 550 if job.get('run_now') else 650
        return bool(make_job_json('lens_turbo_u4', '', out_dir, args, str(d['pending']), priority=prio))
    except Exception as e:
        try:
            print('[queue] enqueue_lens_turbo_u4 failed:', e)
        except Exception:
            pass
        return False
