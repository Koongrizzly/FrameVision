import os
from pathlib import Path
def default_outdir(is_video: bool=False, purpose: str='upscale') -> str:
    base = Path('.').resolve()
    if purpose == 'rife':
        d = base / 'output' / 'video' / 'interpolated'
    else:
        d = base / 'output' / ('video' if is_video else 'photo') / 'upscaled'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
def _base_root():
    root = Path('.').resolve()
    return root

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
    inp = _read_field(inner, ['edit_input','edit_input_path','line_input','input_path','edit_source'])
    outdir = _read_field(inner, ['edit_outdir','edit_output','line_output','output_dir'])
    scale = _read_field(inner, ['spin_scale','spin_factor'], getter='value')
    if not scale: scale = _safe_int(_read_field(inner, ['edit_scale','line_scale'], getter='text'), 4)
    model = _read_field(inner, ['edit_model','line_model'], getter='text') or _read_field(inner, ['combo_model'], getter='currentText') or 'RealESRGAN-general-x4v3'
    inp = os.path.abspath(inp) if inp else ''; outdir = os.path.abspath(outdir) if outdir else ''
    return inp, outdir, _safe_int(scale, 4), model
def enqueue(job_type, input_path, out_dir, factor, model, fmt='png'):
    from helpers.job_helper import make_job_json
    d = jobs_dirs()
    args = {'factor': int(factor), 'model': model}
    if job_type == 'upscale_photo': args['format'] = fmt
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


def enqueue_tool_job(job_type: str, input_path: str, out_dir: str, args: dict, priority: int=600):
    from helpers.job_helper import make_job_json
    d = jobs_dirs()
    return make_job_json(job_type, input_path, out_dir, args or {}, str(d['pending']), priority=int(priority))

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

def enqueue_external(job: dict):
    """Enqueue a single external command job via tools_ffmpeg.
    Expects job like { 'cmd': [...], 'out_dir': optional }.
    """
    import os as _os
    out_dir = job.get('out_dir') or default_outdir(False, 'tools')
    args = {'cmd': job.get('cmd')}
    if not args['cmd']:
        raise RuntimeError('enqueue_external: missing job["cmd"]')
    # input_path not required for tools job
    return enqueue_tool_job('tools_ffmpeg', '', out_dir, args, priority=550)
# <<< FRAMEVISION_QWEN_END


def default_txt2img_outdir():
    from pathlib import Path
    base = Path('.').resolve()
    d = base/'output'/'photo'/'txt2img'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)

def enqueue_txt2img(job: dict) -> bool:
    """Convenience: enqueue a txt2img job via the standard queue (priority depends on run_now)."""
    try:
        from helpers.queue_adapter import enqueue_tool_job
        preview = (job.get("prompt") or "")[:80] or "[txt2img]"
        out_dir = job.get("output") or default_txt2img_outdir()
        # Filter args to a safe subset for the worker
        keys = [
            "prompt","negative","seed","seed_policy","batch","cfg_scale",
            "width","height","steps","sampler","model_path","lora_path",
            "lora_scale","lora2_path","lora2_scale","attn_slicing","vae_device",
            "gpu_index","threads","format","filename_template","hires_helper","fit_check",
            "vram_profile","a1111_url","qwen_cli_template"
        ]
        args = {k: job.get(k) for k in keys if k in job}
        prio = 550 if job.get("run_now") else 600
        return bool(enqueue_tool_job("txt2img", preview, out_dir, args, priority=prio))
    except Exception as e:
        try:
            print("[queue] enqueue_txt2img failed:", e)
        except Exception:
            pass
        return False

# Use type 'txt2img' for compatibility; worker handles both.
        return make_job_json('txt2img', '', out_dir, args, str(d['pending']), priority=500)
    except Exception as e:
        print('[queue] enqueue_txt2img failed:', e)
        return False
