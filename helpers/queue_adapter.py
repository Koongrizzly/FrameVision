import os
from pathlib import Path
def default_outdir(is_video: bool=False, purpose: str='upscale') -> str:
    base = Path('.').resolve()
    if purpose == 'rife':
        d = base / 'output' / 'video' / 'interpolated'
    elif purpose == 'wan22':
        d = base / 'output' / 'video' / 'wan22'
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
    # out_dir is carried separately in the job JSON; don't duplicate it in args
    args.pop('out_dir', None)
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
            "width","height","steps","sampler","model_path",
            "lora_path","lora_scale","lora2_path","lora2_scale",
            "attn_slicing","vae_device","gpu_index","threads",
            "format","filename_template","hires_helper","fit_check",
            "vram_profile","engine"
        ]
        base = {k: job.get(k) for k in keys if k in job}
        base["label"] = preview

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

        prio = 550 if job.get("run_now") else 600
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


def enqueue_wan22_from_widget(inner) -> bool:
    """Read fields from Wan22Pane and enqueue a Wan 2.2 text/image-to-video job.

    This mirrors the direct-run settings in helpers/wan22.py but sends them
    to the worker queue instead of launching generate.py in-process.
    """
    try:
        # Resolve mode from combo box; fall back to text2video
        try:
            mode = str(inner.cmb_mode.currentText()).strip().lower()
        except Exception:
            mode = "text2video"
        job_type = "wan22_image2video" if "image" in mode else "wan22_text2video"

        # Prompt & image
        try:
            prompt = str(inner.ed_prompt.toPlainText()).strip()
        except Exception:
            prompt = ""
        try:
            image = str(inner.ed_image.text()).strip()
        except Exception:
            image = ""

        if job_type == "wan22_image2video" and not image:
            raise RuntimeError("Start image is required for image2video mode.")

        # Core numeric settings
        try:
            size_str = str(inner.cmb_size.currentText()).strip()
        except Exception:
            size_str = "1280*704"
        try:
            steps = int(inner.spn_steps.value())
        except Exception:
            steps = 30
        try:
            guidance = float(inner.spn_guidance.value())
        except Exception:
            guidance = 7.0
        try:
            frames = int(inner.spn_frames.value())
        except Exception:
            frames = 121
        try:
            seed = int(inner.spn_seed.value())
        except Exception:
            seed = 42
        try:
            random_seed = bool(inner.chk_random_seed.isChecked())
        except Exception:
            random_seed = False

        # Output hint / directory
        try:
            out_hint = str(inner.ed_out.text()).strip()
        except Exception:
            out_hint = ""

        out_dir = ""
        save_file = ""
        if out_hint:
            p = Path(out_hint)
            if not p.suffix:
                p = p.with_suffix(".mp4")
            if p.is_absolute():
                out_dir = str(p.parent)
                save_file = p.name
            else:
                # Treat as relative file name inside default Wan22 folder
                out_dir = default_outdir(True, "wan22")
                save_file = p.name
        else:
            out_dir = default_outdir(True, "wan22")
            save_file = ""

        # Human-friendly label for the queue row
        label = (prompt[:80] or ("Wan2.2 " + ("image2video" if job_type == "wan22_image2video" else "text2video")))

        args = {
            "prompt": prompt,
            "mode": "image2video" if job_type == "wan22_image2video" else "text2video",
            "image": image if job_type == "wan22_image2video" else "",
            "size": size_str or "1280*704",
            "steps": int(steps),
            "guidance": float(guidance),
            "frames": int(frames),
            "seed": int(seed),
            "random_seed": bool(random_seed),
            "save_file": save_file,
            "label": label,
        }

        # For image2video, also store the start image as the job's input.
        # For pure text2video jobs, the queue UI expects some input path,
        # so we create a small text file containing the prompt.
        if job_type == "wan22_image2video":
            input_path = image
        else:
            from pathlib import Path as _Path
            try:
                base_dir = _Path(out_dir) if out_dir else _Path(".").resolve()
            except Exception:
                base_dir = _Path(".").resolve()
            dummy = base_dir / "wan22_text_prompt.txt"
            try:
                if not dummy.exists():
                    dummy.parent.mkdir(parents=True, exist_ok=True)
                    snippet = (prompt or label or "Wan2.2 text2video job")[:160]
                    dummy.write_text(snippet, encoding="utf-8")
                input_path = str(dummy)
            except Exception:
                input_path = ""

        return bool(enqueue_tool_job(job_type, input_path, out_dir, args, priority=550))
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
