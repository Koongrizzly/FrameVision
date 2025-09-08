
"""
helpers/gif.py
Unified backend + UI for animated exports (GIF/WebP/APNG).
- Builds the full control group inside Tools via install_ui()
- Collects options via options_from_ui()
- Renders via encode() or encode_with_progress() (UI progress/ETA + "Play result")
"""
from __future__ import annotations

import os, re, shlex, subprocess, time
from dataclasses import dataclass, asdict
from pathlib import Path
try:
    from typing import Literal
except Exception:
    from typing_extensions import Literal
from typing import Optional, List, Dict, Any, Tuple

# ---- Types ----
Format = Literal["gif", "webp", "apng"]
StatsMode = Literal["single", "diff"]
Dither = Literal["sierra2_4a", "bayer", "floyd_steinberg", "none"]
FitMode = Literal["fit", "fill"]
ArPreset = Literal["16:9", "9:16", "4:3", "1:1"]

class GifError(Exception): pass

# ---- Tooltips (texts exactly as requested) ----
TOOLTIPS: Dict[str, str] = {
    "max_colors": "Limits palette size (2–256). Fewer colors = smaller file, more banding.",
    "stats_mode": "diff samples frame-to-frame changes—best when background is steady.",
    "dither": "How missing colors are faked. sierra2_4a = quality, bayer = patterned, floyd_steinberg = classic, none = flat.",
    "two_pass": "Generates a palette, then applies it (GIF). Best quality, slower.",
    "loop": "0 = infinite loop. Any other number plays that many times.",
    "keep_aspect": "Preserve original proportions when resizing.",
    "fit_mode": "Fit pads to target size; Fill center-crops then scales.",
    "ar_preset": "Snap to 16:9, 9:16, 4:3, or 1:1 (applies when resizing).",
    "trim": "Export only a section (seconds). Leave end blank/0 for full length.",
    "boomerang": "Plays forward then reversed for a seamless loop.",
    "frame_step": "Drop every Nth frame (2 keeps every second frame).",
    "keep_alpha": "Preserve alpha (WebP/APNG). GIF uses 1-bit transparency.",
}

# ---- Options ----
@dataclass
class GifOptions:
    format: Format = "gif"
    fps: Optional[int] = None
    two_pass: bool = True
    use_palette: bool = True

    max_colors: int = 256
    stats_mode: StatsMode = "diff"
    dither: Dither = "sierra2_4a"
    bayer_scale: int = 3
    loop: int = 0

    width: Optional[int] = None
    height: Optional[int] = None
    keep_aspect: bool = True
    fit_mode: FitMode = "fit"
    ar_preset: Optional[ArPreset] = None

    crop_x: Optional[float] = None
    crop_y: Optional[float] = None
    crop_w: Optional[float] = None
    crop_h: Optional[float] = None

    trim_start: float = 0.0
    trim_end: Optional[float] = None
    frame_step: int = 1
    boomerang: bool = False

    keep_alpha: bool = True
    lossless_webp: bool = False
    quality_webp: int = 80
    compression_webp: int = 6

    palette_cache_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GifOptions":
        base = cls()
        for k,v in d.items():
            if hasattr(base,k): setattr(base,k,v)
        return base

# ---- Small helpers ----
def _ensure_parent(path: Path): Path(path).parent.mkdir(parents=True, exist_ok=True)

def _ffmpeg_exec(cmd: List[str]):
    try: subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise GifError(f"FFmpeg failed: {' '.join(shlex.quote(x) for x in cmd)}\n{e}") from e

def _ffprobe_from_ffmpeg(ffmpeg: str) -> str:
    if not ffmpeg: return "ffprobe"
    name = Path(ffmpeg).name
    if "ffmpeg" in name:
        return str(Path(ffmpeg).with_name(name.replace("ffmpeg","ffprobe")))
    return "ffprobe"

def _probe_info(input_path: Path, ffprobe: str) -> Dict[str, Any]:
    info = {"fps": None, "duration": None}
    try:
        out = subprocess.check_output([ffprobe, "-v","error","-select_streams","v:0",
                                       "-show_entries","stream=avg_frame_rate",
                                       "-show_entries","format=duration",
                                       "-of","default=nokey=1:noprint_wrappers=1",
                                       str(input_path)], universal_newlines=True)
        vals = [x.strip() for x in out.splitlines() if x.strip()]
        dur = None; fps = None
        for v in vals:
            if "/" in v:
                a,b = v.split("/",1)
                try: fps = float(a)/float(b) if float(b)!=0 else None
                except Exception: pass
            else:
                try: dur = float(v)
                except Exception: pass
        info["fps"] = fps; info["duration"] = dur
    except Exception:
        pass
    return info

def _input_trim_args(opts: GifOptions) -> List[str]:
    args: List[str] = []
    if opts.trim_start and float(opts.trim_start) > 0:
        args += ["-ss", f"{opts.trim_start}"]
    if opts.trim_end is not None:
        try:
            te = float(opts.trim_end); ts = float(opts.trim_start or 0.0)
            if te > 0 and te > ts:
                args += ["-to", f"{te}"]
        except Exception:
            pass
    return args

def _calc_aspect_tuple(preset: ArPreset) -> Tuple[int,int]:
    return {"16:9":(16,9),"9:16":(9,16),"4:3":(4,3),"1:1":(1,1)}[preset]

def apply_ar_preset(opts: GifOptions) -> None:
    if not opts.ar_preset: return
    W,H = _calc_aspect_tuple(opts.ar_preset)
    if opts.width and not opts.height:  opts.height = int(round(opts.width * H / W))
    elif opts.height and not opts.width: opts.width  = int(round(opts.height * W / H))

def _norm_crop_value(val: float, axis: str) -> str:
    if val is None: raise GifError("crop value missing")
    try: f = float(val)
    except Exception: return str(val)
    if 0.0 < f <= 1.0: return f"i{'w' if axis in ('x','w') else 'h'}*{f}"
    return str(int(round(f)))

def _build_crop_expr(opts: GifOptions) -> Optional[str]:
    if opts.crop_w is None or opts.crop_h is None: return None
    w = _norm_crop_value(opts.crop_w,'w'); h = _norm_crop_value(opts.crop_h,'h')
    x = _norm_crop_value(opts.crop_x,'x') if opts.crop_x is not None else f"(iw-{w})/2"
    y = _norm_crop_value(opts.crop_y,'y') if opts.crop_y is not None else f"(ih-{h})/2"
    return f"crop=w={w}:h={h}:x={x}:y={y}"

def _build_scale_pad_expr(opts: GifOptions) -> List[str]:
    if not opts.width and not opts.height: return []
    apply_ar_preset(opts)
    if not (opts.width and opts.height):
        if opts.width:  return [f"scale=w={opts.width}:h=-1:flags=lanczos"]
        if opts.height: return [f"scale=w=-1:h={opts.height}:flags=lanczos"]
        return []
    W,H = opts.width, opts.height
    if not opts.keep_aspect: return [f"scale={W}:{H}:flags=lanczos"]
    if opts.fit_mode == "fit":
        color = "0x00000000" if (opts.format in ("webp","apng") and opts.keep_alpha) else "black"
        scale = f"scale='iw*min({W}/iw\\,{H}/ih)':'ih*min({W}/iw\\,{H}/ih)':flags=lanczos"
        pad   = f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:{color}"
        return [scale, pad]
    # fill
    scale = f"scale='iw*max({W}/iw\\,{H}/ih)':'ih*max({W}/iw\\,{H}/ih)':flags=lanczos"
    crop  = f"crop={W}:{H}"
    return [scale, crop]

def _palettegen(opts: GifOptions) -> str:
    # Reserve transparent only when >2 colors to avoid ffmpeg error on alpha sources
    reserve = "" if opts.max_colors>2 else ":reserve_transparent=0"
    return f"palettegen=max_colors={opts.max_colors}:stats_mode={opts.stats_mode}{reserve}"

def _gif_filters(opts: GifOptions) -> str:
    fs = []
    if opts.frame_step and int(opts.frame_step)>1:
        fs.append(f"select=not(mod(n\\,{int(opts.frame_step)}))")
        fs.append("setpts=N/FRAME_RATE/TB")
    ce = _build_crop_expr(opts)
    if ce: fs.append(ce)
    fs += _build_scale_pad_expr(opts)
    return ",".join(fs) if fs else "null"

# ---- Command builders ----
def build_gif_cmds(input_path: Path, out_path: Path, opts: GifOptions, ffmpeg: str = "ffmpeg", work_dir: Optional[Path] = None) -> List[List[str]]:
    input_path = Path(input_path); out_path = Path(out_path)
    work_dir = Path(work_dir) if work_dir else out_path.parent
    _ensure_parent(out_path); work_dir.mkdir(parents=True, exist_ok=True)
    filters = _gif_filters(opts)
    if not opts.use_palette:
        vf = f"{filters}" if filters else "null"
        cmd = [ffmpeg, "-y", *_input_trim_args(opts), "-i", str(input_path), "-vf", vf, "-r"]
        cmd += [str(opts.fps or 0)] if opts.fps else []
        cmd += ["-loop","0" if opts.loop==0 else str(opts.loop), str(out_path)]
        return [cmd]
    # With palette
    if opts.two_pass:
        pal = work_dir / (out_path.stem + "_pal.png")
        cmd1 = [ffmpeg, "-y", *_input_trim_args(opts), "-i", str(input_path),
                "-vf", f"{filters},{_palettegen(opts)}", str(pal)]
        dither = "bayer:bayer_scale="+str(max(1,min(5,opts.bayer_scale))) if opts.dither=="bayer" else opts.dither
        paluse = f"paletteuse=dither={dither}"
        cmd2 = [ffmpeg, "-y", "-i", str(input_path), "-i", str(pal),
                "-lavfi", f"{filters} [x]; [x][1:v] {paluse}",
                "-r"]
        cmd2 += [str(opts.fps or 0)] if opts.fps else []
        cmd2 += ["-loop","0" if opts.loop==0 else str(opts.loop), str(out_path)]
        return [cmd1, cmd2]
    # Single pass (split) chain
    dither = "bayer:bayer_scale="+str(max(1,min(5,opts.bayer_scale))) if opts.dither=="bayer" else opts.dither
    paluse = f"paletteuse=dither={dither}"
    vf = f"{filters},split [a][b]; [a] {_palettegen(opts)} [p]; [b][p] {paluse}"
    cmd = [ffmpeg, "-y", *_input_trim_args(opts), "-i", str(input_path),
           "-vf", vf, "-r"]
    cmd += [str(opts.fps or 0)] if opts.fps else []
    cmd += ["-loop","0" if opts.loop==0 else str(opts.loop), str(out_path)]
    return [cmd]

def build_webp_cmd(input_path: Path, out_path: Path, opts: GifOptions, ffmpeg: str = "ffmpeg") -> List[str]:
    cmd = [ffmpeg, "-y", *_input_trim_args(opts), "-i", str(input_path), "-r"]
    cmd += [str(opts.fps or 0)] if opts.fps else []
    vf = _gif_filters(opts); 
    # keep alpha / quality knobs
    if opts.lossless_webp:
        cmd += ["-lossless", "1"]
    else:
        cmd += ["-q:v", str(max(0,min(100,opts.quality_webp)))]
    cmd += ["-compression_level", str(max(0,min(9,opts.compression_webp)))]
    if vf: cmd += ["-vf", vf]
    cmd += [str(out_path)]
    return cmd

def build_apng_cmd(input_path: Path, out_path: Path, opts: GifOptions, ffmpeg: str = "ffmpeg") -> List[str]:
    vf = _gif_filters(opts)
    cmd = [ffmpeg, "-y", *_input_trim_args(opts), "-i", str(input_path)]
    if vf: cmd += ["-vf", vf]
    if opts.fps: cmd += ["-r", str(opts.fps)]
    cmd += [str(out_path)]
    return cmd

def build_commands(input_path: Path, out_path: Path, opts: GifOptions, ffmpeg: str = "ffmpeg", work_dir: Optional[Path] = None) -> List[List[str]]:
    if opts.format == "gif":
        return build_gif_cmds(input_path, out_path, opts, ffmpeg=ffmpeg, work_dir=work_dir)
    if opts.format == "webp":
        return [build_webp_cmd(input_path, out_path, opts, ffmpeg=ffmpeg)]
    if opts.format == "apng":
        return [build_apng_cmd(input_path, out_path, opts, ffmpeg=ffmpeg)]
    raise GifError(f"Unknown format: {opts.format}")

def encode(input_path: Path, out_path: Path, opts: GifOptions, ffmpeg: str = "ffmpeg", work_dir: Optional[Path] = None) -> None:
    _ensure_parent(out_path)
    for c in build_commands(input_path, out_path, opts, ffmpeg=ffmpeg, work_dir=work_dir):
        _ffmpeg_exec(c)

# ---- UI helpers ----
def output_name_for(stem: str, fmt: str, batch: bool=False) -> str:
    f = fmt.lower()
    if f=="gif":  return (f"{stem}.gif" if batch else f"{stem}_gif.gif")
    if f=="webp": return (f"{stem}.webp" if batch else f"{stem}_anim.webp")
    return (f"{stem}.png" if batch else f"{stem}_anim.png")

def _resolve_fps_from_ui(pane, input_path: Path, same_as_video: bool, spinner_val: int) -> Optional[int]:
    try:
        if same_as_video:
            # Use host probe if available
            if hasattr(pane, "probe_media"):
                info = pane.probe_media(input_path)
                f = info.get("fps") if isinstance(info, dict) else None
            else:
                inf = _probe_info(input_path, _ffprobe_from_ffmpeg(getattr(pane, "ffmpeg_path", lambda: "ffmpeg")()))
                f = inf.get("fps")
            return int(round(float(f))) if f else None
        v = int(spinner_val);  return v if v > 0 else None
    except Exception:
        return None

def options_from_ui(pane, input_path: Path, same_as_video: bool, spinner_val: int, batch: bool=False) -> GifOptions:
    fmt = (pane.gif_fmt.currentText() if hasattr(pane,"gif_fmt") else "GIF").strip().lower()
    ar_txt = pane.gif_ar.currentText() if hasattr(pane,"gif_ar") else "None"
    ar_preset = None if ar_txt=="None" else ar_txt
    colors = int(pane.gif_colors.value()) if hasattr(pane,"gif_colors") else 256
    opts = GifOptions(
        format={"gif":"gif","webp":"webp","apng":"apng"}.get(fmt,"gif"),
        fps=_resolve_fps_from_ui(pane, input_path, same_as_video, spinner_val),
        two_pass=bool(pane.gif_two_pass.isChecked()) if fmt=="gif" else False,
        use_palette=bool(getattr(pane,"gif_use_palette",None).isChecked()) if fmt=="gif" and hasattr(pane,"gif_use_palette") else True,
        max_colors=max(2, min(256, colors)),
        stats_mode=pane.gif_stats.currentText() if hasattr(pane,"gif_stats") else "diff",
        dither=pane.gif_dither.currentText() if hasattr(pane,"gif_dither") else "sierra2_4a",
        bayer_scale=max(1, int(pane.gif_bayer.value())) if hasattr(pane,"gif_bayer") else 3,
        loop=int(pane.gif_loop.value()) if hasattr(pane,"gif_loop") else 0,
        width=int(pane.gif_w.value()) or None if hasattr(pane,"gif_w") else None,
        height=int(pane.gif_h.value()) or None if hasattr(pane,"gif_h") else None,
        keep_aspect=bool(pane.gif_keep_ar.isChecked()) if hasattr(pane,"gif_keep_ar") else True,
        fit_mode=("fit" if getattr(pane,"gif_fitmode",None).currentText()=="Fit" else "fill") if hasattr(pane,"gif_fitmode") else "fit",
        ar_preset=ar_preset,
        trim_start=float(pane.gif_trim_start.value()) if hasattr(pane,"gif_trim_start") else 0.0,
        trim_end=(lambda t: (None if (t is None or t.strip()=='' or float(t)<=0) else float(t)))(pane.gif_trim_end.text()) if hasattr(pane,"gif_trim_end") else None,
        frame_step=max(1, int(pane.gif_frame_step.value())) if hasattr(pane,"gif_frame_step") else 1,
        boomerang=bool(pane.gif_boomer.isChecked()) if hasattr(pane,"gif_boomer") else False,
        keep_alpha=bool(pane.gif_keep_alpha.isChecked()) if hasattr(pane,"gif_keep_alpha") else True,
        lossless_webp=bool(pane.gif_webp_lossless.isChecked()) if hasattr(pane,"gif_webp_lossless") else False,
        quality_webp=int(pane.gif_webp_quality.value()) if hasattr(pane,"gif_webp_quality") else 80,
        compression_webp=int(pane.gif_webp_comp.value()) if hasattr(pane,"gif_webp_comp") else 6,
    )
    if opts.format=="gif" and opts.boomerang: opts.two_pass=False
    if batch: opts.two_pass=False
    return opts

# ---- Presets (saved/loaded by tools_tab, but logic lives here) ----
def preset_from_ui(pane, input_path: Path, same_as_video: bool, spinner_val: int) -> Dict[str, Any]:
    """Return a JSON-serializable dict for presets."""
    opts = options_from_ui(pane, input_path, same_as_video, spinner_val)
    data = {"tool":"gif","same_as_video": bool(same_as_video),"fps": spinner_val, "options": opts.to_dict()}
    return data

def apply_preset(pane, data: Dict[str, Any]) -> None:
    """Apply a preset dict created by preset_from_ui()."""
    try:
        opts = GifOptions.from_dict(data.get("options", {}))
    except Exception:
        opts = GifOptions()
    # fps area (host controls)
    try:
        same = bool(data.get("same_as_video", False)); pane.gif_same.setChecked(same)
        if not same:
            val = int(data.get("fps", opts.fps or 12)); 
            pane.gif_fps.setValue(val); pane.spin_gif_fps.setValue(val)
    except Exception: pass
    # core
    mapping = [
        ("gif_fmt", {"gif":"GIF","webp":"WebP","apng":"APNG"}.get(opts.format,"GIF")),
        ("gif_two_pass", bool(opts.two_pass)),
        ("gif_use_palette", bool(opts.use_palette)),
        ("gif_loop", int(opts.loop)),
        ("gif_colors", int(max(2, min(256, opts.max_colors)))),
        ("gif_stats", "diff" if opts.stats_mode not in ("single","diff") else opts.stats_mode),
        ("gif_dither", opts.dither),
        ("gif_bayer", max(1, min(5, int(opts.bayer_scale)))),
        ("gif_keep_ar", bool(opts.keep_aspect)),
        ("gif_fitmode", "Fit" if opts.fit_mode=="fit" else "Fill"),
        ("gif_ar", opts.ar_preset if opts.ar_preset else "None"),
        ("gif_w", int(opts.width or 0)),
        ("gif_h", int(opts.height or 0)),
        ("gif_trim_start", float(opts.trim_start or 0.0)),
        ("gif_trim_end", "" if opts.trim_end in (None,0) else str(opts.trim_end)),
        ("gif_frame_step", int(opts.frame_step or 1)),
        ("gif_boomer", bool(opts.boomerang)),
        ("gif_keep_alpha", bool(opts.keep_alpha)),
        ("gif_webp_lossless", bool(opts.lossless_webp)),
        ("gif_webp_quality", int(opts.quality_webp)),
        ("gif_webp_comp", int(opts.compression_webp)),
    ]
    for name, val in mapping:
        w = getattr(pane, name, None)
        try:
            if w is None: continue
            if hasattr(w, "setCurrentText"): w.setCurrentText(str(val))
            elif hasattr(w, "setChecked"): w.setChecked(bool(val))
            elif hasattr(w, "setValue"): w.setValue(val)
            elif name=="gif_trim_end" and hasattr(w, "setText"): w.setText(str(val))
        except Exception: pass
    try:
        # trigger enable/disable refresh
        if hasattr(pane, "gif_fmt"): pane.gif_fmt.currentTextChanged.emit(pane.gif_fmt.currentText())
        if hasattr(pane, "gif_dither"): pane.gif_dither.currentTextChanged.emit(pane.gif_dither.currentText())
        if hasattr(pane, "gif_use_palette"): pane.gif_use_palette.toggled.emit(pane.gif_use_palette.isChecked())
    except Exception: 
        pass

# ---- UI factory ----
def install_ui(pane, lay_gif, sec_gif) -> None:
    """Build the advanced controls and a tiny result/progress line."""
    try:
        from PySide6.QtWidgets import (QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QLabel, QHBoxLayout, QProgressBar, QPushButton)
    except Exception:
        # PySide6 not available; skip UI build to avoid import-time crash
        return
    # Controls + defaults
    pane.gif_fmt = QComboBox(); pane.gif_fmt.addItems(["GIF","WebP","APNG"]); pane.gif_fmt.setCurrentText("GIF")
    pane.gif_two_pass = QCheckBox("Two-pass palette (best)"); pane.gif_two_pass.setChecked(True); pane.gif_two_pass.setToolTip(TOOLTIPS["two_pass"])
    pane.gif_boomer = QCheckBox("Boomerang"); pane.gif_boomer.setToolTip(TOOLTIPS["boomerang"])
    pane.gif_use_palette = QCheckBox("Use palette (GIF quality)"); pane.gif_use_palette.setChecked(True)
    pane.gif_loop = QSpinBox(); pane.gif_loop.setRange(0,1000); pane.gif_loop.setValue(0); pane.gif_loop.setToolTip(TOOLTIPS["loop"])
    pane.gif_colors = QSpinBox(); pane.gif_colors.setRange(2,256); pane.gif_colors.setValue(256); pane.gif_colors.setToolTip(TOOLTIPS["max_colors"])
    pane.gif_stats = QComboBox(); pane.gif_stats.addItems(["single","diff"]); pane.gif_stats.setCurrentText("diff"); pane.gif_stats.setToolTip(TOOLTIPS["stats_mode"])
    pane.gif_dither = QComboBox(); pane.gif_dither.addItems(["sierra2_4a","bayer","floyd_steinberg","none"]); pane.gif_dither.setToolTip(TOOLTIPS["dither"])
    pane.gif_bayer = QSpinBox(); pane.gif_bayer.setRange(1,5); pane.gif_bayer.setValue(3)
    pane.gif_keep_ar = QCheckBox("Keep aspect"); pane.gif_keep_ar.setChecked(True); pane.gif_keep_ar.setToolTip(TOOLTIPS["keep_aspect"])
    pane.gif_fitmode = QComboBox(); pane.gif_fitmode.addItems(["Fit","Fill"]); pane.gif_fitmode.setCurrentText("Fit"); pane.gif_fitmode.setToolTip(TOOLTIPS["fit_mode"])
    pane.gif_ar = QComboBox(); pane.gif_ar.addItems(["None","16:9","9:16","4:3","1:1"]); pane.gif_ar.setCurrentText("None"); pane.gif_ar.setToolTip(TOOLTIPS["ar_preset"])
    pane.gif_w = QSpinBox(); pane.gif_w.setRange(0,8192); pane.gif_w.setValue(0)
    pane.gif_h = QSpinBox(); pane.gif_h.setRange(0,8192); pane.gif_h.setValue(0)
    pane.gif_trim_start = QDoubleSpinBox(); pane.gif_trim_start.setRange(0.0,1e7); pane.gif_trim_start.setDecimals(3); pane.gif_trim_start.setValue(0.0); pane.gif_trim_start.setToolTip(TOOLTIPS["trim"])
    pane.gif_trim_end = QLineEdit(); pane.gif_trim_end.setPlaceholderText("blank or 0 = to end"); pane.gif_trim_end.setToolTip(TOOLTIPS["trim"])
    pane.gif_frame_step = QSpinBox(); pane.gif_frame_step.setRange(1,100); pane.gif_frame_step.setValue(1); pane.gif_frame_step.setToolTip(TOOLTIPS["frame_step"])
    pane.gif_keep_alpha = QCheckBox("Keep alpha"); pane.gif_keep_alpha.setChecked(True); pane.gif_keep_alpha.setToolTip(TOOLTIPS["keep_alpha"])
    pane.gif_webp_lossless = QCheckBox("WebP lossless"); pane.gif_webp_lossless.setChecked(False)
    pane.gif_webp_quality = QSpinBox(); pane.gif_webp_quality.setRange(0,100); pane.gif_webp_quality.setValue(80)
    pane.gif_webp_comp = QSpinBox(); pane.gif_webp_comp.setRange(0,9); pane.gif_webp_comp.setValue(6)

    # Enable/disable rules
    def _sync_two_pass():
        if pane.gif_fmt.currentText()=="GIF" and (pane.gif_boomer.isChecked() or not pane.gif_use_palette.isChecked()):
            pane.gif_two_pass.setChecked(False); pane.gif_two_pass.setEnabled(False)
        else:
            pane.gif_two_pass.setEnabled(pane.gif_fmt.currentText()=="GIF")
    def _update_en():
        fmt = pane.gif_fmt.currentText()
        is_gif = (fmt=="GIF"); is_webp = (fmt=="WebP")
        use_pal = (pane.gif_use_palette.isChecked() if is_gif else False)
        pane.gif_bayer.setEnabled(pane.gif_dither.currentText()=="bayer")
        for w in (pane.gif_two_pass, pane.gif_colors, pane.gif_stats, pane.gif_dither, pane.gif_bayer):
            w.setEnabled(is_gif and use_pal and (w is not pane.gif_bayer or pane.gif_dither.currentText()=="bayer"))
        pane.gif_keep_alpha.setEnabled(not is_gif)
        pane.gif_webp_lossless.setEnabled(is_webp)
        pane.gif_webp_quality.setEnabled(is_webp and not pane.gif_webp_lossless.isChecked())
        pane.gif_webp_comp.setEnabled(is_webp)
        _sync_two_pass()
    pane.gif_fmt.currentTextChanged.connect(lambda _=None: _update_en())
    pane.gif_dither.currentTextChanged.connect(lambda _=None: _update_en())
    pane.gif_webp_lossless.toggled.connect(lambda _=None: _update_en())
    pane.gif_use_palette.toggled.connect(lambda _=None: _update_en())
    pane.gif_boomer.toggled.connect(lambda _=None: _sync_two_pass())
    _update_en()

    # Layout rows
    row_fmt = QHBoxLayout(); row_fmt.addWidget(QLabel("Format")); row_fmt.addWidget(pane.gif_fmt); row_fmt.addStretch(1)
    row_pal = QHBoxLayout(); row_pal.addWidget(pane.gif_two_pass); row_pal.addWidget(pane.gif_boomer); row_pal.addWidget(pane.gif_use_palette); row_pal.addStretch(1)
    row_loop = QHBoxLayout(); row_loop.addWidget(QLabel("Loop (0=∞)")); row_loop.addWidget(pane.gif_loop); row_loop.addStretch(1)
    row_cols = QHBoxLayout(); row_cols.addWidget(QLabel("Max colors")); row_cols.addWidget(pane.gif_colors); row_cols.addWidget(QLabel("Stats")); row_cols.addWidget(pane.gif_stats); row_cols.addStretch(1)
    row_dith = QHBoxLayout(); row_dith.addWidget(QLabel("Dither")); row_dith.addWidget(pane.gif_dither); row_dith.addWidget(QLabel("Bayer 1–5")); row_dith.addWidget(pane.gif_bayer); row_dith.addStretch(1)
    row_sz = QHBoxLayout(); row_sz.addWidget(QLabel("W")); row_sz.addWidget(pane.gif_w); row_sz.addWidget(QLabel("H")); row_sz.addWidget(pane.gif_h); row_sz.addWidget(pane.gif_keep_ar); row_sz.addStretch(1)
    row_fit = QHBoxLayout(); row_fit.addWidget(QLabel("Mode")); row_fit.addWidget(pane.gif_fitmode); row_fit.addWidget(QLabel("AR")); row_fit.addWidget(pane.gif_ar); row_fit.addStretch(1)
    row_trim = QHBoxLayout(); row_trim.addWidget(QLabel("Trim start (s)")); row_trim.addWidget(pane.gif_trim_start); row_trim.addWidget(QLabel("Trim end (s)")); row_trim.addWidget(pane.gif_trim_end); row_trim.addStretch(1)
    row_step = QHBoxLayout(); row_step.addWidget(QLabel("Frame step")); row_step.addWidget(pane.gif_frame_step); row_step.addStretch(1)

    lay_gif.addRow(row_fmt); lay_gif.addRow(row_pal); lay_gif.addRow(row_loop); lay_gif.addRow(row_cols)
    lay_gif.addRow(row_dith); lay_gif.addRow(row_sz); lay_gif.addRow(row_fit); lay_gif.addRow(row_trim); lay_gif.addRow(row_step)

    # WebP/APNG line
    row_w = QHBoxLayout(); row_w.addWidget(pane.gif_keep_alpha); row_w.addWidget(pane.gif_webp_lossless)
    row_w.addWidget(QLabel("quality")); row_w.addWidget(pane.gif_webp_quality)
    row_w.addWidget(QLabel("compression")); row_w.addWidget(pane.gif_webp_comp); row_w.addStretch(1)
    lay_gif.addRow(row_w)

    # Progress + result
    pane.gif_pb = QProgressBar(); pane.gif_pb.setRange(0,100); pane.gif_pb.setVisible(False)
    pane.gif_eta = QLabel(""); pane.gif_eta.setVisible(False)
    pane.gif_btn_play = QPushButton("Play last"); pane.gif_btn_play.setEnabled(False)
    def _play_last():
        try:
            p = getattr(pane, "_gif_last_out", None)
            if p and getattr(pane, "main", None) and getattr(pane.main, "video", None):
                pane.main.video.open(str(p))
        except Exception:
            pass
    pane.gif_btn_play.clicked.connect(_play_last)
    row_res = QHBoxLayout(); row_res.addWidget(pane.gif_pb); row_res.addWidget(pane.gif_eta); row_res.addWidget(pane.gif_btn_play); row_res.addStretch(1)
    lay_gif.addRow(row_res)

# ---- Progress encode (GUI) ----
def encode_with_progress(pane, input_path: Path, out_path: Path, opts: GifOptions, ffmpeg: str = "ffmpeg", work_dir: Optional[Path] = None) -> None:
    """Spawn ffmpeg with -progress and show ETA like interp.py. Falls back to encode() on error."""
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QMessageBox
    except Exception:
        return encode(input_path, out_path, opts, ffmpeg=ffmpeg, work_dir=work_dir)

    cmds = build_commands(input_path, out_path, opts, ffmpeg=ffmpeg, work_dir=work_dir)
    ffprobe = _ffprobe_from_ffmpeg(ffmpeg)
    info = _probe_info(input_path, ffprobe)
    src_fps = (info.get("fps") or 0.0) or 25.0
    use_fps = float(opts.fps or src_fps)
    step = max(1, int(opts.frame_step or 1))
    # estimate frames within trim
    duration = info.get("duration") or 0.0
    if opts.trim_start: duration = max(0.0, duration - float(opts.trim_start))
    if opts.trim_end and float(opts.trim_end)>0: duration = min(duration, float(opts.trim_end) - float(opts.trim_start or 0.0))
    est_frames = max(1, int(duration * use_fps / step))
    if opts.boomerang and opts.format=="gif": est_frames *= 2
    # progress infra
    progress_file = Path(work_dir or out_path.parent) / "_gif_progress.txt"
    try:
        progress_file.unlink(missing_ok=True)
    except Exception:
        pass

    # Inject -progress into each cmd
    def _inject(cmd: List[str]) -> List[str]:
        return [cmd[0], "-hide_banner", "-nostats", "-progress", str(progress_file)] + cmd[1:]

    proc = None
    pane.gif_pb.setVisible(True); pane.gif_eta.setVisible(True); pane.gif_pb.setValue(0); pane.gif_eta.setText("Starting…")
    start = time.time()

    def _parse():
        try:
            txt = progress_file.read_text('utf-8', errors='ignore')
        except Exception:
            return
        done = 0
        for ln in txt.splitlines()[::-1]:
            if ln.startswith("frame="):
                try:
                    done = int(ln.split("=")[1].strip())
                    break
                except Exception: pass
            if ln.startswith("out_time_ms="):
                try:
                    ms = int(ln.split("=")[1]); done = int((ms/1000.0) * use_fps / step)
                    break
                except Exception: pass
        done = max(0, min(est_frames, done))
        pct = int(round(100.0 * done / max(1, est_frames)))
        pane.gif_pb.setValue(pct)
        elapsed = time.time() - start
        if done>0 and pct<100:
            total_est = elapsed * est_frames / max(1, done)
            eta = max(0, int(total_est - elapsed))
            pane.gif_eta.setText(f"{pct}% — ETA {eta}s")
        else:
            pane.gif_eta.setText(f"{pct}%")

    timer = QTimer(pane); timer.setInterval(300)
    def _tick(): 
        _parse()
        # completed?
        if proc is None: return
        ret = proc.poll()
        if ret is None: return
        timer.stop()
        pane.gif_pb.setVisible(False); pane.gif_eta.setVisible(False)
        if ret != 0:
            try:
                QMessageBox.critical(pane, "FFmpeg error", f"Command failed (code {ret}).")
            except Exception: pass
            return
        # success
        pane._gif_last_out = Path(out_path)
        try: pane.gif_btn_play.setEnabled(True)
        except Exception: pass
        try:
            if getattr(pane, "cb_autoplay_gif", None) and pane.cb_autoplay_gif.isChecked():
                if getattr(pane, "main", None) and getattr(pane.main, "video", None):
                    pane.main.video.open(str(out_path))
        except Exception: pass

    timer.timeout.connect(_tick); timer.start()
    # Run
    try:
        for i, c in enumerate(cmds):
            proc = subprocess.Popen(_inject(c), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            while True:
                _tick()
                if proc.poll() is not None: break
                time.sleep(0.1)
            if proc.returncode != 0: break
    finally:
        try: progress_file.unlink(missing_ok=True)
        except Exception: pass

# ---- Misc ----
def output_ext_for(fmt: str) -> str:
    f = fmt.lower()
    return { "gif":".gif", "webp":".webp", "apng":".png" }.get(f, ".gif")
