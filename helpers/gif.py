
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
    # Per-image duration for images→animation builds (seconds); overrides Images FPS when set
    img_duration_sec: Optional[float] = None

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
    if (getattr(opts, 'img_duration_sec', None) in (None, 0)) and opts.fps:
            cmd += ["-r", str(opts.fps)]
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
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
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

    # New compact top row: Format + Loop + Frame step + FPS (compact)
    row_top = QHBoxLayout()
    row_top.addWidget(QLabel("Format")); row_top.addWidget(pane.gif_fmt)
    row_top.addSpacing(12); row_top.addWidget(QLabel("Loop (0=∞)")); row_top.addWidget(pane.gif_loop)
    row_top.addSpacing(12); row_top.addWidget(QLabel("Frame step")); row_top.addWidget(pane.gif_frame_step)
    # Move the existing FPS spin from the host row into here; make it compact
    try:
        # Hide the original FPS row's label and spin (keep them for backend reads)
        lbl_fps = lay_gif.labelForField(pane.spin_gif_fps)
        if lbl_fps:
            lbl_fps.hide()
        try:
            pane.spin_gif_fps.setVisible(False)
        except Exception:
            pass
        # Compact inline FPS spin that mirrors the original
        pane.spin_gif_fps_inline = QSpinBox()
        try:
            pane.spin_gif_fps_inline.setRange(pane.spin_gif_fps.minimum(), pane.spin_gif_fps.maximum())
            pane.spin_gif_fps_inline.setValue(pane.spin_gif_fps.value())
        except Exception:
            pane.spin_gif_fps_inline.setRange(1, 120); pane.spin_gif_fps_inline.setValue(12)
        pane.spin_gif_fps_inline.setMaximumWidth(60)
        # Two-way sync without loops
        def _sync_inline_to_orig(v):
            try:
                if pane.spin_gif_fps.value() != int(v):
                    old = pane.spin_gif_fps.blockSignals(True)
                    pane.spin_gif_fps.setValue(int(v))
                    pane.spin_gif_fps.blockSignals(old)
            except Exception:
                pass
        def _sync_orig_to_inline(v):
            try:
                if pane.spin_gif_fps_inline.value() != int(v):
                    old = pane.spin_gif_fps_inline.blockSignals(True)
                    pane.spin_gif_fps_inline.setValue(int(v))
                    pane.spin_gif_fps_inline.blockSignals(old)
            except Exception:
                pass
        try:
            pane.spin_gif_fps_inline.valueChanged.connect(_sync_inline_to_orig)
            pane.spin_gif_fps.valueChanged.connect(_sync_orig_to_inline)
        except Exception:
            pass
        row_top.addSpacing(12); row_top.addWidget(QLabel("FPS")); row_top.addWidget(pane.spin_gif_fps_inline)
    except Exception:
        pass
        row_sz = QHBoxLayout(); row_sz.addWidget(QLabel("W")); row_sz.addWidget(pane.gif_w); row_sz.addWidget(QLabel("H")); row_sz.addWidget(pane.gif_h); row_sz.addWidget(pane.gif_keep_ar); row_sz.addStretch(1)
    row_fit = QHBoxLayout(); row_fit.addWidget(QLabel("Mode")); row_fit.addWidget(pane.gif_fitmode); row_fit.addWidget(QLabel("AR")); row_fit.addWidget(pane.gif_ar); row_fit.addStretch(1)
    # Combined size + mode row
    row_sizefit = QHBoxLayout()
    row_sizefit.addWidget(QLabel("W")); row_sizefit.addWidget(pane.gif_w)
    row_sizefit.addWidget(QLabel("H")); row_sizefit.addWidget(pane.gif_h)
    row_sizefit.addWidget(pane.gif_keep_ar)
    row_sizefit.addSpacing(12); row_sizefit.addWidget(QLabel("Mode")); row_sizefit.addWidget(pane.gif_fitmode)
    row_sizefit.addSpacing(12); pane.lbl_gif_ar = QLabel("AR"); row_sizefit.addWidget(pane.lbl_gif_ar); row_sizefit.addWidget(pane.gif_ar)
    row_sizefit.addStretch(1)
    row_trim = QHBoxLayout(); row_trim.addWidget(QLabel("Trim start (s)")); row_trim.addWidget(pane.gif_trim_start); row_trim.addWidget(QLabel("Trim end (s)")); row_trim.addWidget(pane.gif_trim_end); row_trim.addStretch(1)
    row_step = QHBoxLayout(); row_step.addWidget(QLabel("Frame step")); row_step.addWidget(pane.gif_frame_step); row_step.addStretch(1)

    lay_gif.addRow(row_top)
    pane._gif_adv_rows = getattr(pane, '_gif_adv_rows', [])
    pane._gif_adv_rows.append(row_pal)
    pane._gif_adv_rows.append(row_cols)
    pane._gif_adv_rows.append(row_dith)
    lay_gif.addRow(row_sizefit)
    lay_gif.addRow(row_trim)

    # WebP/APNG line
    row_w = QHBoxLayout(); row_w.addWidget(pane.gif_keep_alpha); row_w.addWidget(pane.gif_webp_lossless)
    row_w.addWidget(QLabel("quality")); row_w.addWidget(pane.gif_webp_quality)
    row_w.addWidget(QLabel("compression")); row_w.addWidget(pane.gif_webp_comp); row_w.addStretch(1)
    pane._gif_adv_rows = getattr(pane, '_gif_adv_rows', []); pane._gif_adv_rows.append(row_w)

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


    # --- Create new from images (single animation) ---
    try:
        from PySide6.QtWidgets import (QGroupBox, QFormLayout, QFileDialog, QLineEdit, QSpinBox, QLabel, QHBoxLayout, QPushButton, QCheckBox, QToolButton, QWidget, QVBoxLayout, QMessageBox, QTableWidget, QTableWidgetItem, QAbstractItemView)
        from PySide6.QtCore import QSettings, Qt
    except Exception:
        QGroupBox = None
    if QGroupBox is not None:
        grp_imgs = QGroupBox("Create new from images")
        grp_imgs.setStyleSheet("QGroupBox{background: palette(base);} QGroupBox:title{subcontrol-origin: margin; left: 6px; padding: 2px 4px;}")
        lay_imgs = QFormLayout(grp_imgs)

        
        # Header row with 'Include subfolders' next to the title
        header_row = QHBoxLayout()
        pane.gif_imgs_subdirs = QCheckBox("Include subfolders")
        header_row.addStretch(1)
        header_row.addWidget(pane.gif_imgs_subdirs)
        lay_imgs.addRow(header_row)
        pane.gif_imgs_dir = QLineEdit(); pane.gif_imgs_dir.setReadOnly(True)
        pane.gif_imgs_browse = QPushButton("Choose folder…")
        pane.gif_imgs_fps = QSpinBox(); pane.gif_imgs_fps.setRange(1, 120); pane.gif_imgs_fps.setValue(12)

        row_i1 = QHBoxLayout()
        row_i1.addWidget(QLabel("Folder")); row_i1.addWidget(pane.gif_imgs_dir); row_i1.addWidget(pane.gif_imgs_browse)
        row_i2 = QHBoxLayout()
        row_i2.addWidget(QLabel("Images FPS")); row_i2.addWidget(pane.gif_imgs_fps)
        pane.gif_img_sec = QDoubleSpinBox(); pane.gif_img_sec.setRange(0.1, 10.0); pane.gif_img_sec.setSingleStep(0.1); pane.gif_img_sec.setValue(1.0)
        row_i2.addSpacing(12); row_i2.addWidget(QLabel("Seconds/image")); row_i2.addWidget(pane.gif_img_sec); row_i2.addStretch(1)
        def _sync_img_fps_enable(val):
            try:
                pane.gif_imgs_fps.setEnabled(False if float(val) > 0 else True)
            except Exception:
                pass
        try:
            pane.gif_img_sec.valueChanged.connect(_sync_img_fps_enable)
        except Exception:
            pass
        _sync_img_fps_enable(pane.gif_img_sec.value())

        try:
            pane.gif_imgs_preview = QTableWidget(0, 3)
            pane.gif_imgs_preview.setHorizontalHeaderLabels(["Name", "Size", "Resolution"])
            pane.gif_imgs_preview.verticalHeader().setVisible(False)
            from PySide6.QtWidgets import QAbstractItemView
            pane.gif_imgs_preview.setSelectionBehavior(QAbstractItemView.SelectRows)
            pane.gif_imgs_preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
            hh = pane.gif_imgs_preview.horizontalHeader(); hh.setStretchLastSection(True)
            pane.gif_imgs_preview.setMinimumHeight(110)
            pane.gif_imgs_preview.setMaximumHeight(140)
            try:
                pane.gif_imgs_preview.setAlternatingRowColors(False)
                _qss = ("QTableWidget { background-color: palette(base); color: palette(text); gridline-color: palette(mid); } QHeaderView::section { background-color: palette(window); color: palette(window-text); padding: 4px; border: 0px; } QTableWidget::item:selected { background: palette(highlight); color: palette(highlighted-text); } QTableWidget::item:selected:!active { background: palette(midlight); color: palette(window-text); }")
                pane.gif_imgs_preview.setStyleSheet(_qss)
            except Exception:
                pass
        except Exception:
            pass

        pane.gif_imgs_create = QPushButton("New")
        pane.gif_imgs_create.setStyleSheet("QPushButton{background: palette(button); color: palette(window-text); border: 1px solid palette(mid); border-radius: 8px; padding: 8px 14px;} QPushButton:hover{background: palette(mid);} QPushButton:pressed{background: palette(dark);}")


        lay_imgs.addRow(row_i1)
        lay_imgs.addRow(row_i2)
        try:
            lay_imgs.addRow(pane.gif_imgs_preview)
        except Exception:
            pass
        pane.gif_open_after = QCheckBox("Open when done"); pane.gif_open_after.setChecked(True)
        lay_imgs.addRow(pane.gif_open_after)
        lay_imgs.addRow(pane.gif_imgs_create)
        lay_gif.addRow(grp_imgs)

        # --- Advanced (images) — collapsed by default with tiny toggle ---
        # We implement a small toggle button that shows/hides the inner layout.
        adv_wrap = QWidget(); adv_v = QVBoxLayout(adv_wrap); adv_v.setContentsMargins(0,0,0,0); adv_v.setSpacing(6)
        pane.gif_adv_toggle = QToolButton(); pane.gif_adv_toggle.setText("Advanced settings"); pane.gif_adv_toggle.setCheckable(True); pane.gif_adv_toggle.setChecked(False)
        pane.gif_adv_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        pane.gif_adv_toggle.setArrowType(Qt.RightArrow)

        adv_inner = QWidget(); adv_lay = QFormLayout(adv_inner)
        pane.gif_imgs_patterns = QLineEdit("*.png;*.jpg;*.jpeg;*.webp")

        adv_lay.addRow(QLabel("File patterns (;-separated)"), pane.gif_imgs_patterns)
        adv_inner.setVisible(False)

        def _adv_toggled(on):
            pane.gif_adv_toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
            adv_inner.setVisible(bool(on))
        pane.gif_adv_toggle.toggled.connect(_adv_toggled)

        adv_v.addWidget(pane.gif_adv_toggle); adv_v.addWidget(adv_inner)
        # Move advanced rows (palette/dither/webp options) into this group
        try:
            for r in getattr(pane, '_gif_adv_rows', []):
                adv_lay.addRow(r)
            pane._gif_adv_rows = []
        except Exception:
            pass
        lay_gif.addRow(adv_wrap)

        # Remember last-used folder
        def _choose_source():
            try:
                settings = QSettings("FrameVision", "FrameVision")
                start = settings.value("tools/export_gif/images_last_dir", "")
            except Exception:
                settings = None; start = ""
            msg = QMessageBox(pane)
            msg.setWindowTitle("Create from images")
            msg.setText("Select image source:")
            b_files = msg.addButton("Choose files", QMessageBox.AcceptRole)
            b_folder = msg.addButton("Choose folder", QMessageBox.ActionRole)
            msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.exec()
            if msg.clickedButton() == b_files:
                pats = pane.gif_imgs_patterns.text().strip() if hasattr(pane, 'gif_imgs_patterns') else "*.png;*.jpg;*.jpeg;*.webp"
                exts = ' '.join([p.strip() for p in pats.split(';') if p.strip()])
                flt = f"Images ({exts})"
                files, _ = QFileDialog.getOpenFileNames(pane, "Choose images", start or str(Path.home()), flt)
                if files:
                    pane.gif_imgs_files = [Path(f) for f in files]
                    try:
                        common = os.path.commonpath(files)
                    except Exception:
                        common = str(Path(files[0]).parent)
                    pane.gif_imgs_dir.setText(common)
                    try:
                        if settings: settings.setValue("tools/export_gif/images_last_dir", common)
                    except Exception:
                        pass
                    _update_preview()
                return
            elif msg.clickedButton() == b_folder:
                folder = QFileDialog.getExistingDirectory(pane, "Choose images folder", start or str(Path.home()))
                if folder:
                    try:
                        pane.gif_imgs_files = []
                    except Exception:
                        pass
                    pane.gif_imgs_dir.setText(folder)
                    try:
                        if settings: settings.setValue("tools/export_gif/images_last_dir", folder)
                    except Exception:
                        pass
                    _update_preview()
                return
            else:
                return
        pane.gif_imgs_browse.clicked.connect(_choose_source)

        def _update_preview():
            try:
                table = getattr(pane, 'gif_imgs_preview', None)
                if table is None:
                    return
                explicit = getattr(pane, 'gif_imgs_files', [])
                if explicit:
                    imgs = [Path(f) for f in explicit]
                else:
                    folder_txt = pane.gif_imgs_dir.text().strip()
                    if folder_txt:
                        folder = Path(folder_txt)
                    else:
                        folder = None
                    if folder and folder.exists():
                        imgs = _gather_images(folder, pane.gif_imgs_patterns.text().strip(), bool(pane.gif_imgs_subdirs.isChecked()))
                    else:
                        imgs = []
                table.setRowCount(len(imgs))
                ffprobe = _ffprobe_from_ffmpeg(getattr(pane, 'ffmpeg_path', lambda: 'ffmpeg')())
                for i, p in enumerate(imgs):
                    try:
                        name = p.name
                        try:
                            size = _human_size(p.stat().st_size)
                        except Exception:
                            size = "-"
                        res = _fast_image_resolution(p, ffprobe)
                        res_txt = (f"{res[0]}×{res[1]}" if res else "—×—")
                        table.setItem(i, 0, QTableWidgetItem(name))
                        table.setItem(i, 1, QTableWidgetItem(size))
                        table.setItem(i, 2, QTableWidgetItem(res_txt))
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            pane.gif_imgs_subdirs.toggled.connect(lambda _=None: _update_preview())
            pane.gif_imgs_patterns.textChanged.connect(lambda _=None: _update_preview())
        except Exception:
            pass

        # Action: build one animation from the selected folder
        def _do_create():
            folder = pane.gif_imgs_dir.text().strip()
            if not folder:
                try:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(pane, "No folder", "Please choose an images folder first.")
                except Exception:
                    pass
                return
            fmt = pane.gif_fmt.currentText().strip().lower()
            # Build options from current UI (palette, resize, etc), FPS = images FPS
            dummy = Path(folder)
            opts = options_from_ui(pane, dummy, same_as_video=False, spinner_val=int(pane.gif_imgs_fps.value()), batch=False)
            try:
                dur = float(pane.gif_img_sec.value())
                if dur > 0:
                    opts.img_duration_sec = dur
            except Exception:
                pass
            # Build default name based on selected folder or explicit files
            explicit = getattr(pane, 'gif_imgs_files', [])
            base_name = Path(folder).name
            if explicit:
                try:
                    common = str(Path(explicit[0]).parent)
                    base_name = Path(common).name
                except Exception:
                    pass
            # Determine extension from format
            ext = 'gif' if fmt == 'gif' else ('webp' if fmt == 'webp' else 'png')
            # Default directory
            try:
                out_dir_default = Path(getattr(pane, 'OUT_VIDEOS', getattr(pane, 'out_dir', Path.cwd())))
            except Exception:
                out_dir_default = Path.cwd()
            # Show Save dialog
            suggested = str(out_dir_default / f"{base_name}.{ext}")
            filt = f"Animation (*.{ext})"
            save_path, _ = QFileDialog.getSaveFileName(pane, 'Save animation', suggested, filt)
            if not save_path:
                try:
                    pane.gif_imgs_create.setEnabled(True)
                except Exception:
                    pass
                return
            # Ensure extension
            sp = Path(save_path)
            if sp.suffix.lower() != f'.{ext}':
                sp = sp.with_suffix(f'.{ext}')
            try:
                sp.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            # Remember last save dir
            try:
                settings = QSettings('FrameVision', 'FrameVision')
                settings.setValue('tools/export_gif/last_save_dir', str(sp.parent))
            except Exception:
                pass
            out = sp

            encode_images_with_progress(
                pane,
                Path(folder),
                out,
                opts,
                patterns=pane.gif_imgs_patterns.text().strip(),
                include_subdirs=bool(pane.gif_imgs_subdirs.isChecked()),
                ffmpeg=getattr(pane, 'ffmpeg_path', lambda: 'ffmpeg')(),
                work_dir=Path(getattr(pane, 'OUT_TEMP', out.parent)),
                explicit_imgs=list(explicit) if explicit else None,
            )
            # Open folder after export
            try:
                (
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out.parent)))
                if (getattr(pane, 'gif_open_after', None) is not None and bool(pane.gif_open_after.isChecked()))
                else None
            )
            except Exception:
                pass
            try:
                pane._gif_last_out = out
                pane.gif_btn_play.setEnabled(True)
            except Exception:
                pass

        pane.gif_imgs_create.clicked.connect(_do_create)
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




# ---- File/metadata helpers ----
def _human_size(num_bytes: int) -> str:
    try:
        num = float(num_bytes)
    except Exception:
        return "-"
    units = ["B","KB","MB","GB","TB"]
    i = 0
    while num >= 1024 and i < len(units)-1:
        num /= 1024.0
        i += 1
    return f"{num:.1f} {units[i]}" if i>0 else f"{int(num)} {units[i]}"

def _fast_image_resolution(path: Path, ffprobe: str):
    # Try PIL (fast header read); fallback to ffprobe; else None
    try:
        from PIL import Image
        try:
            with Image.open(str(path)) as im:
                w, h = im.size
                return (int(w), int(h))
        except Exception:
            pass
    except Exception:
        pass
    try:
        out = subprocess.check_output([ffprobe, "-v","error","-select_streams","v:0",
                                       "-show_entries","stream=width,height",
                                       "-of","csv=s=x:p=0", str(path)],
                                      universal_newlines=True)
        line = out.strip().splitlines()[0] if out.strip() else ""
        if "x" in line:
            w,h = line.split("x",1)
            return (int(w), int(h))
    except Exception:
        pass
    return None

# ---- Image-sequence helpers ----
def _nat_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _gather_images(folder: Path, patterns: str, include_subdirs: bool) -> List[Path]:
    pats = [p.strip() for p in patterns.split(";") if p.strip()]
    files: List[Path] = []
    for pat in pats:
        it = folder.rglob(pat) if include_subdirs else folder.glob(pat)
        files.extend(it)
    files = [Path(f) for f in files if Path(f).is_file()]
    files.sort(key=lambda x: _nat_key(str(x)))
    return files

def _write_concat_list(imgs: List[Path], fps: float, boomerang: bool, work_dir: Path, img_duration_sec: Optional[float] = None) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    lst = work_dir / "img_sequence.txt"
    dur = (float(img_duration_sec) if (img_duration_sec is not None and float(img_duration_sec) > 0) else (1.0 / max(0.1, float(fps))))
    seq = list(imgs)
    if boomerang and len(imgs) > 1:
        seq = imgs + imgs[-2:0:-1]  # forward + reverse (no duplicate ends)
    with lst.open("w", encoding="utf-8") as f:
        for p in seq:
            sp = str(p).replace('\\','/')
            sp = sp.replace("'", "'\\''")
            f.write("file '" + sp + "'\n")
            f.write(f"duration {dur}\n")
        if seq:
            sp2 = str(seq[-1]).replace('\\','/')
            sp2 = sp2.replace("'", "'\\''")
            f.write("file '" + sp2 + "'\n")  # concat quirk
    return lst

def build_commands_from_images(
    folder: Path,
    out_path: Path,
    opts: GifOptions,
    patterns: str,
    include_subdirs: bool,
    ffmpeg: str = "ffmpeg",
    work_dir: Optional[Path] = None,
    explicit_imgs: list[Path] | None = None,
) -> List[List[str]]:
    folder = Path(folder)
    work_dir = Path(work_dir) if work_dir else out_path.parent
    imgs = list(explicit_imgs) if explicit_imgs else _gather_images(folder, patterns, include_subdirs)
    try:
        imgs.sort(key=lambda x: _nat_key(str(x)))
    except Exception:
        pass
    if not imgs:
        raise GifError("No images found in the selected folder.")
    lst = _write_concat_list(imgs, float(opts.fps or 12), (opts.boomerang if opts.format == "gif" else False), work_dir, img_duration_sec=getattr(opts, 'img_duration_sec', None))
    filters = _gif_filters(opts)

    if opts.format == "gif":
        if not opts.use_palette:
            cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
            if filters: cmd += ["-vf", filters]
            if (getattr(opts, 'img_duration_sec', None) in (None, 0)) and opts.fps:
                cmd += ["-r", str(opts.fps)]
            cmd += ["-loop", "0" if opts.loop == 0 else str(opts.loop), str(out_path)]
            return [cmd]

        if opts.two_pass:
            pal = work_dir / (out_path.stem + "_pal.png")
            cmd1 = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-vf", f"{filters},{_palettegen(opts)}", str(pal)]
            dither = "bayer:bayer_scale=" + str(max(1, min(5, opts.bayer_scale))) if opts.dither == "bayer" else opts.dither
            paluse = f"paletteuse=dither={dither}"
            cmd2 = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(pal),
                    "-lavfi", f"{filters} [x]; [x][1:v] {paluse}"]
            if opts.fps: cmd2 += ["-r", str(opts.fps)]
            cmd2 += ["-loop", "0" if opts.loop == 0 else str(opts.loop), str(out_path)]
            return [cmd1, cmd2]

        dither = "bayer:bayer_scale=" + str(max(1, min(5, opts.bayer_scale))) if opts.dither == "bayer" else opts.dither
        paluse = f"paletteuse=dither={dither}"
        vf = f"{filters},split [a][b]; [a] {_palettegen(opts)} [p]; [b][p] {paluse}"
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", vf]
        if (getattr(opts, 'img_duration_sec', None) in (None, 0)) and opts.fps:
            cmd += ["-r", str(opts.fps)]
        cmd += ["-loop", "0" if opts.loop == 0 else str(opts.loop), str(out_path)]
        return [cmd]

    if opts.format == "webp":
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
        if filters: cmd += ["-vf", filters]
        if opts.lossless_webp: cmd += ["-lossless", "1"]
        else: cmd += ["-q:v", str(max(0, min(100, opts.quality_webp)))]
        cmd += ["-compression_level", str(max(0, min(9, opts.compression_webp)))]
        if (getattr(opts, 'img_duration_sec', None) in (None, 0)) and opts.fps:
            cmd += ["-r", str(opts.fps)]
        cmd += [str(out_path)]
        return [cmd]

    if opts.format == "apng":
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
        if filters: cmd += ["-vf", filters]
        if (getattr(opts, 'img_duration_sec', None) in (None, 0)) and opts.fps:
            cmd += ["-r", str(opts.fps)]
        cmd += [str(out_path)]
        return [cmd]

    raise GifError("Unknown format")



def encode_images_with_progress(
    pane,
    folder: Path,
    out_path: Path,
    opts: GifOptions,
    patterns: str,
    include_subdirs: bool,
    ffmpeg: str = "ffmpeg",
    work_dir: Optional[Path] = None,
    explicit_imgs: list[Path] | None = None,
) -> None:
    # GUI path: show progress + ETA; fallback = run commands directly
    try:
        from PySide6.QtWidgets import QMessageBox
    except Exception:
        for c in build_commands_from_images(folder, out_path, opts, patterns, include_subdirs, ffmpeg=ffmpeg, work_dir=work_dir):
            _ffmpeg_exec(c)
        return

    cmds = build_commands_from_images(folder, out_path, opts, patterns, include_subdirs, ffmpeg=ffmpeg, work_dir=work_dir, explicit_imgs=explicit_imgs)

    imgs = (list(explicit_imgs) if explicit_imgs else _gather_images(Path(folder), patterns, include_subdirs))
    est_frames = max(1, len(imgs))
    if opts.boomerang and opts.format == "gif" and len(imgs) > 1:
        est_frames = len(imgs) * 2 - 2

    progress_file = Path(work_dir or out_path.parent) / "_gif_progress.txt"
    try: progress_file.unlink(missing_ok=True)
    except Exception: pass

    def _inject(cmd: List[str]) -> List[str]:
        return [cmd[0], "-hide_banner", "-nostats", "-progress", str(progress_file)] + cmd[1:]

    try:
        pane.gif_pb.setVisible(True); pane.gif_eta.setVisible(True)
        pane.gif_pb.setValue(0); pane.gif_eta.setText("Starting…")
    except Exception: pass

    import subprocess, time
    start = time.time()

    def _parse_done():
        try:
            txt = progress_file.read_text("utf-8", errors="ignore")
        except Exception:
            return 0
        for ln in reversed(txt.splitlines()):
            if ln.startswith("frame="):
                try: return int(ln.split("=",1)[1].strip())
                except Exception: return 0
        return 0

    while cmds:
        c = cmds.pop(0)
        p = subprocess.Popen(_inject(c), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        while True:
            done = _parse_done()
            pct = int(round(100.0 * done / max(1, est_frames)))
            try:
                pane.gif_pb.setValue(pct)
                pane.gif_eta.setText(f"{pct}%")
            except Exception:
                pass
            if p.poll() is not None:
                break
            time.sleep(0.1)
        if p.returncode != 0:
            break

    try: progress_file.unlink(missing_ok=True)
    except Exception: pass
    try: pane.gif_pb.setVisible(False); pane.gif_eta.setVisible(False)
    except Exception: pass