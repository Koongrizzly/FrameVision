
"""
helpers/safe_scale.py
Modular Safe-Scale for Real-ESRGAN: native→target with chaining.
Small surface in upsc.py; all logic is here.
"""
import re
from pathlib import Path

def _native_from_model_text(txt: str, default=4) -> int:
    try:
        m = re.search(r"(?:-|_)?x([234])\b", (txt or "").lower())
        if m:
            v = int(m.group(1))
            if v in (1,2,3,4): return v
    except Exception:
        pass
    return default

def _plan_chain(desired: int, native: int):
    """
    Return (passes, residual_ratio, effective_native_mult).
    passes: number of Real-ESRGAN native passes (>=1 when engine is Real-ESRGAN)
    residual_ratio: final resize factor to reach desired
    effective_native_mult: native**passes
    """
    desired = max(1, min(4, int(desired)))
    native = max(1, min(4, int(native)))
    if native == 1:
        return 0, float(desired), 1
    mult = 1
    passes = 0
    # at most two passes given slider upper bound 4x
    while mult * native <= desired and passes < 3:
        mult *= native
        passes += 1
    residual = float(desired) / float(mult)
    return passes, residual, mult

def _join_vf(existing: str, extra: str) -> str:
    if not existing: return extra
    if not extra: return existing
    return f"{existing},{extra}"

def _even_scale_expr(ratio: float) -> str:
    # ensure even dimensions for yuv420p, keep lanczos
    return f"scale=ceil(iw*{ratio:.6f}/2)*2:ceil(ih*{ratio:.6f}/2)*2:flags=lanczos"

def install(env: dict):
    """
    env: globals() from upsc.py so we can access UpscPane, FFMPEG, etc.
    """
    UpscPane = env.get("UpscPane")
    FFMPEG   = env.get("FFMPEG")
    _exists  = env.get("_exists")
    _VIDEO_EXTS = env.get("_VIDEO_EXTS", set())
    _IMAGE_EXTS = env.get("_IMAGE_EXTS", set())

    if UpscPane is None or FFMPEG is None:
        return  # not our environment

    # ---- Wrap _build_post_filters to append final scaler for any residual ----
    _orig_post = getattr(UpscPane, "_build_post_filters", None)
    def _wrap_post(self) -> str:
        post = _orig_post(self) if callable(_orig_post) else ""
        try:
            if "Real-ESRGAN" not in str(self.combo_engine.currentText()):
                return post
            desired = int(round(float(self.spin_scale.value())))
            model = self.combo_model_realsr.currentText()
            native = _native_from_model_text(model, 4)
            eff_mult = getattr(self, "_fv_chain_native_mult", native)
            if desired == eff_mult:
                return post
            ratio = float(desired) / float(eff_mult)
            scale_expr = _even_scale_expr(ratio)
            return _join_vf(post, scale_expr) if post else scale_expr
        except Exception:
            return post
    if callable(_orig_post):
        setattr(UpscPane, "_build_post_filters", _wrap_post)

    # ---- Wrap _build_cmds_for_path: insert chained passes for video/images ----
    _orig_build = getattr(UpscPane, "_build_cmds_for_path", None)
    def _build_cmds_for_path_v3(self, src, outd_override=None):
        cmds, outfile = _orig_build(self, src, outd_override)
        try:
            if "Real-ESRGAN" not in str(self.combo_engine.currentText()):
                return cmds, outfile

            desired = int(round(float(self.spin_scale.value())))
            model = self.combo_model_realsr.currentText()
            native = _native_from_model_text(model, 4)
            passes, residual, mult = _plan_chain(desired, native)
            setattr(self, "_fv_chain_native_mult", mult)

            if not cmds:
                return cmds, outfile

            # Video heuristic: [extract, upscaler, encode]
            if len(cmds) >= 3 and isinstance(cmds[1], list) and isinstance(cmds[-1], list):
                up1 = cmds[1]
                # Force first upscaler to native
                try:
                    if "-s" in up1:
                        i = len(up1) - 1 - up1[::-1].index("-s")
                        if i+1 < len(up1): up1[i+1] = str(native)
                except Exception:
                    pass

                # If two passes requested, duplicate upscaler command to create a second native pass
                if passes >= 2:
                    up2 = list(up1)
                    try:
                        # change output directory from ".../out" to ".../out2"
                        # This is robust for our builder which uses in/out directories
                        last = up2[-1]
                        from pathlib import Path as _P
                        up2[-1] = str(_P(last).with_name("out2"))
                    except Exception:
                        pass
                    # Insert just after first upscale
                    cmds.insert(2, up2)
                    # Encoder must read from out2 frames
                    enc = cmds[-1]
                    try:
                        if "-i" in enc:
                            j = enc.index("-i")
                            in_seq = Path(enc[j+1])
                            enc[j+1] = str(in_seq.parents[0].with_name("out2") / in_seq.name)
                    except Exception:
                        pass
                return cmds, outfile

            # Image path
            if len(cmds) >= 1 and isinstance(cmds[0], list):
                outp = Path(outfile)
                sr1 = outp.with_name(outp.stem + "_sr1.png")
                up1 = cmds[0]
                try:
                    if "-s" in up1:
                        i = len(up1) - 1 - up1[::-1].index("-s")
                        if i+1 < len(up1): up1[i+1] = str(native)
                    if "-o" in up1:
                        j = len(up1) - 1 - up1[::-1].index("-o")
                        if j+1 < len(up1): up1[j+1] = str(sr1)
                    else:
                        up1 += ["-o", str(sr1)]
                except Exception:
                    pass

                if passes >= 2:
                    sr2 = outp.with_name(outp.stem + "_sr2.png")
                    exe = self.combo_engine.currentData()
                    up2 = self._realsr_cmd_file(exe, sr1, sr2, model, native)
                    if residual != 1.0:
                        resize = [env["FFMPEG"], "-hide_banner", "-loglevel", "warning", "-y",
                                  "-i", str(sr2), "-vf", _even_scale_expr(residual), str(outfile)]
                        return [up1, up2, resize], outfile
                    else:
                        copy = [env["FFMPEG"], "-hide_banner", "-loglevel", "warning", "-y",
                                "-i", str(sr2), str(outfile)]
                        return [up1, up2, copy], outfile
                else:
                    if residual != 1.0:
                        resize = [env["FFMPEG"], "-hide_banner", "-loglevel", "warning", "-y",
                                  "-i", str(sr1), "-vf", _even_scale_expr(residual), str(outfile)]
                        return [up1, resize], outfile
                    else:
                        copy = [env["FFMPEG"], "-hide_banner", "-loglevel", "warning", "-y",
                                "-i", str(sr1), str(outfile)]
                        return [up1, copy], outfile
        except Exception:
            return cmds, outfile
    if callable(_orig_build):
        setattr(UpscPane, "_build_cmds_for_path", _build_cmds_for_path_v3)

    # ---- Help _run_one record effective multiplier for single-run logs ----
    _orig_run_one = getattr(UpscPane, "_run_one", None)
    def _run_one_v3(self, src):
        try:
            if "Real-ESRGAN" in str(self.combo_engine.currentText()):
                desired = int(round(float(self.spin_scale.value())))
                model = self.combo_model_realsr.currentText()
                native = _native_from_model_text(model, 4)
                passes, residual, mult = _plan_chain(desired, native)
                setattr(self, "_fv_chain_native_mult", mult)
        except Exception:
            pass
        return _orig_run_one(self, src)
    if callable(_orig_run_one):
        setattr(UpscPane, "_run_one", _run_one_v3)

