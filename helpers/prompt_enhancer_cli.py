from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import io
import contextlib

# -----------------------------------------------------------------------------
# Robust imports of the app's prompt helper
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve()
HELPERS_DIR = HERE.parent
APP_ROOT = HELPERS_DIR.parent

# Ensure helpers and app root are importable even if 'helpers' isn't a package.
for p in (str(APP_ROOT), str(HELPERS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

HAVE_PROMPT = False
IMPORT_ERROR = None

try:
    from helpers.prompt import (
        _load_settings,
        LENGTH_PRESETS,
        DEFAULT_TEMPLATE_BASE,
        DEFAULT_STYLE,
        DEFAULT_NEG,
        OPENING_VARIANTS,
        _merge_negatives,
        _qwen3_local_folder,
        _generate_with_qwen_text,
    )
    HAVE_PROMPT = True
except Exception as e1:
    try:
        from prompt import (
            _load_settings,
            LENGTH_PRESETS,
            DEFAULT_TEMPLATE_BASE,
            DEFAULT_STYLE,
            DEFAULT_NEG,
            OPENING_VARIANTS,
            _merge_negatives,
            _qwen3_local_folder,
            _generate_with_qwen_text,
        )
        HAVE_PROMPT = True
    except Exception as e2:
        IMPORT_ERROR = e2 or e1


def _compose_from_settings(seed: str, extra_neg: str = "", target_override: str = ""):
    """Build system/user prompts using saved Prompt tab settings with optional target override."""
    try:
        settings = _load_settings() if HAVE_PROMPT else {}
    except Exception:
        settings = {}

    target = (settings.get("target") or "image").strip().lower()
    if target not in ("image", "video"):
        target = "image"

    # Apply CLI override if provided
    if target_override:
        try:
            to = (target_override or "").strip().lower()
            if to in ("image", "video"):
                target = to
        except Exception:
            pass

    length_label = settings.get("length_choice") or "Medium (80–120 words)"
    length_words, _max_tokens = LENGTH_PRESETS.get(
        length_label,
        LENGTH_PRESETS.get("Medium (80–120 words)"),
    )

    template = DEFAULT_TEMPLATE_BASE.format(
        length_words=length_words,
        target="image" if target == "image" else "video",
    )

    style = (settings.get("style") or DEFAULT_STYLE).strip()
    negatives = (settings.get("negatives") or DEFAULT_NEG).strip()

    # Merge negatives from settings + CLI
    if extra_neg and extra_neg.strip():
        try:
            negatives = _merge_negatives(negatives, extra_neg)
        except Exception:
            negatives = (negatives + ", " + extra_neg).strip(", ")

    # Opening variant for light randomness
    try:
        import random as _r
        opener = _r.choice(OPENING_VARIANTS)
    except Exception:
        opener = "A cinematic shot of"

    # For video prompting, prefer a motion-oriented opener
    if target == "video":
        opener = "A cinematic video of"

    try:
        import re as _re
        neg_clean = negatives if _re.search(r"[A-Za-z0-9]", negatives or "") else ""
    except Exception:
        neg_clean = negatives

    sys_prompt = (
        "You are a visual prompt engineer. "
        "Expand short seeds into a single richly detailed prompt. "
        "Follow the template and style hints exactly."
    )

    user_prompt = (
        f"{template} "
        f"Begin the single sentence with {opener} (no quotes). "
    )
    if style:
        user_prompt += f"Use the style: {style}. "
    user_prompt += f"Seed: {seed}"

    return sys_prompt, user_prompt, neg_clean, target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prompt enhancer helper for FrameVision (Qwen3-VL).")
    parser.add_argument("--seed", required=True, help="Base seed/idea text.")
    parser.add_argument("--neg", default="", help="Extra negative prompt text to merge.")
    parser.add_argument("--target", default="", help="Override target mode: image or video.")
    parser.add_argument("--out-json", default="-", help="Output JSON file or '-' for stdout.")
    args = parser.parse_args(argv)

    # Always output machine-readable JSON on stdout.
    if not HAVE_PROMPT:
        payload = {"ok": False, "error": f"helpers.prompt import failed: {IMPORT_ERROR}"}
        _emit(payload, args.out_json)
        return 1

    seed = (args.seed or "").strip()
    if not seed:
        payload = {"ok": False, "error": "Empty seed text."}
        _emit(payload, args.out_json)
        return 1

    sys_p, usr_p, merged_neg, target = _compose_from_settings(seed, args.neg or "", args.target or "")

    folder = _qwen3_local_folder()
    if not folder or not Path(folder).exists():
        payload = {
            "ok": False,
            "error": (
                f"Qwen3-VL local folder not found at {folder!r}. "
                f"Open the Prompt tab once and configure a Qwen3-VL model."
            ),
        }
        _emit(payload, args.out_json)
        return 2

    try:
        settings = _load_settings()
    except Exception:
        settings = {}

    try:
        temperature = float(settings.get("temperature", 0.85))
    except Exception:
        temperature = 0.85

    try:
        max_new = int(settings.get("max_new_tokens", 280))
    except Exception:
        max_new = 280

    # Prevent any library prints from polluting stdout (UI expects clean JSON).
    capture_out = io.StringIO()
    capture_err = io.StringIO()

    try:
        with contextlib.redirect_stdout(capture_out), contextlib.redirect_stderr(capture_err):
            text = _generate_with_qwen_text(Path(folder), sys_p, usr_p, temperature, max_new)
    except KeyboardInterrupt:
        payload = {"ok": False, "error": "Cancelled."}
        _emit(payload, args.out_json)
        return 3
    except Exception as e:
        # Include any captured stderr for debugging (still inside JSON)
        extra = ""
        try:
            err_txt = capture_err.getvalue().strip()
            if err_txt:
                extra = f" | stderr: {err_txt[:500]}"
        except Exception:
            extra = ""
        payload = {"ok": False, "error": f"{type(e).__name__}: {e}{extra}"}
        _emit(payload, args.out_json)
        return 4

    payload = {
        "ok": True,
        "prompt": (text or "").strip(),
        "negatives": merged_neg,
        "target": target,
    }

    _emit(payload, args.out_json)
    return 0


def _emit(payload: dict, out: str):
    """Write payload either to a file or stdout. Always write pure JSON."""
    try:
        js = json.dumps(payload, ensure_ascii=False)
    except Exception:
        js = '{"ok": false, "error": "Failed to serialize JSON."}'

    out = (out or "-").strip()
    if out and out != "-":
        try:
            Path(out).write_text(js, encoding="utf-8")
            return
        except Exception:
            # Fall through to stdout
            pass

    # ONLY JSON on stdout
    sys.stdout.write(js)
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
