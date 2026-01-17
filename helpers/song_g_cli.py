"""
SongG CLI wrapper for FrameVision SongGeneration repo

Goals:
- No edits to upstream repo files
- Reduce console noise by using correct env vars (HF_HOME) and optional warning suppression
- Normalize common flags (--bgm/--vocal/--separate) -> --generate_type
- Safely enable --use_flash_attn only if flash_attn_func is importable inside .song_g_env

Usage (from FrameVision root):
  .song_g_env\Scripts\python.exe helpers\song_g_cli.py ^
      --ckpt songgeneration_large ^
      --input_jsonl output\music\song_g\jsonl\song_001_xxx.jsonl ^
      --save_dir output\music\song_g ^
      --low_mem ^
      --generate_type mixed

Notes:
- ckpt can be: "songgeneration_large" OR "ckpt/songgeneration_large" OR an absolute path.
- By default, we set HF_HOME and remove TRANSFORMERS_CACHE to avoid the deprecation warning.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _is_abs_path(p: str) -> bool:
    if not p:
        return False
    p = p.strip()
    return (
        (len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/"))
        or p.startswith("\\\\")
        or p.startswith("/")
    )


def _app_root() -> Path:
    # helpers/song_g_cli.py -> FrameVision root
    return Path(__file__).resolve().parents[1]


def _repo_dir(root: Path) -> Path:
    return (root / "models" / "song_generation").resolve()


def _venv_python(root: Path) -> Path:
    return (root / ".song_g_env" / "Scripts" / "python.exe").resolve()


def _normalize_ckpt_arg(ckpt: str, repo: Path) -> str:
    ckpt = (ckpt or "").strip().strip('"')
    if not ckpt:
        raise SystemExit("Missing --ckpt")

    # absolute directory
    if _is_abs_path(ckpt) or os.path.isdir(ckpt):
        p = Path(ckpt).resolve()
        try:
            rel = p.relative_to(repo)
            return rel.as_posix().replace("\\", "/")
        except Exception:
            return str(p)

    low = ckpt.replace("\\", "/")
    if low.lower().startswith("ckpt/"):
        return low

    # bare folder name
    return f"ckpt/{ckpt}"


def _flash_attn_available(vpy: Path, repo: Path) -> bool:
    # Check inside the SongG venv (important)
    code = (
        "import sys\n"
        "try:\n"
        "    from flash_attn import flash_attn_func\n"
        "    print('ok')\n"
        "except Exception:\n"
        "    try:\n"
        "        from flash_attn.flash_attn_interface import flash_attn_func\n"
        "        print('ok')\n"
        "    except Exception:\n"
        "        print('no')\n"
    )
    try:
        p = subprocess.run(
            [str(vpy), "-c", code],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (p.stdout or "").strip().lower()
        return out.endswith("ok")
    except Exception:
        return False


def _build_env(root: Path, quiet_warnings: bool) -> dict:
    env = dict(os.environ)

    # Correct naming: avoid TRANSFORMERS_CACHE warning by using HF_HOME.
    env.pop("TRANSFORMERS_CACHE", None)

    if not env.get("HF_HOME"):
        env["HF_HOME"] = str((root / ".hf_home").resolve())

    # Reduce noise (optional). This does NOT change model behavior.
    if quiet_warnings:
        env.setdefault(
            "PYTHONWARNINGS",
            ",".join(
                [
                    "ignore:Using `TRANSFORMERS_CACHE` is deprecated.*:FutureWarning",
                    "ignore:Using `is_flash_attn_available` is deprecated.*:UserWarning",
                    "ignore:Special tokens have been added.*:UserWarning",
                    "ignore:You are using an old version of the checkpointing format.*:UserWarning",
                ]
            ),
        )
        env.setdefault("TRANSFORMERS_VERBOSITY", "error")

    return env


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="FrameVision SongG CLI wrapper (no upstream edits)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint folder name, ckpt/<name>, or full path")
    ap.add_argument("--input_jsonl", required=True, help="Input JSONL path")
    ap.add_argument("--save_dir", required=True, help="Output folder")
    ap.add_argument("--generate_type", default="mixed", choices=["mixed", "bgm", "vocal", "separate"])
    ap.add_argument("--low_mem", action="store_true")
    ap.add_argument("--use_flash_attn", action="store_true")
    ap.add_argument(
        "--quiet_warnings", action="store_true", default=True, help="Suppress noisy warnings (default: on)"
    )
    ap.add_argument("--no_quiet_warnings", dest="quiet_warnings", action="store_false")

    # Convenience aliases to match what you tried earlier
    ap.add_argument("--bgm", action="store_true", help="Alias for --generate_type bgm")
    ap.add_argument("--vocal", action="store_true", help="Alias for --generate_type vocal")
    ap.add_argument("--separate", action="store_true", help="Alias for --generate_type separate")

    args = ap.parse_args(argv)

    if args.bgm:
        args.generate_type = "bgm"
    elif args.vocal:
        args.generate_type = "vocal"
    elif args.separate:
        args.generate_type = "separate"

    root = _app_root()
    repo = _repo_dir(root)
    vpy = _venv_python(root)

    if not vpy.exists():
        print(f"[ERR] Missing SongG venv python: {vpy}")
        return 2
    if not repo.exists():
        print(f"[ERR] Missing SongGeneration repo: {repo}")
        return 2

    ckpt_arg = _normalize_ckpt_arg(args.ckpt, repo)

    inp = Path(args.input_jsonl)
    if not _is_abs_path(str(inp)):
        inp = (root / inp).resolve()

    out = Path(args.save_dir)
    if not _is_abs_path(str(out)):
        out = (root / out).resolve()

    cmd = [
        str(vpy),
        str((repo / "generate.py").resolve()),
        "--ckpt_path",
        ckpt_arg,
        "--input_jsonl",
        str(inp),
        "--save_dir",
        str(out),
        "--generate_type",
        args.generate_type,
    ]
    if args.low_mem:
        cmd.append("--low_mem")

    if args.use_flash_attn:
        if _flash_attn_available(vpy, repo):
            cmd.append("--use_flash_attn")
        else:
            print(
                "[WARN] Flash Attention requested, but 'flash_attn_func' is not importable in .song_g_env. "
                "Running without --use_flash_attn to avoid crash."
            )

    env = _build_env(root, args.quiet_warnings)

    print("[RUN] " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    p = subprocess.run(cmd, cwd=str(repo), env=env)
    return int(p.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
