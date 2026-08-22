#!/usr/bin/env python3
"""FrameVision queued HYPIR video command-chain runner.

Runs extract -> HYPIR frames -> encode as one queue job. The temporary work
folder is deleted only when every stage succeeds. Failed/cancelled jobs keep
frames for diagnosis/recovery.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FrameVision HYPIR queued video chain")
    p.add_argument("--payload", required=True, help="JSON payload written by upsc.py")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload_path = Path(args.payload).resolve()
    if not payload_path.exists():
        raise FileNotFoundError(f"HYPIR queue payload not found: {payload_path}")

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    cmds = payload.get("cmds") or []
    cwd = payload.get("cwd") or None
    cleanup_dir = payload.get("cleanup_dir") or ""

    if not isinstance(cmds, list) or not cmds:
        raise RuntimeError("HYPIR queue payload has no commands")

    print(f"[HYPIR QUEUE] stages={len(cmds)}")
    if cleanup_dir:
        print(f"[HYPIR QUEUE] temp={cleanup_dir}")

    env = os.environ.copy()
    extra_env = payload.get("env")
    if isinstance(extra_env, dict):
        for k, v in extra_env.items():
            if k is not None and v is not None:
                env[str(k)] = str(v)

    for idx, cmd in enumerate(cmds, 1):
        if not isinstance(cmd, list) or not cmd:
            raise RuntimeError(f"HYPIR queue stage {idx} is invalid")
        print(f"[HYPIR QUEUE] stage {idx}/{len(cmds)}")
        print("[HYPIR QUEUE] " + " ".join(str(x) for x in cmd))
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line.rstrip(), flush=True)
        code = int(proc.wait())
        if code != 0:
            print(f"[HYPIR QUEUE] stage {idx} failed with code {code}; temporary frames kept.", flush=True)
            return code

    if cleanup_dir:
        try:
            work = Path(cleanup_dir).resolve()
            # Conservative guard: only remove HYPIR's own temporary work folders.
            if work.exists() and work.is_dir() and "_hypir_x" in work.name.lower() and work.name.lower().endswith("_work"):
                shutil.rmtree(work)
                print(f"[HYPIR QUEUE] cleaned temporary frames: {work}", flush=True)
            else:
                print(f"[HYPIR QUEUE] cleanup skipped by safety guard: {work}", flush=True)
        except Exception as exc:
            # Output is already successfully encoded, so cleanup failure should
            # not convert the finished video job into a failed generation.
            print(f"[HYPIR QUEUE] cleanup warning: {exc}", flush=True)

    print("[HYPIR QUEUE] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
