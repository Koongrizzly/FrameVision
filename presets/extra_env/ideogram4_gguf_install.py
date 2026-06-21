from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


SDCPP_ZIP_URL = "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-679-f3fd359/sd-master-f3fd359-bin-win-cuda12-x64.zip"
IDEOGRAM_REPO = "https://huggingface.co/stduhpf/ideogram-4-gguf/resolve/main"

CONDITIONAL_FILES = {
    "Q5_0": {
        "name": "ideogram4-Q5_0.gguf",
        "min_bytes": 6 * 1024 * 1024 * 1024,
    },
    "Q6_K": {
        "name": "ideogram4-Q6_K.gguf",
        "min_bytes": 7 * 1024 * 1024 * 1024,
    },
    "Q8_0": {
        "name": "ideogram4-Q8_0.gguf",
        "min_bytes": 9 * 1024 * 1024 * 1024,
    },
}

UNCONDITIONAL_FILES = {
    "Q2_K": {
        "name": "ideogram4_unconditional-Q2_K.gguf",
        "min_bytes": 3 * 1024 * 1024 * 1024,
    },
    "Q8_0": {
        "name": "ideogram4_unconditional-Q8_0.gguf",
        "min_bytes": 9 * 1024 * 1024 * 1024,
    },
}

SHARED_MODEL_FILES = [
    {
        "name": "qwen3-vl-8b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/wareef/Qwen3-VL-8B-Instruct-Q4_K_M-GGUF/resolve/main/qwen3-vl-8b-instruct-q4_k_m.gguf?download=true",
        "min_bytes": 1024 * 1024 * 1024,
    },
    {
        "name": "flux2-vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors?download=true",
        "min_bytes": 50 * 1024 * 1024,
    },
]


def framevision_root() -> Path:
    # File is expected at /presets/extra_env/ideogram4_gguf_install.py
    return Path(__file__).resolve().parents[2]


def fmt_bytes(value: int | float) -> str:
    value = float(value or 0)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.2f} {units[idx]}"


def print_header(component: str, conditional_quant: str, unconditional_quant: str) -> None:
    print("=" * 72)
    print("FrameVision Ideogram 4 GGUF installer")
    print("=" * 72)
    print("Source repo:")
    print("  - https://huggingface.co/stduhpf/ideogram-4-gguf")
    print("Selected install:")
    print(f"  - component:             {component}")
    print(f"  - conditional model:     {conditional_quant}")
    print(f"  - unconditional model:   {unconditional_quant}")
    print("Shared files when needed:")
    print("  - Qwen3-VL 8B GGUF")
    print("  - Flux2 VAE")
    print("  - stable-diffusion.cpp CUDA12 sd-cli release")
    print()


def ideogram_file_url(name: str) -> str:
    return f"{IDEOGRAM_REPO}/{name}?download=true"


def build_download_list(component: str, conditional_quant: str, unconditional_quant: str) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []

    include_shared = component in {"all", "runtime", "conditional", "unconditional"}
    include_conditional = component in {"all", "conditional"}
    include_unconditional = component in {"all", "unconditional"}

    if include_conditional:
        item = dict(CONDITIONAL_FILES[conditional_quant])
        item["url"] = ideogram_file_url(str(item["name"]))
        files.append(item)

    if include_unconditional:
        item = dict(UNCONDITIONAL_FILES[unconditional_quant])
        item["url"] = ideogram_file_url(str(item["name"]))
        files.append(item)

    if include_shared:
        files.extend(dict(x) for x in SHARED_MODEL_FILES)

    return files


def url_download(url: str, target: Path, *, force: bool = False, min_bytes: int = 1) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force and target.stat().st_size >= min_bytes:
        print(f"[skip] {target.name} already exists ({fmt_bytes(target.stat().st_size)})")
        return

    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    print(f"[download] {target.name}")
    print(f"           {url}")
    started = time.time()
    last_print = 0.0

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FrameVision-Ideogram4-GGUF-Installer/1.1",
            "Accept": "*/*",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as fh:
            total_text = response.headers.get("Content-Length") or "0"
            try:
                total = int(total_text)
            except Exception:
                total = 0
            done = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                now = time.time()
                if now - last_print >= 2.0:
                    if total:
                        pct = done * 100.0 / max(1, total)
                        print(f"           {fmt_bytes(done)} / {fmt_bytes(total)} ({pct:.1f}%)")
                    else:
                        print(f"           {fmt_bytes(done)}")
                    last_print = now
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise

    size = tmp.stat().st_size if tmp.exists() else 0
    if size < min_bytes:
        try:
            tmp.unlink()
        except Exception:
            pass
        raise RuntimeError(f"Downloaded file is too small: {target.name} ({fmt_bytes(size)})")

    if target.exists():
        target.unlink()
    tmp.rename(target)
    elapsed = max(0.1, time.time() - started)
    print(f"[ok] {target.name} ({fmt_bytes(size)}, {fmt_bytes(size / elapsed)}/s)")


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"Unsafe zip entry blocked: {info.filename}")
        zf.extractall(extract_dir)


def install_sd_cli(bin_dir: Path, *, force: bool = True) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="framevision_sdcpp_") as td:
        tmp_dir = Path(td)
        zip_path = tmp_dir / "sd-master-f3fd359-bin-win-cuda12-x64.zip"
        url_download(SDCPP_ZIP_URL, zip_path, force=True, min_bytes=10 * 1024 * 1024)

        extract_dir = tmp_dir / "extract"
        print("[extract] stable-diffusion.cpp CUDA12 package")
        safe_extract_zip(zip_path, extract_dir)

        files = [p for p in extract_dir.rglob("*") if p.is_file()]
        if not any(p.name.lower() == "sd-cli.exe" for p in files):
            raise RuntimeError("The downloaded stable-diffusion.cpp zip did not contain sd-cli.exe")

        copied = 0
        for src in files:
            # Put the runnable package flat in presets/bin. This matches how
            # FrameVision already calls presets/bin/sd-cli.exe and keeps DLLs nearby.
            dest = bin_dir / src.name
            if dest.exists() and not force:
                continue
            shutil.copy2(src, dest)
            copied += 1

        print(f"[ok] copied {copied} stable-diffusion.cpp file(s) to {bin_dir}")

    # Temporary directory removes the downloaded zip and extracted files here.


def verify_install(root: Path, component: str, files: list[dict[str, object]], *, skip_sd_cli: bool) -> None:
    model_dir = root / "models" / "ideogram4_gguf"
    bin_dir = root / "presets" / "bin"
    required = [model_dir / str(f["name"]) for f in files]
    if not skip_sd_cli and component in {"all", "runtime", "conditional", "unconditional"}:
        required.append(bin_dir / "sd-cli.exe")

    missing = [p for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Install incomplete. Missing:\n  - " + "\n  - ".join(str(p) for p in missing))

    print()
    print("[verify] required files found")
    print(f"         model folder: {model_dir}")
    if (bin_dir / "sd-cli.exe").exists():
        print(f"         sd-cli:       {bin_dir / 'sd-cli.exe'}")
    print()
    print("Quick test:")
    print(f'"{bin_dir / "sd-cli.exe"}" --help | findstr /i "uncond"')


def normalize_quant(value: str) -> str:
    v = (value or "").strip().upper().replace("-", "_")
    aliases = {
        "Q5": "Q5_0",
        "Q6": "Q6_K",
        "Q8": "Q8_0",
        "Q2": "Q2_K",
    }
    return aliases.get(v, v)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Ideogram 4 GGUF support for FrameVision.")
    parser.add_argument("--force", action="store_true", help="Re-download model files even if they already exist.")
    parser.add_argument("--skip-models", action="store_true", help="Only install/update sd-cli.exe.")
    parser.add_argument("--skip-sd-cli", action="store_true", help="Only download model files.")
    parser.add_argument(
        "--component",
        choices=("all", "runtime", "conditional", "unconditional"),
        default="all",
        help="Which part to install. Model components also include the shared Qwen/VAE files so they are usable on their own.",
    )
    parser.add_argument(
        "--conditional-quant",
        default="Q5_0",
        help="Conditional Ideogram4 quant to download: Q5_0, Q6_K, Q8_0. Short aliases Q5/Q6/Q8 are accepted.",
    )
    parser.add_argument(
        "--unconditional-quant",
        default="Q2_K",
        help="Unconditional Ideogram4 quant to download: Q2_K or Q8_0. Short aliases Q2/Q8 are accepted.",
    )
    args = parser.parse_args()

    args.conditional_quant = normalize_quant(args.conditional_quant)
    args.unconditional_quant = normalize_quant(args.unconditional_quant)
    if args.conditional_quant not in CONDITIONAL_FILES:
        raise SystemExit(f"Unsupported conditional quant: {args.conditional_quant}. Use Q5_0, Q6_K, or Q8_0.")
    if args.unconditional_quant not in UNCONDITIONAL_FILES:
        raise SystemExit(f"Unsupported unconditional quant: {args.unconditional_quant}. Use Q2_K or Q8_0.")

    if args.skip_models:
        args.component = "runtime"

    root = framevision_root()
    model_dir = root / "models" / "ideogram4_gguf"
    bin_dir = root / "presets" / "bin"
    files = [] if args.skip_models else build_download_list(args.component, args.conditional_quant, args.unconditional_quant)

    print_header(args.component, args.conditional_quant, args.unconditional_quant)
    print(f"FrameVision root: {root}")
    print(f"Model folder:     {model_dir}")
    print(f"sd-cli folder:    {bin_dir}")
    print()

    try:
        if not args.skip_models:
            for item in files:
                url_download(
                    str(item["url"]),
                    model_dir / str(item["name"]),
                    force=bool(args.force),
                    min_bytes=int(item.get("min_bytes", 1)),
                )

        if not args.skip_sd_cli:
            install_sd_cli(bin_dir, force=True)

        verify_install(root, args.component, files, skip_sd_cli=bool(args.skip_sd_cli))
        print("[done] Ideogram 4 GGUF support is installed.")
        return 0
    except urllib.error.HTTPError as exc:
        print(f"[error] HTTP {exc.code}: {exc.reason}")
        print("Some Hugging Face files may require retrying, login, or manual download if the host blocks the request.")
        return 1
    except Exception as exc:
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
