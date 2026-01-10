#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

UA = "FrameVision-ZImage-GGUF-Installer/1.5"

# ---- Download URLs (kept explicit so we don't need per-variant .bat files)
# Z-Image Turbo diffusion GGUF quants:
# - Q4_0 / Q5_0 / Q6_K / Q8_0 are available in leejet/Z-Image-Turbo-GGUF.
ZIMAGE_DIFFUSION_URLS = {
    "Q4_0": "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q4_0.gguf",
    "Q5_0": "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q5_0.gguf",
    "Q6_K": "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q6_K.gguf",
    "Q8_0": "https://huggingface.co/leejet/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q8_0.gguf",
}

# Qwen3-4B text encoder GGUF quants (prefer unsloth; fallback to ggml-org if present).
QWEN_TEXT_URLS = {
    "Q4_K_M": [
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    ],
    "Q5_K_M": [
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q5_K_M.gguf",
        "https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q5_K_M.gguf",
    ],
    "Q6_K": [
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q6_K.gguf",
        "https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q6_K.gguf",
    ],
    "Q8_0": [
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q8_0.gguf",
        "https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q8_0.gguf",
    ],
}

# ---- Safetensors (ComfyUI-style) downloads
# Default repo with single-file diffusion models in fp32/fp16/bf16 + matching text encoder + VAE.
SAFE_REPO_DEFAULT = "tsqn/Z-Image-Turbo_fp32-fp16-bf16_comfyui"
SAFE_BASE_URL = "https://huggingface.co/{repo}/resolve/main/"

SAFE_DIFFUSION_FILES = {
    "fp16": "z_image_turbo_fp16.safetensors",
    "bf16": "z_image_turbo_bf16.safetensors",
    "fp32": "z_image_turbo_fp32.safetensors",
}

SAFE_TEXT_ENCODER_FILE = "text_encoders/qwen_3_4b_bf16.safetensors"

SAFE_VAE_FILES = [
    "vae/ae.safetensors",
    "vae/ae_bf16.safetensors",
]



# ---- Diffusers (folder) snapshot downloads (used by the FP16 installer)
# This installs the *Diffusers* layout (model_index.json + subfolders like transformer/, text_encoder/, vae/, scheduler/, tokenizer/, ...).
# Default points to the official model repo; override via --diffusers-repo if you maintain your own repack.
DIFFUSERS_REPO_DEFAULT = "Tongyi-MAI/Z-Image-Turbo"
DIFFUSERS_REVISION_DEFAULT = "main"
HF_MODEL_API = "https://huggingface.co/api/models/{repo}"

VAE_URL = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"
SDCPP_LATEST_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"

# Map diffusion quant -> a sensible Qwen text quant.
MATCHING_QWEN_FOR_DIFF = {
    "Q4_0": "Q4_K_M",
    "Q5_0": "Q5_K_M",
    "Q6_K": "Q6_K",
    "Q8_0": "Q8_0",
}

def log(msg: str) -> None:
    print(msg, flush=True)

def _shared_sdcli_dir(root: Path) -> Path:
    # Shared location for stable-diffusion.cpp CLI + DLLs used by multiple model installers.
    return (root / "presets" / "bin").resolve()

def _sdcli_present(bin_dir: Path) -> bool:
    sdcli = bin_dir / "sd-cli.exe"
    dll_ok = (bin_dir / "stable-diffusion.dll").exists() or (bin_dir / "diffusers.dll").exists()
    return sdcli.exists() and dll_ok

def _sha256_file(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _utc_now_iso() -> str:
    # Simple UTC timestamp without pulling in datetime (keeps deps tiny)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _safe_write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8", errors="replace")
        try:
            if path.exists():
                try:
                    os.chmod(path, 0o666)
                except Exception:
                    pass
                path.unlink()
        except Exception:
            pass
        os.replace(tmp, path)
    except Exception as e:
        log(f"[zimage-gguf] warn: could not write {path}: {e}")

def _update_3rd_party_licenses_json(root: Path, item_id: str, updates: dict) -> None:
    """Best-effort update of presets/info/3rd_party_licenses.json.

    Adds 'installed_*' fields so the License Viewer can show what was actually installed
    when using a moving 'latest' download.
    """
    try:
        lic_path = (root / "presets" / "info" / "3rd_party_licenses.json").resolve()
        if not lic_path.exists():
            log(f"[zimage-gguf] warn: licenses json not found: {lic_path}")
            return

        data = json.loads(lic_path.read_text(encoding="utf-8", errors="replace"))
        items = data.get("items") or []
        hit = None
        for it in items:
            if it.get("id") == item_id:
                hit = it
                break
        if hit is None:
            log(f"[zimage-gguf] warn: license item id not found: {item_id}")
            return

        for k, v in updates.items():
            hit[k] = v

        data["generated_at_utc"] = _utc_now_iso()
        _safe_write_text(lic_path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"[zimage-gguf] warn: could not update 3rd_party_licenses.json: {e}")

def _collect_sdcpp_build_text(exe: Path) -> str:
    """Try to collect a human-readable version blob from sd-cli.exe."""
    import subprocess

    def _run(args: list[str]) -> tuple[int, str, str]:
        kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.run([str(exe), *args], **kwargs)
        out = (p.stdout or b"").decode("utf-8", errors="replace")
        err = (p.stderr or b"").decode("utf-8", errors="replace")
        return p.returncode, out, err

    for args in (["--version"], ["-v"], ["--help"]):
        try:
            rc, out, err = _run(args)
            blob = (out.strip() + "\n" + err.strip()).strip()
            if blob:
                return f"$ {exe.name} {' '.join(args)}\n{blob}\n(returncode={rc})\n"
        except Exception:
            pass
    return f"$ {exe.name} --help\n(no output)\n"

def _token_headers() -> dict:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if tok:
        return {"Authorization": f"Bearer {tok}"}
    return {}

def _normalize_hf_url(url: str) -> str:
    if "huggingface.co/" in url and "download=true" not in url:
        return url + ("&download=true" if "?" in url else "?download=true")
    return url

def http_get(url: str) -> bytes:
    headers = {"User-Agent": UA, **_token_headers()}
    req = Request(url, headers=headers)
    with urlopen(req, timeout=120) as r:
        return r.read()

def download(url: str, dst: Path, retries: int = 6) -> None:
    """Robust downloader with resume + atomic finalization.

    Writes to <dst>.part and renames to <dst> when complete.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If a complete file exists, keep it.
    if dst.exists() and dst.stat().st_size > 0:
        log(f"[zimage-gguf] exists: {dst.name}")
        return

    part = dst.with_suffix(dst.suffix + ".part")
    last = None

    for i in range(retries):
        try:
            u = _normalize_hf_url(url)
            existing = part.stat().st_size if part.exists() else 0

            headers = {
                "User-Agent": UA,
                "Accept": "application/octet-stream",
                **_token_headers(),
            }

            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                log(f"[zimage-gguf] resume: {u} (+{existing} bytes)")
            else:
                log(f"[zimage-gguf] download: {u}")

            req = Request(u, headers=headers)
            with urlopen(req, timeout=180) as r:
                status = getattr(r, "status", None)
                # If server didn't honor Range (status 200), restart the download.
                if existing > 0 and status == 200:
                    try:
                        part.unlink()
                    except Exception:
                        pass
                    existing = 0

                mode = "ab" if existing > 0 else "wb"
                with open(part, mode) as f:
                    total = existing
                    while True:
                        chunk = r.read(1024 * 1024 * 4)
                        if not chunk:
                            break
                        f.write(chunk)
                        total += len(chunk)
                        if total and total % (32 * 1024 * 1024) == 0:
                            log(f"[zimage-gguf]   {total / (1024*1024):.0f} MiB")

            if not part.exists() or part.stat().st_size == 0:
                raise RuntimeError("downloaded 0 bytes")

            # Atomic finalize.
            try:
                if dst.exists():
                    try:
                        os.chmod(dst, 0o666)
                    except Exception:
                        pass
                    dst.unlink()
            except Exception:
                pass

            os.replace(part, dst)
            return

        except Exception as e:
            last = e
            # Backoff and try again. Keep .part for resume unless it's clearly bogus.
            time.sleep(1.2 * (i + 1))

    raise RuntimeError(f"failed download after {retries} retries: {url} ({last})")

def download_any(urls: list[str], dst: Path) -> None:
    last = None
    for u in urls:
        try:
            download(u, dst, retries=2)
            return
        except Exception as e:
            last = e
            log(f"[zimage-gguf]   mirror failed: {u} ({type(e).__name__})")
    raise RuntimeError(f"all mirrors failed for {dst.name}\n{last}")


def hf_list_repo_files(repo: str, revision: str = "main") -> list[str]:
    """List file paths in a Hugging Face model repo using the public API."""
    url = HF_MODEL_API.format(repo=repo)
    if revision:
        url = url + ("?revision=" + revision)
    data = json.loads(http_get(url).decode("utf-8", errors="replace"))
    sib = data.get("siblings") or []
    files: list[str] = []
    for s in sib:
        fn = s.get("rfilename") or s.get("path") or s.get("name")
        if fn:
            files.append(fn)
    return sorted(set(files))

def _install_diffusers_snapshot(repo: str, revision: str, target: Path) -> None:
    """Download the full Diffusers snapshot for Z-Image Turbo.

    Mirrors the original fp16_install.bat which used huggingface_hub.snapshot_download(),
    but avoids requiring huggingface_hub to be installed.
    """
    target.mkdir(parents=True, exist_ok=True)

    log(f"[zimage-diff] repo    : {repo}")
    log(f"[zimage-diff] revision: {revision}")

    files = hf_list_repo_files(repo, revision)
    if not files:
        raise RuntimeError(f"No files returned by HF API for repo: {repo}")

    log(f"[zimage-diff] files   : {len(files)}")

    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    for rel in files:
        if not rel or rel.endswith("/"):
            continue
        dst = target / rel
        download(base + rel, dst)

    # Verify model_index.json exists (Diffusers layout sanity check).
    mi = target / "model_index.json"
    if not mi.exists():
        (target / "_DIFFUSERS_INCOMPLETE.txt").write_text(
            "\n".join(
                [
                    "[WARN] model_index.json not found after download.",
                    "This usually indicates an incomplete download.",
                    "Try deleting the folder and running the installer again.",
                    "",
                    f"Repo: {repo}",
                    f"Revision: {revision}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        log("[zimage-diff] warn: model_index.json not found (incomplete download?)")
    else:
        log("[zimage-diff] ok: model_index.json found")

    (target / "_diffusers_sources.txt").write_text(
        "\n".join(
            [
                f"Repo: {repo}",
                f"Revision: {revision}",
                f"File count: {len(files)}",
                "",
                "Files:",
                *[f"  {f}" for f in files],
                "",
            ]
        ),
        encoding="utf-8",
    )

def extract_zip(zip_path: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir, ignore_errors=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst_dir)

def _zip_exe_entries(zip_path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            return [n for n in z.namelist() if n.lower().endswith(".exe")]
    except Exception:
        return []

def _find_sdcpp_exe(root: Path):
    preferred = {"sd-cli.exe", "sd.exe", "stable-diffusion.exe"}
    candidates = []
    for p in root.rglob("*.exe"):
        name = p.name.lower()
        if name in preferred:
            candidates.append(p)
    if candidates:
        candidates.sort(key=lambda p: (0 if p.name.lower()=="sd-cli.exe" else 1 if p.name.lower()=="sd.exe" else 2, len(str(p))))
        return candidates[0]
    for p in root.rglob("*.exe"):
        n = p.name.lower()
        if "sd" in n and "test" not in n and "bench" not in n:
            return p
    return None



def _is_likely_debug_asset_name(name: str) -> bool:
    n = (name or "").lower()
    # Not perfect, but avoids obvious debug-labelled assets.
    return ("debug" in n) or ("-dbg" in n) or n.endswith("_dbg.zip")


def _probe_sdcpp_exe(exe: Path) -> tuple[bool, str]:
    """Return (ok, reason) where ok means the sd executable can start.

    We suppress Windows system error dialogs (missing DLL / side-by-side) so
    a bad binary won't pop modal dialogs during installs.
    """
    import subprocess

    def _run_probe() -> tuple[bool, str]:
        cmd = [str(exe), "--help"]
        kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.run(cmd, **kwargs)
        # Some builds return 0 or 1 for --help; both indicate the exe loaded.
        if p.returncode in (0, 1):
            return True, "ok"
        err = (p.stderr or b"").decode("utf-8", errors="ignore")
        out = (p.stdout or b"").decode("utf-8", errors="ignore")
        blob = (err + "\n" + out).lower()
        for key in ("vcruntime140d", "msvcp140d", "ucrtbased"):
            if key in blob:
                return False, f"missing debug runtime ({key})"
        return False, f"returncode={p.returncode}"

    if os.name != "nt":
        return _run_probe()

    try:
        import ctypes
        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOGPFAULTERRORBOX = 0x0002
        SEM_NOOPENFILEERRORBOX = 0x8000
        kernel32 = ctypes.windll.kernel32
        prev = kernel32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX)
        try:
            return _run_probe()
        finally:
            kernel32.SetErrorMode(prev)
    except Exception:
        # Fallback: just run probe.
        try:
            return _run_probe()
        except OSError as e:
            return False, f"oserror: {e}"
def _copy_sdcpp_bins(extracted: Path, bin_dir: Path) -> None:
    """Install stable-diffusion.cpp runtime files into a shared bin folder.

    We keep this shared so Z-Image GGUF + Qwen2512 (+ future Qwen2511) don't each
    download/extract their own copy.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_copy(src: Path, dest: Path, retries: int = 10) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        last = None
        for i in range(retries):
            try:
                if dest.exists():
                    try:
                        os.chmod(dest, 0o666)
                    except Exception:
                        pass
                    try:
                        dest.unlink()
                    except Exception:
                        pass

                with open(src, "rb") as rf, open(tmp, "wb") as wf:
                    shutil.copyfileobj(rf, wf, length=1024 * 1024)
                os.replace(tmp, dest)
                return
            except PermissionError as e:
                last = e
                time.sleep(0.35 * (i + 1))
            except Exception as e:
                last = e
                time.sleep(0.20 * (i + 1))
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
        raise RuntimeError(f"Permission/IO error writing {dest.name}: {last}")

    def _best_effort_copy(src: Path, dest: Path, label: str) -> bool:
        try:
            _atomic_copy(src, dest)
            return True
        except Exception as e:
            log(f"[zimage-gguf]   warn: could not write {label} ({dest}): {e}")
            return False

    exe = _find_sdcpp_exe(extracted)
    if not exe:
        exes = [str(p.relative_to(extracted)) for p in extracted.rglob("*.exe")]
        raise RuntimeError("sd-cli.exe (or sd.exe) not found in extracted archive. Found exes: " + ", ".join(exes[:50]))
    log(f"[zimage-gguf] sdcpp exe: {exe}")

    _atomic_copy(exe, bin_dir / "sd-cli.exe")
    if not (bin_dir / "sd.exe").exists():
        _best_effort_copy(exe, bin_dir / "sd.exe", "sd.exe")

    exe_dir = exe.parent
    dlls = list(exe_dir.glob("*.dll"))
    if not dlls:
        dlls = list(extracted.rglob("*.dll"))

    for d in dlls:
        _best_effort_copy(d, bin_dir / d.name, f"dll {d.name}")

    stabledll = bin_dir / "stable-diffusion.dll"
    if not stabledll.exists():
        for d in extracted.rglob("stable-diffusion.dll"):
            _best_effort_copy(d, stabledll, "stable-diffusion.dll")
            break

    if not stabledll.exists():
        raise RuntimeError("stable-diffusion.dll not found after extraction/copy. Antivirus may have quarantined it.")

    diffdll = bin_dir / "diffusers.dll"
    if not diffdll.exists() and stabledll.exists():
        try:
            shutil.copy2(stabledll, diffdll)
            log("[zimage-gguf] ok: created diffusers.dll from stable-diffusion.dll")
        except Exception as e:
            log(f"[zimage-gguf] warn: could not create diffusers.dll: {e}")

def _nvidia_present() -> bool:
    try:
        import subprocess
        p = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
        return p.returncode == 0
    except Exception:
        return False

def _score_asset(name: str, want_cuda: bool):
    n = name.lower()
    if not n.endswith(".zip"):
        return (999, 999, len(n))
    score = 100
    if "win" in n and "x64" in n:
        score -= 30
    if "sd" in n:
        score -= 10
    if "sd-bin" in n:
        score -= 20
    # Prefer CUDA builds when an NVIDIA GPU is present, but do NOT hard-require them.
    # Some releases may not provide a working CUDA zip (or it might be temporarily broken),
    # so we keep Vulkan/CPU zips as fallback.
    if "cu12" in n or "cuda" in n or "cudart" in n or "cublas" in n:
        score += (-35 if want_cuda else +25)
    if "vulkan" in n:
        score += (-25 if not want_cuda else +10)
    if "avx2" in n:
        score -= 8
    if "src" in n or "source" in n:
        score += 50
    return (score, len(n))



def _select_and_install_sdcpp(latest: dict, want_cuda: bool, tmp: Path, bin_dir: Path) -> dict:
    assets = latest.get("assets") or []
    if not assets:
        raise RuntimeError("No assets in stable-diffusion.cpp latest release JSON.")

    sorted_assets = sorted(assets, key=lambda a: _score_asset(a.get("name", ""), want_cuda=want_cuda))

    tried = 0
    saw_debug_runtime = False
    last_reason = None
    for a in sorted_assets:
        name = a.get("name", "")
        url = a.get("browser_download_url", "")
        if not name or not url:
            continue

        ln = name.lower()
        if "win" not in ln or "x64" not in ln or not ln.endswith(".zip"):
            continue

        if _is_likely_debug_asset_name(name):
            continue

        # Don't hard-filter by backend type. Scoring handles preference, but we keep fallbacks.

        tried += 1
        log(f"[zimage-gguf] trying sd.cpp asset: {name}")
        zip_path = tmp / name
        download(url, zip_path, retries=3)

        exe_entries = _zip_exe_entries(zip_path)
        if not exe_entries:
            log(f"[zimage-gguf]   no .exe inside {name}, skipping.")
            continue

        log(f"[zimage-gguf]   exe entries in zip: {', '.join(exe_entries[:8])}{' ...' if len(exe_entries) > 8 else ''}")

        extract_dir = tmp / "sdcpp_extract"
        try:
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
        except Exception:
            pass

        log(f"[zimage-gguf] extracting: {name}")
        extract_zip(zip_path, extract_dir)

        exe = _find_sdcpp_exe(extract_dir)
        if not exe:
            log(f"[zimage-gguf]   no sd executable found after extract, skipping.")
            continue

        ok, reason = _probe_sdcpp_exe(exe)
        if not ok:
            log(f"[zimage-gguf]   sd executable not usable ({reason}); trying next asset.")
            last_reason = reason
            if "debug runtime" in (reason or ""):
                saw_debug_runtime = True
            continue

        try:
            log("[zimage-gguf] Installing sdcpp binaries...")
            _copy_sdcpp_bins(extract_dir, bin_dir)
            log(f"[zimage-gguf] installed from: {name}")
            return {
                "asset_name": name,
                "asset_url": url,
            }
        except Exception as e:
            log(f"[zimage-gguf]   install failed for {name}: {e}")
            continue

    # Raise a detailed error that the caller can turn into a user-facing message.
    if tried == 0:
        raise RuntimeError("No matching Windows x64 .zip assets were found in the stable-diffusion.cpp latest release.")
    if saw_debug_runtime:
        raise RuntimeError("stable-diffusion.cpp binary requires MSVC *Debug* runtime (VCRUNTIME140D/MSVCP140D/ucrtbased).")
    raise RuntimeError(f"Could not find a usable stable-diffusion.cpp Windows x64 binary zip. Tried {tried} candidate assets. Last: {last_reason}")


def _print_sdcpp_requirements_hint() -> None:
    log("[zimage-gguf] ---")
    log("[zimage-gguf] This PC is missing runtime files needed to run the GGUF backend (sd-cli.exe).")
    log("[zimage-gguf] Fix options:")
    log("[zimage-gguf]  1) Install Microsoft Visual C++ Redistributable 2015–2022 (x64).")
    log("[zimage-gguf]     (Most common missing: VCRUNTIME140.dll / MSVCP140.dll)")
    log("[zimage-gguf]  2) If the error mentions ...140D.dll or ucrtbased.dll, that is a DEBUG build.")
    log("[zimage-gguf]     Install Visual Studio 2022 (or Build Tools) with 'Desktop development with C++' OR use a Release sd-cli build.")
    log("[zimage-gguf]  3) After installing runtimes, re-run Optional Installs: Z-Image Turbo GGUF (Q4/Q5/Q6/Q8).")
    log("[zimage-gguf] ---")

def auto_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]

def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="zimage_gguf_install.py")
    # Back-compat: allow positional root + target (as older .bat wrapper passed).
    p.add_argument("root", nargs="?", default=None, help="FrameVision root folder (optional)")
    p.add_argument("target", nargs="?", default=None, help="Target install folder (optional)")
    p.add_argument("--mode", default="gguf", choices=["gguf", "safetensors", "diffusers"],
               help="Install mode: 'gguf' installs GGUF diffusion+text+VAE + sd.cpp binaries; "
                    "'safetensors' installs ComfyUI-style safetensors (diffusion+text encoder+VAE); "
                    "'diffusers' downloads the full Diffusers snapshot (model_index.json + subfolders).")
    p.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"],
                   help="Safetensors precision to download (only used when --mode safetensors).")
    p.add_argument("--safe-repo", default=SAFE_REPO_DEFAULT,
                   help="Hugging Face repo for safetensors downloads (only used when --mode safetensors).")

    p.add_argument("--diffusers-repo", default=DIFFUSERS_REPO_DEFAULT,
                   help="Hugging Face repo for Diffusers snapshot download (only used when --mode diffusers).")
    p.add_argument("--diffusers-revision", default=DIFFUSERS_REVISION_DEFAULT,
                   help="Revision/branch for Diffusers snapshot download (only used when --mode diffusers).")
    p.add_argument("--quant", default="Q5_0", choices=list(ZIMAGE_DIFFUSION_URLS.keys()),
                   help="Z-Image Turbo diffusion GGUF quant to download")
    p.add_argument("--text-quant", default=None, choices=list(QWEN_TEXT_URLS.keys()),
                   help="Override Qwen text encoder quant (GGUF). If omitted, --match-text-quant selects one.")
    p.add_argument("--match-text-quant", default="1", choices=["0", "1"],
                   help="If 1, pick a Qwen text quant that roughly matches the diffusion quant.")
    return p.parse_args(argv)


def _resolve_root_and_target(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.root).resolve() if args.root else auto_root_from_script()
    if args.target:
        target = Path(args.target).resolve()
    else:
        # Defaults differ by mode to avoid mixing installs.
        mode = getattr(args, "mode", "gguf")
        if mode == "diffusers":
            # Diffusers snapshot layout (model_index.json + subfolders)
            target = (root / "models" / "Z-Image-Turbo").resolve()
        elif mode == "safetensors":
            # ComfyUI-style safetensors layout (diffusion_models/text_encoders/vae)
            target = (root / "models" / "Z-Image-Turbo").resolve()
        else:
            # GGUF backend layout
            target = (root / "models" / "Z-Image-Turbo GGUF").resolve()
    return root, target





def _install_safetensors(precision: str, target: Path, repo: str) -> None:
    """Download ComfyUI-style safetensors weights (diffusion + text encoder + VAE)."""
    target.mkdir(parents=True, exist_ok=True)

    diff_dir = target / "diffusion_models"
    te_dir = target / "text_encoders"
    vae_dir = target / "vae"
    diff_dir.mkdir(parents=True, exist_ok=True)
    te_dir.mkdir(parents=True, exist_ok=True)
    vae_dir.mkdir(parents=True, exist_ok=True)

    base = SAFE_BASE_URL.format(repo=repo).rstrip("/") + "/"

    # Diffusion model
    diff_name = SAFE_DIFFUSION_FILES.get(precision)
    if not diff_name:
        raise RuntimeError(f"Unknown precision: {precision}")
    diff_dst = diff_dir / diff_name
    download(base + diff_name, diff_dst)

    # Text encoder (repo provides bf16 only; works across diffusion precisions in most setups)
    te_dst = te_dir / Path(SAFE_TEXT_ENCODER_FILE).name
    download(base + SAFE_TEXT_ENCODER_FILE, te_dst)

    # VAE (download both, pick whichever your runtime prefers)
    for rel in SAFE_VAE_FILES:
        vae_dst = vae_dir / Path(rel).name
        download(base + rel, vae_dst)

    (target / "_safetensors_sources.txt").write_text(
        "\n".join(
            [
                f"Repo: {repo}",
                f"Precision: {precision}",
                f"Diffusion: {base + diff_name}",
                f"Text encoder: {base + SAFE_TEXT_ENCODER_FILE}",
                "VAE:",
                *[f"  {base + v}" for v in SAFE_VAE_FILES],
                "",
            ]
        ),
        encoding="utf-8",
    )

def main() -> int:
    args = _parse_args(sys.argv[1:])
    root, target = _resolve_root_and_target(args)

    if getattr(args, "mode", "gguf") == "diffusers":
        log(f"[zimage-diff] Root  : {root}")
        log(f"[zimage-diff] Target: {target}")
        _install_diffusers_snapshot(args.diffusers_repo, args.diffusers_revision, target)
        log("[zimage-diff] done")
        return 0

    if getattr(args, "mode", "gguf") == "safetensors":
        log(f"[zimage-safe] Root  : {root}")
        log(f"[zimage-safe] Target: {target}")
        log(f"[zimage-safe] Repo  : {args.safe_repo}")
        _install_safetensors(args.precision, target, args.safe_repo)
        log("[zimage-safe] ok")
        return 0

    shared_bin = _shared_sdcli_dir(root)
    shared_bin.mkdir(parents=True, exist_ok=True)

    tmp = shared_bin / "_tmp_sdcpp"
    tmp.mkdir(parents=True, exist_ok=True)

    log(f"[zimage-gguf] Root  : {root}")
    log(f"[zimage-gguf] Target: {target}")
    log(f"[zimage-gguf] Shared bin: {shared_bin}")

    # Shared sd-cli/dlls: if already installed, don't download again.
    latest = None
    if _sdcli_present(shared_bin):
        log("[zimage-gguf] sdcpp already present in shared bin; skipping download")
    else:
        api = SDCPP_LATEST_API
        latest = json.loads(http_get(api).decode("utf-8", errors="replace"))

    want_cuda = _nvidia_present()
    log(f"[zimage-gguf] NVIDIA detected: {want_cuda}")

    sdcpp_meta = None

    sdcpp_ok = True
    try:
        sdcpp_meta = _select_and_install_sdcpp(latest, want_cuda=want_cuda, tmp=tmp, bin_dir=shared_bin)
    except Exception as e:
        sdcpp_ok = False
        log(f"[zimage-gguf] WARN: GGUF backend (sd-cli.exe) could not be installed: {e}")
        _print_sdcpp_requirements_hint()
        # Continue anyway: model files can still be downloaded.


    # Record exactly which sd-cli.exe binary was installed (important when using the moving 'latest' URL).
    if sdcpp_ok:
        try:
            exe_path = (target / "bin" / "sd-cli.exe")
            if exe_path.exists():
                build_text = _collect_sdcpp_build_text(exe_path)
                info_path = (root / "presets" / "info" / "sd_cli_build_info.txt").resolve()
                header = "\n".join(
                    [
                        "stable-diffusion.cpp (sd-cli.exe) install receipt",
                        f"installed_at_utc: {_utc_now_iso()}",
                        f"release_tag: {latest.get('tag_name', '')}",
                        f"release_published_at: {latest.get('published_at', '')}",
                        f"release_html_url: {latest.get('html_url', '')}",
                        f"asset_name: {(sdcpp_meta or {}).get('asset_name', '')}",
                        f"asset_url: {(sdcpp_meta or {}).get('asset_url', '')}",
                        f"sha256(sd-cli.exe): {_sha256_file(exe_path)}",
                        "",
                    ]
                )
                _safe_write_text(info_path, header + build_text)

                _update_3rd_party_licenses_json(
                    root,
                    item_id="sd_cli_stable_diffusion_cpp",
                    updates={
                        "installed_at_utc": _utc_now_iso(),
                        "installed_release_tag": latest.get("tag_name", ""),
                        "installed_release_published_at": latest.get("published_at", ""),
                        "installed_release_html_url": latest.get("html_url", ""),
                        "installed_asset_name": (sdcpp_meta or {}).get("asset_name", ""),
                        "installed_asset_url": (sdcpp_meta or {}).get("asset_url", ""),
                        "installed_sdcli_sha256": _sha256_file(exe_path),
                        "build_info_path": str(info_path.relative_to(root)).replace('\\', '/'),
                    },
                )
                log(f"[zimage-gguf] wrote sd-cli build receipt: {info_path}")
            else:
                log("[zimage-gguf] warn: sd-cli.exe not found after install (no receipt written)")
        except Exception as e:
            log(f"[zimage-gguf] warn: could not write sd-cli receipt: {e}")

    log("[zimage-gguf] Downloading model files...")

    # Diffusion model (Z-Image Turbo GGUF)
    diff_quant = args.quant
    zimg_name = f"z_image_turbo-{diff_quant}.gguf"
    zimg = target / zimg_name
    download(ZIMAGE_DIFFUSION_URLS[diff_quant], zimg)

    # Text encoder (Qwen3 4B GGUF)
    if args.text_quant:
        qwen_quant = args.text_quant
    elif args.match_text_quant == "1":
        qwen_quant = MATCHING_QWEN_FOR_DIFF.get(diff_quant, "Q4_K_M")
    else:
        qwen_quant = "Q4_K_M"

    qwen_name = f"Qwen3-4B-Instruct-2507-{qwen_quant}.gguf"
    qwen = target / qwen_name
    download_any(QWEN_TEXT_URLS[qwen_quant], qwen)

    vae = target / "ae.safetensors"
    download(VAE_URL, vae)

    (target / "_gguf_sources.txt").write_text(
        "\n".join(
            [
                f"Selected diffusion: {diff_quant} -> {ZIMAGE_DIFFUSION_URLS[diff_quant]}",
                f"Selected Qwen text: {qwen_quant} -> {QWEN_TEXT_URLS[qwen_quant][0]}",
                f"VAE: {VAE_URL}",
                f"sdcpp latest API: {SDCPP_LATEST_API}",
                "",
                "All diffusion quants:",
                *[f"  {k}: {v}" for k, v in ZIMAGE_DIFFUSION_URLS.items()],
                "",
                "All Qwen text quants (primary URL):",
                *[f"  {k}: {v[0]}" for k, v in QWEN_TEXT_URLS.items()],
                "",
            ]
        ),
        encoding="utf-8",
    )

    if not sdcpp_ok:
        (target / "_GGUF_BACKEND_MISSING.txt").write_text(
            "\n".join(
                [
                    "The GGUF model files were downloaded, but the GGUF backend (sd-cli.exe) could not run on this PC.",
                    "Install Microsoft Visual C++ Redistributable 2015–2022 (x64).",
                    "If the missing DLL names end with 'D' (VCRUNTIME140D/MSVCP140D/ucrtbased),", 
                    "then the backend binary is a Debug build and requires Visual Studio 2022 C++ runtime.",
                    "After installing runtimes, re-run Optional Installs: Z-Image Turbo GGUF.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        log("[zimage-gguf] DONE (models downloaded; backend missing)")
        return 0

    log("[zimage-gguf] ok")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[zimage-gguf] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1)
