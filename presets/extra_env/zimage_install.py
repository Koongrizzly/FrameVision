#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import subprocess
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

UA = "FrameVision-ZImage-Installer/2.0"

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

def download(url: str, dst: Path, retries: int = 6, label: str = "[zimage-gguf]") -> None:
    """Robust downloader with resume + atomic finalization.

    Writes to <dst>.part and renames to <dst> when complete.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If a complete file exists, keep it.
    if dst.exists() and dst.stat().st_size > 0:
        log(f"{label} exists: {dst.name}")
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
                log(f"{label} resume: {u} (+{existing} bytes)")
            else:
                log(f"{label} download: {u}")

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
                            log(f"{label}   {total / (1024*1024):.0f} MiB")

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

def download_any(urls: list[str], dst: Path, label: str = "[zimage-gguf]") -> None:
    last = None
    for u in urls:
        try:
            download(u, dst, retries=2, label=label)
            return
        except Exception as e:
            last = e
            log(f"{label}   mirror failed: {u} ({type(e).__name__})")
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
        download(base + rel, dst, label="[zimage-diff]")

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

DEFAULT_PIP_REQUIREMENTS = [
    # Keep requirements embedded so this single installer script is self-contained.
    "git+https://github.com/huggingface/diffusers.git",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "einops",
    "PySide6",
    "Pillow",
    "tqdm",
]

REQUIRED_IMPORT_CHECKS = [
    ("diffusers", "diffusers"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors", "safetensors"),
    ("einops", "einops"),
    ("PySide6", "PySide6"),
    ("PIL", "Pillow"),
    ("tqdm", "tqdm"),
]


def _env_dir(root: Path) -> Path:
    """Shared image-model environment location used by FrameVision."""
    return (root / "environments" / ".images_models").resolve()


def _env_python_candidates(root: Path) -> list[Path]:
    env = _env_dir(root)
    return [
        # Conda envs on Windows place python.exe directly in the env root.
        env / "python.exe",
        # Keep venv-style fallbacks for older/migrated installs.
        env / "Scripts" / "python.exe",
        env / "scripts" / "python.exe",
        env / "bin" / "python",
        env / "bin" / "python3",
    ]


def _env_python(root: Path) -> Path | None:
    for c in _env_python_candidates(root):
        try:
            if c.exists() and c.is_file():
                return c
        except Exception:
            pass
    return None


def _run_cmd(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    log("[zimage] $ " + " ".join(str(x) for x in cmd))
    kwargs = dict(cwd=str(cwd) if cwd else None, env=env)
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    p = subprocess.run([str(x) for x in cmd], **kwargs)
    if p.returncode != 0:
        raise RuntimeError(f"command failed with exit code {p.returncode}: {' '.join(str(x) for x in cmd)}")


def _check_import(py: Path, module: str) -> bool:
    try:
        kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.run([str(py), "-c", f"import {module}"], **kwargs)
        return p.returncode == 0
    except Exception:
        return False


def _find_conda_exe(root: Path) -> str | None:
    """Find a conda executable without relying on PATH only."""
    candidates: list[str] = []
    for env_name in ("CONDA_EXE", "MAMBA_EXE"):
        try:
            val = os.environ.get(env_name)
            if val:
                candidates.append(val)
        except Exception:
            pass

    try:
        import shutil as _shutil
        for name in ("conda.exe", "conda", "mamba.exe", "mamba"):
            found = _shutil.which(name)
            if found:
                candidates.append(found)
    except Exception:
        pass

    # Common local/miniconda/portable locations near FrameVision.
    for c in [
        root / "conda" / "Scripts" / "conda.exe",
        root / "miniconda3" / "Scripts" / "conda.exe",
        root / "tools" / "conda" / "Scripts" / "conda.exe",
        root / "tools" / "miniconda3" / "Scripts" / "conda.exe",
        Path.home() / "miniconda3" / "Scripts" / "conda.exe",
        Path.home() / "anaconda3" / "Scripts" / "conda.exe",
    ]:
        candidates.append(str(c))

    seen = set()
    for c in candidates:
        try:
            if not c or c in seen:
                continue
            seen.add(c)
            p = Path(c)
            if p.exists() and p.is_file():
                return str(p)
        except Exception:
            continue
    return None


def _create_conda_env(root: Path, env: Path) -> bool:
    """Create the shared image-model env with conda when available."""
    conda = _find_conda_exe(root)
    if not conda:
        return False
    env.parent.mkdir(parents=True, exist_ok=True)
    log(f"[zimage-env] creating shared conda environment: {env}")
    _run_cmd([conda, "create", "-y", "-p", str(env), "python=3.11", "pip"], cwd=root)
    return True


def _install_python_env(root: Path, args: argparse.Namespace) -> Path:
    """Create/update root/environments/.images_models and install Z-Image runtime deps."""
    env = _env_dir(root)
    py = _env_python(root)

    old_envs = [
        (root / ".zimage_env").resolve(),
        (root / "environments" / ".zimage_env").resolve(),
    ]
    for old_env in old_envs:
        if old_env.exists() and old_env != env:
            log(f"[zimage-env] note: old private env exists and is no longer used: {old_env}")
    log(f"[zimage-env] shared env location: {env}")

    if py is None:
        created_with_conda = _create_conda_env(root, env)
        if not created_with_conda:
            env.parent.mkdir(parents=True, exist_ok=True)
            log("[zimage-env] conda was not found; falling back to venv so install can still continue")
            log(f"[zimage-env] creating virtual environment: {env}")
            _run_cmd([sys.executable, "-m", "venv", str(env)], cwd=root)
        py = _env_python(root)
        if py is None:
            tried = ", ".join(str(c) for c in _env_python_candidates(root))
            raise RuntimeError(f"Z-Image shared env was created but python.exe was not found. Tried: {tried}")
    else:
        log(f"[zimage-env] environment already exists: {env}")

    log(f"[zimage-env] python: {py}")

    if not getattr(args, "skip_deps", False):
        _run_cmd([str(py), "-m", "pip", "install", "--upgrade", "pip"], cwd=root)

        if getattr(args, "skip_torch", False):
            log("[zimage-env] skipping torch install by request")
        elif _check_import(py, "torch") and not getattr(args, "force_deps", False):
            log("[zimage-env] torch already imports; skipping torch install")
        else:
            torch_index = getattr(args, "torch_index_url", "") or "https://download.pytorch.org/whl/cu128"
            log(f"[zimage-env] installing CUDA PyTorch from: {torch_index}")
            if "cu128" in str(torch_index).lower():
                torch_pkgs = ["torch==2.8.0+cu128", "torchvision==0.23.0+cu128", "torchaudio==2.8.0+cu128"]
            else:
                torch_pkgs = ["torch", "torchvision", "torchaudio"]
            _run_cmd([str(py), "-m", "pip", "install", *torch_pkgs, "--index-url", torch_index], cwd=root)

        missing = []
        for mod, pkg in REQUIRED_IMPORT_CHECKS:
            if getattr(args, "force_deps", False) or not _check_import(py, mod):
                missing.append(pkg)
        if missing:
            log("[zimage-env] installing missing Python packages: " + ", ".join(missing))
            # Install the embedded full requirements list so dependency versions resolve together.
            _run_cmd([str(py), "-m", "pip", "install", *DEFAULT_PIP_REQUIREMENTS], cwd=root)
        else:
            log("[zimage-env] Python requirements already import; skipping requirements install")

    marker = env / "FRAMEVISION_ZIMAGE_ENV.txt"
    try:
        marker.write_text(
            "FrameVision shared image-model environment\n"
            f"location={env}\n"
            "created_by=presets/extra_env/zimage_install.py\nshared_for=Z-Image and compatible image tools\n"
            f"updated_at_utc={_utc_now_iso()}\n",
            encoding="utf-8",
        )
    except Exception as e:
        log(f"[zimage-env] warn: could not write marker: {e}")

    return py


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="zimage_install.py")
    # Optional root is kept for FrameVision Optional Installs and manual use.
    p.add_argument("root", nargs="?", default=None, help="FrameVision root folder. Defaults to two folders above this script.")
    p.add_argument("target", nargs="?", default=None, help="Optional target model folder. Usually leave empty.")
    p.add_argument("--mode", default="env", choices=["env", "fp16", "diffusers", "gguf", "safetensors", "all"],
                   help="Install mode. env=environment only; fp16/diffusers=full Diffusers model folder; safetensors=single-file ComfyUI-style BF16/FP16/FP32 files; gguf=GGUF model/backend; all=env+Diffusers+GGUF.")
    p.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"],
                   help="Preferred runtime precision for the full Diffusers model, or file precision for --mode safetensors. auto prefers BF16 when the active GPU/PyTorch setup supports it, otherwise FP16.")
    p.add_argument("--safe-repo", default=SAFE_REPO_DEFAULT,
                   help="Hugging Face repo for safetensors downloads (only used when --mode safetensors).")
    p.add_argument("--diffusers-repo", default=DIFFUSERS_REPO_DEFAULT,
                   help="Hugging Face repo for Diffusers snapshot download (used by --mode fp16/diffusers/all).")
    p.add_argument("--diffusers-revision", default=DIFFUSERS_REVISION_DEFAULT,
                   help="Revision/branch for Diffusers snapshot download.")
    p.add_argument("--quant", default="Q5_0", choices=list(ZIMAGE_DIFFUSION_URLS.keys()),
                   help="Z-Image Turbo diffusion GGUF quant to download.")
    p.add_argument("--text-quant", default=None, choices=list(QWEN_TEXT_URLS.keys()),
                   help="Override Qwen text encoder quant (GGUF). If omitted, --match-text-quant selects one.")
    p.add_argument("--match-text-quant", default="1", choices=["0", "1"],
                   help="If 1, pick a Qwen text quant that roughly matches the diffusion quant.")
    p.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu128",
                   help="PyTorch wheel index for CUDA torch installs.")
    p.add_argument("--skip-deps", action="store_true", help="Create/find env but do not install or update dependencies.")
    p.add_argument("--skip-torch", action="store_true", help="Do not install torch automatically.")
    p.add_argument("--force-deps", action="store_true", help="Reinstall torch/requirements even when imports work.")
    p.add_argument("--force-model", action="store_true", help="Do not skip model download just because marker files already exist.")
    return p.parse_args(argv)


def _resolve_root_and_target(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.root).resolve() if args.root else auto_root_from_script()
    if args.target:
        target = Path(args.target).resolve()
    else:
        # Defaults differ by mode to avoid mixing installs.
        mode = getattr(args, "mode", "env")
        if mode in ("fp16", "diffusers"):
            # Diffusers snapshot layout (model_index.json + subfolders)
            target = (root / "models" / "Z-Image-Turbo").resolve()
        elif mode == "safetensors":
            # ComfyUI-style safetensors layout (diffusion_models/text_encoders/vae)
            target = (root / "models" / "Z-Image-Turbo").resolve()
        elif mode == "all":
            # The all mode installs both; individual stage functions resolve their own target.
            target = (root / "models" / "Z-Image-Turbo").resolve()
        else:
            # GGUF backend layout. For env-only this is unused but harmless.
            target = (root / "models" / "Z-Image-Turbo GGUF").resolve()
    return root, target



def _diffusers_target(root: Path) -> Path:
    return (root / "models" / "Z-Image-Turbo").resolve()


def _gguf_target(root: Path) -> Path:
    return (root / "models" / "Z-Image-Turbo GGUF").resolve()



def _run_python_probe(py: Path, code: str, timeout: int = 45) -> tuple[int, str, str]:
    """Run a small Python probe through the Z-Image env and return rc/stdout/stderr."""
    try:
        kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.run([str(py), "-c", code], **kwargs)
        out = (p.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (p.stderr or b"").decode("utf-8", errors="replace").strip()
        return int(p.returncode), out, err
    except Exception as e:
        return 1, "", str(e)


def _nvidia_gpu_names() -> list[str]:
    """Best-effort GPU names from nvidia-smi, used only as a fallback if torch probing fails."""
    names: list[str] = []
    try:
        kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            **kwargs,
        )
        if p.returncode == 0:
            out = (p.stdout or b"").decode("utf-8", errors="replace")
            names = [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        names = []
    return names


def _gpu_name_suggests_bf16(name: str) -> bool:
    """Fallback heuristic for common NVIDIA GPUs with BF16-capable Tensor Cores."""
    n = (name or "").lower()
    bf16_markers = (
        "rtx 30", "rtx 40", "rtx 50",
        "rtx a", "a100", "a10", "a16", "a30", "a40", "l4", "l40",
        "h100", "h200", "b100", "b200",
    )
    return any(m in n for m in bf16_markers)


def _resolve_safetensors_precision(precision: str, root: Path, env_py: Path | None = None, label: str = "[zimage-safe]") -> tuple[str, str]:
    """Resolve auto/fp16/bf16/fp32 into an actual safetensors precision and a readable reason."""
    requested = (precision or "auto").strip().lower()
    if requested in ("fp16", "bf16", "fp32"):
        return requested, f"manual {requested.upper()} selected"

    if requested != "auto":
        return "fp16", f"unknown precision {precision!r}; fallback FP16"

    py = env_py or _env_python(root)
    if py is not None:
        probe = """
import json
result = {"cuda": False, "bf16": False, "name": "", "capability": "", "error": ""}
try:
    import torch
    result["cuda"] = bool(torch.cuda.is_available())
    if result["cuda"]:
        result["name"] = str(torch.cuda.get_device_name(0))
        try:
            result["capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
        except Exception:
            result["capability"] = ""
        try:
            result["bf16"] = bool(torch.cuda.is_bf16_supported())
        except Exception as e:
            result["error"] = "is_bf16_supported failed: " + str(e)
except Exception as e:
    result["error"] = str(e)
print(json.dumps(result))
"""
        rc, out, err = _run_python_probe(py, probe)
        if rc == 0 and out:
            try:
                info = json.loads(out.splitlines()[-1])
                name = str(info.get("name") or "GPU")
                cap = str(info.get("capability") or "")
                if bool(info.get("cuda")) and bool(info.get("bf16")):
                    cap_txt = f" compute capability {cap}" if cap else ""
                    return "bf16", f"Auto selected BF16 for {name}{cap_txt}"
                if bool(info.get("cuda")):
                    extra = str(info.get("error") or "")
                    if extra:
                        extra = f" ({extra})"
                    return "fp16", f"Auto selected FP16 for {name}; BF16 not reported as supported{extra}"
                return "fp16", "Auto selected FP16 because CUDA is not available in the Z-Image environment"
            except Exception as e:
                log(f"{label} warn: could not parse torch precision probe: {e}")
        else:
            msg = (err or out or "torch probe failed").strip()
            if msg:
                log(f"{label} warn: torch precision probe failed: {msg}")

    names = _nvidia_gpu_names()
    if names:
        for name in names:
            if _gpu_name_suggests_bf16(name):
                return "bf16", f"Auto selected BF16 from GPU name fallback: {name}"
        return "fp16", "Auto selected FP16 from GPU name fallback; BF16 support was not detected"

    return "fp16", "Auto selected FP16 because no CUDA/BF16 support could be confirmed"


def _install_safetensors(precision: str, target: Path, repo: str, root: Path, env_py: Path | None = None) -> None:
    """Download ComfyUI-style safetensors weights (diffusion + text encoder + VAE)."""
    target.mkdir(parents=True, exist_ok=True)

    requested_precision = (precision or "auto").strip().lower()
    precision, precision_reason = _resolve_safetensors_precision(requested_precision, root, env_py, label="[zimage-safe]")
    log(f"[zimage-safe] Precision request : {requested_precision}")
    log(f"[zimage-safe] Precision selected: {precision.upper()} ({precision_reason})")

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
    download(base + diff_name, diff_dst, label="[zimage-safe]")

    # Text encoder (repo provides bf16 only; works across diffusion precisions in most setups)
    te_dst = te_dir / Path(SAFE_TEXT_ENCODER_FILE).name
    download(base + SAFE_TEXT_ENCODER_FILE, te_dst, label="[zimage-safe]")

    # VAE (download both, pick whichever your runtime prefers)
    for rel in SAFE_VAE_FILES:
        vae_dst = vae_dir / Path(rel).name
        download(base + rel, vae_dst, label="[zimage-safe]")

    (target / "_safetensors_sources.txt").write_text(
        "\n".join(
            [
                f"Repo: {repo}",
                f"Precision request: {requested_precision}",
                f"Precision selected: {precision}",
                f"Precision reason: {precision_reason}",
                f"Diffusion: {base + diff_name}",
                f"Text encoder: {base + SAFE_TEXT_ENCODER_FILE}",
                "VAE:",
                *[f"  {base + v}" for v in SAFE_VAE_FILES],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (target / "_active_precision.txt").write_text(
        f"{precision}\n{precision_reason}\n",
        encoding="utf-8",
    )


def _write_active_precision_marker(target: Path, precision: str, reason: str) -> None:
    """Record the precision the runtime should prefer for the Diffusers model."""
    try:
        target.mkdir(parents=True, exist_ok=True)
        (target / "_active_precision.txt").write_text(
            f"{precision}\n{reason}\n",
            encoding="utf-8",
        )
    except Exception as e:
        log(f"[zimage-diff] warn: could not write _active_precision.txt: {e}")


def _diffusers_precision_prepare(root: Path, target: Path, requested_precision: str, env_py: Path | None) -> tuple[str, str]:
    """Resolve and record the BF16/FP16 runtime preference used by zimage_cli.py."""
    req = (requested_precision or "auto").strip().lower()
    precision, reason = _resolve_safetensors_precision(req, root, env_py, label="[zimage-diff]")
    log(f"[zimage-diff] Precision request : {req}")
    log(f"[zimage-diff] Runtime precision : {precision.upper()} ({reason})")
    _write_active_precision_marker(target, precision, reason)
    return precision, reason


def _install_diffusers_full(root: Path, target: Path, repo: str, revision: str, requested_precision: str, env_py: Path | None, *, force_model: bool = False) -> None:
    """Install/repair the full Diffusers repo layout and record Auto BF16/FP16 preference.

    Important: this path is what helpers/zimage_cli.py uses with DiffusionPipeline.from_pretrained(...).
    It must contain model_index.json and the repo subfolders. The old safetensors-only path is
    not enough for the current helper.
    """
    _diffusers_precision_prepare(root, target, requested_precision, env_py)

    if force_model:
        log("[zimage-diff] force-model enabled; checking/redownloading repo files")
    else:
        log("[zimage-diff] checking full Diffusers repo layout and downloading missing files")

    _install_diffusers_snapshot(repo, revision, target)

    mi = target / "model_index.json"
    if not mi.exists():
        raise RuntimeError(f"Diffusers install incomplete: missing {mi}")

def _install_gguf(args: argparse.Namespace, root: Path, target: Path) -> int:
    shared_bin = _shared_sdcli_dir(root)
    shared_bin.mkdir(parents=True, exist_ok=True)

    tmp = shared_bin / "_tmp_sdcpp"
    tmp.mkdir(parents=True, exist_ok=True)

    log(f"[zimage-gguf] Root  : {root}")
    log(f"[zimage-gguf] Target: {target}")
    log(f"[zimage-gguf] Shared bin: {shared_bin}")

    latest = None
    sdcpp_meta = None
    sdcpp_ok = True

    # Shared sd-cli/dlls: if already installed, don't download again.
    if _sdcli_present(shared_bin):
        log("[zimage-gguf] sdcpp already present in shared bin; skipping download")
    else:
        try:
            latest = json.loads(http_get(SDCPP_LATEST_API).decode("utf-8", errors="replace"))
            want_cuda = _nvidia_present()
            log(f"[zimage-gguf] NVIDIA detected: {want_cuda}")
            sdcpp_meta = _select_and_install_sdcpp(latest, want_cuda=want_cuda, tmp=tmp, bin_dir=shared_bin)
        except Exception as e:
            sdcpp_ok = False
            log(f"[zimage-gguf] WARN: GGUF backend (sd-cli.exe) could not be installed: {e}")
            _print_sdcpp_requirements_hint()
            # Continue anyway: model files can still be downloaded.

    # Record exactly which sd-cli.exe binary was installed (important when using the moving 'latest' URL).
    if sdcpp_ok:
        try:
            exe_path = shared_bin / "sd-cli.exe"
            if exe_path.exists():
                build_text = _collect_sdcpp_build_text(exe_path)
                info_path = (root / "presets" / "info" / "sd_cli_build_info.txt").resolve()
                header = "\n".join(
                    [
                        "stable-diffusion.cpp (sd-cli.exe) install receipt",
                        f"installed_at_utc: {_utc_now_iso()}",
                        f"release_tag: {(latest or {}).get('tag_name', '')}",
                        f"release_published_at: {(latest or {}).get('published_at', '')}",
                        f"release_html_url: {(latest or {}).get('html_url', '')}",
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
                        "installed_release_tag": (latest or {}).get("tag_name", ""),
                        "installed_release_published_at": (latest or {}).get("published_at", ""),
                        "installed_release_html_url": (latest or {}).get("html_url", ""),
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

    target.mkdir(parents=True, exist_ok=True)
    log("[zimage-gguf] Downloading model files...")

    diff_quant = args.quant
    zimg_name = f"z_image_turbo-{diff_quant}.gguf"
    zimg = target / zimg_name
    if zimg.exists() and zimg.stat().st_size > 0 and not getattr(args, "force_model", False):
        log(f"[zimage-gguf] exists: {zimg.name}")
    else:
        download(ZIMAGE_DIFFUSION_URLS[diff_quant], zimg, label="[zimage-gguf]")

    if args.text_quant:
        qwen_quant = args.text_quant
    elif args.match_text_quant == "1":
        qwen_quant = MATCHING_QWEN_FOR_DIFF.get(diff_quant, "Q4_K_M")
    else:
        qwen_quant = "Q4_K_M"

    qwen_name = f"Qwen3-4B-Instruct-2507-{qwen_quant}.gguf"
    qwen = target / qwen_name
    if qwen.exists() and qwen.stat().st_size > 0 and not getattr(args, "force_model", False):
        log(f"[zimage-gguf] exists: {qwen.name}")
    else:
        download_any(QWEN_TEXT_URLS[qwen_quant], qwen, label="[zimage-gguf]")

    vae = target / "ae.safetensors"
    if vae.exists() and vae.stat().st_size > 0 and not getattr(args, "force_model", False):
        log(f"[zimage-gguf] exists: {vae.name}")
    else:
        download(VAE_URL, vae, label="[zimage-gguf]")

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
                    "Install Microsoft Visual C++ Redistributable 2015-2022 (x64).",
                    "If the missing DLL names end with 'D' (VCRUNTIME140D/MSVCP140D/ucrtbased),",
                    "then the backend binary is a Debug build and requires Visual Studio 2022 C++ runtime.",
                    "After installing runtimes, re-run the Z-Image installer in GGUF mode.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        log("[zimage-gguf] DONE (models downloaded; backend missing)")
        return 0

    log("[zimage-gguf] ok")
    return 0


def main() -> int:
    args = _parse_args(sys.argv[1:])
    root, target = _resolve_root_and_target(args)
    mode = getattr(args, "mode", "env")
    if mode == "fp16":
        mode = "diffusers"

    log("===============================================")
    log("  FrameVision Z-Image unified installer")
    log(f"  Root: {root}")
    log(f"  Env : {_env_dir(root)}")
    log(f"  Mode: {mode}")
    if mode in ("safetensors", "diffusers"):
        log(f"  Precision request: {getattr(args, 'precision', 'auto')}")
    log("===============================================")

    # Every Z-Image backend is launched through the dedicated env python, so create it first.
    env_py = _install_python_env(root, args)

    if mode == "env":
        log("[zimage] environment install/update complete")
        return 0

    if mode == "all":
        diff_target = _diffusers_target(root)
        gguf_target = _gguf_target(root)
        log(f"[zimage-diff] Root  : {root}")
        log(f"[zimage-diff] Target: {diff_target}")
        _install_diffusers_full(
            root,
            diff_target,
            args.diffusers_repo,
            args.diffusers_revision,
            getattr(args, "precision", "auto"),
            env_py,
            force_model=getattr(args, "force_model", False),
        )
        return _install_gguf(args, root, gguf_target)

    if mode == "diffusers":
        target = _diffusers_target(root) if args.target is None else target
        log(f"[zimage-diff] Root  : {root}")
        log(f"[zimage-diff] Target: {target}")
        _install_diffusers_full(
            root,
            target,
            args.diffusers_repo,
            args.diffusers_revision,
            getattr(args, "precision", "auto"),
            env_py,
            force_model=getattr(args, "force_model", False),
        )
        log("[zimage-diff] done")
        return 0

    if mode == "safetensors":
        log(f"[zimage-safe] Root  : {root}")
        log(f"[zimage-safe] Target: {target}")
        log(f"[zimage-safe] Repo  : {args.safe_repo}")
        _install_safetensors(args.precision, target, args.safe_repo, root, env_py)
        log("[zimage-safe] ok")
        return 0

    if mode == "gguf":
        target = _gguf_target(root) if args.target is None else target
        return _install_gguf(args, root, target)

    raise RuntimeError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[zimage] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1)
