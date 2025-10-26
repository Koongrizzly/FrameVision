# Simple VRAM detector for Windows; prints "VRAM_MB=<int>" or empty
import subprocess, sys

def detect_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        # take the max across GPUs
        vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()]
        if not vals:
            return None
        return max(vals)
    except Exception:
        return None

if __name__ == "__main__":
    mb = detect_vram_mb()
    if mb is None:
        print("VRAM_MB=")
    else:
        print(f"VRAM_MB={mb}")
