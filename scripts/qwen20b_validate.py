# scripts/qwen20b_validate.py
# Usage: python -u scripts\qwen20b_validate.py ".\models\Qwen20B"
import os, sys

def validate(path: str) -> int:
    req = [
        "qwen_image_20B_quanto_bf16_int8.safetensors",
        "qwen_vae.safetensors",
        "qwen_image_20B.json",
        "qwen_vae_config.json",
        "qwen_scheduler_config.json",
    ]
    missing = [f for f in req if not os.path.exists(os.path.join(path, f))]
    pipe_py = os.path.join(path, "qwen", "pipeline_qwenimage.py")
    if missing or not os.path.exists(pipe_py):
        if missing:
            print("[qwen20b] missing:", ", ".join(missing))
        else:
            print("[qwen20b] pipeline_qwenimage.py not found")
        return 2
    print("[qwen20b] OK at", path)
    return 0

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else r".\models\Qwen20B"
    sys.exit(validate(target))
