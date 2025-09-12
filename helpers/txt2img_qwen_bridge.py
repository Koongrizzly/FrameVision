# helpers/txt2img_qwen_bridge.py
# Thin wrapper so existing UI can import a stable call for Qwen Image T2I.
from .qwen_image_runner import QwenImageRunner

_runner_cache = {}

def get_runner(model_dir: str, torch_dtype: str = "auto"):
    key = (model_dir, torch_dtype)
    if key not in _runner_cache:
        r = QwenImageRunner(model_dir=model_dir, torch_dtype=torch_dtype)
        r.enable_optimizations(enable_sequential_cpu_offload=False, vae_tiling=True)
        _runner_cache[key] = r
    return _runner_cache[key]

def generate_qwen_image(model_dir: str, prompt: str, negative: str = None,
                        steps: int = 30, scale: float = 5.0,
                        width: int = 1024, height: int = 1024,
                        seed: int = None, out_path: str = None) -> str:
    r = get_runner(model_dir)
    return r.generate(prompt=prompt, negative_prompt=negative, steps=steps,
                      guidance_scale=scale, width=width, height=height,
                      seed=seed, output_path=out_path)
