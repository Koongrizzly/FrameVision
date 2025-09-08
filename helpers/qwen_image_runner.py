# helpers/qwen_image_runner.py
# A lightweight Diffusers runner for Qwen-Image (20B). Keep one instance alive for multiple jobs.
from __future__ import annotations
import os, torch
from typing import Optional, Dict, Any
from diffusers import DiffusionPipeline
from PIL import Image

class QwenImageRunner:
    def __init__(self, model_dir: str, device: Optional[str] = None, torch_dtype: str = "auto"):
        self.model_dir = model_dir
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)
        self.dtype = dtype
        self.pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=dtype)
        self.pipe = self.pipe.to(self.device)

    def enable_optimizations(self, enable_sequential_cpu_offload: bool = False, vae_tiling: bool = True):
        if enable_sequential_cpu_offload:
            try: self.pipe.enable_sequential_cpu_offload()
            except Exception: pass
        if vae_tiling and hasattr(self.pipe, "vae"):
            try: self.pipe.enable_vae_tiling()
            except Exception: pass

    def generate(self, prompt: str, negative_prompt: Optional[str] = None, steps: int = 30,
                 guidance_scale: float = 5.0, width: int = 1024, height: int = 1024,
                 seed: Optional[int] = None, output_path: Optional[str] = None,
                 extra_kwargs: Optional[Dict[str, Any]] = None) -> str:
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
        kwargs = dict(prompt=prompt, negative_prompt=negative_prompt,
                      num_inference_steps=steps, guidance_scale=guidance_scale,
                      width=width, height=height, generator=g)
        if extra_kwargs: kwargs.update(extra_kwargs)
        images = self.pipe(**kwargs).images
        img: Image.Image = images[0]
        if output_path is None:
            os.makedirs("outputs", exist_ok=True)
            output_path = os.path.join("outputs", "qwen_image_out.png")
        img.save(output_path)
        return output_path
