# LTX (GGUF) runtime deps for .ltx_env
# Torch is installed separately in ltx_install.bat (CUDA wheel index-url)

diffusers>=0.36.0
transformers>=4.48.0
accelerate>=1.2.0
huggingface_hub>=0.25.0
safetensors>=0.4.5

# GGUF loader for diffusers
gguf>=0.10.0

# T5 tokenizer dependency
sentencepiece>=0.2.0
protobuf>=4.25.0

numpy>=1.26.0
pillow>=10.0.0
tqdm>=4.66.0
einops>=0.7.0

# video export
imageio>=2.34.0
imageio-ffmpeg>=0.5.1
av>=12.0.0
packaging>=23.2
