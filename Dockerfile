FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (PyTorch 2.5.1 natively supports NumPy 2.x)
RUN pip install --no-cache-dir \
    runpod brotlicffi \
    diffusers==0.31.0 \
    transformers==4.46.0 \
    accelerate==1.1.0 \
    safetensors \
    insightface \
    onnxruntime-gpu==1.20.0 \
    opencv-python \
    gfpgan \
    huggingface_hub==0.25.0

# Replace PyPI basicsr with patched version (official fix for torchvision >= 0.18 compat)
# Must run AFTER gfpgan install to override the broken basicsr it pulls in
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    git+https://github.com/XPixelGroup/BasicSR@8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a

WORKDIR /app

# Verify all imports at build time
RUN python -c "\
import torch; print(f'torch {torch.__version__}'); \
import numpy; print(f'numpy {numpy.__version__}'); \
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler; \
import cv2; import insightface; \
from gfpgan import GFPGANer; \
print('All imports OK')"

# Download RealVisXL V5.0
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('SG161222/RealVisXL_V5.0', \
    local_dir='/app/models/RealVisXL_V5.0', \
    ignore_patterns=['*.bin', '*.ckpt', '*non_ema*'])"

# Download inswapper_128.onnx (face swap model)
RUN mkdir -p /app/models && \
    wget -q -O /app/models/inswapper_128.onnx \
    https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx

# Download GFPGANv1.4 (face restoration)
RUN wget -q -O /app/models/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# Download InsightFace buffalo_l models (face detection + recognition)
RUN mkdir -p /app/models/insightface/models/buffalo_l && \
    python -c "\
from huggingface_hub import hf_hub_download; \
files = ['det_10g.onnx', '1k3d68.onnx', '2d106det.onnx', 'genderage.onnx', 'w600k_r50.onnx']; \
[hf_hub_download('DIAMONIK7777/buffalo_l', filename=f, \
    local_dir='/app/models/insightface/models/buffalo_l') for f in files]"

# Pre-download facexlib detection models (used by GFPGAN internally)
RUN python -c "\
from gfpgan import GFPGANer; \
r = GFPGANer(model_path='/app/models/GFPGANv1.4.pth', upscale=1, \
    arch='clean', channel_multiplier=2, bg_upsampler=None); \
print('GFPGAN initialized OK')"

COPY handler.py /app/handler.py

CMD ["python", "handler.py"]
