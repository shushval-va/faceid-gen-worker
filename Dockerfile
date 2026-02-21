FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps + patched basicsr (single layer)
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
    huggingface_hub==0.25.0 \
    && pip install --no-cache-dir --force-reinstall --no-deps \
    git+https://github.com/XPixelGroup/BasicSR@8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a

# Clone InstantID repo (only pipeline + ip_adapter code needed)
RUN git clone --depth 1 https://github.com/instantX-research/InstantID.git /app/InstantID \
    && rm -rf /app/InstantID/.git /app/InstantID/gradio_demo /app/InstantID/examples /app/InstantID/assets

# Verify all imports at build time
RUN python -c "\
import sys; sys.path.insert(0, '/app/InstantID'); \
import torch; print(f'torch {torch.__version__}'); \
from diffusers.models import ControlNetModel; \
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps; \
import cv2; import insightface; from gfpgan import GFPGANer; \
print('All imports OK')"

# Download all models in a single layer to reduce image size
RUN python -c "\
from huggingface_hub import snapshot_download, hf_hub_download; \
print('--- RealVisXL V5.0 ---'); \
snapshot_download('SG161222/RealVisXL_V5.0', \
    local_dir='/app/models/RealVisXL_V5.0', \
    ignore_patterns=['*.bin', '*.ckpt', '*non_ema*']); \
print('--- InstantID ControlNet + IP-Adapter ---'); \
hf_hub_download('InstantX/InstantID', filename='ControlNetModel/config.json', \
    local_dir='/app/models/InstantID'); \
hf_hub_download('InstantX/InstantID', filename='ControlNetModel/diffusion_pytorch_model.safetensors', \
    local_dir='/app/models/InstantID'); \
hf_hub_download('InstantX/InstantID', filename='ip-adapter.bin', \
    local_dir='/app/models/InstantID'); \
print('--- antelopev2 ---'); \
snapshot_download('kidyu/antelopev2-for-InstantID-ComfyUI', \
    local_dir='/app/models/insightface/models/antelopev2')" \
    && wget -q -O /app/models/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# Pre-download facexlib detection models (used by GFPGAN internally)
RUN python -c "\
from gfpgan import GFPGANer; \
GFPGANer(model_path='/app/models/GFPGANv1.4.pth', upscale=1, \
    arch='clean', channel_multiplier=2, bg_upsampler=None); \
print('GFPGAN initialized OK')"

COPY handler.py /app/handler.py

CMD ["python", "handler.py"]
