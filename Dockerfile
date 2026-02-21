FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (PyTorch 2.5.1 natively supports NumPy 2.x)
RUN pip install --no-cache-dir \
    runpod brotlicffi \
    diffusers==0.31.0 \
    transformers==4.46.0 \
    accelerate==1.1.0 \
    safetensors \
    einops \
    insightface \
    onnxruntime-gpu==1.20.0 \
    opencv-python \
    huggingface_hub==0.25.0

# Clone IP-Adapter code (provides FaceID adapter classes)
RUN git clone https://github.com/tencent-ailab/IP-Adapter.git /app/IP-Adapter

WORKDIR /app

# Verify all imports work at build time (catches missing deps early)
RUN python -c "\
import numpy; print(f'numpy {numpy.__version__}'); \
import torch; print(f'torch {torch.__version__}'); \
from diffusers import DDIMScheduler, StableDiffusionXLPipeline; \
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection; \
import cv2; import einops; import insightface; \
import sys; sys.path.insert(0, '/app/IP-Adapter'); \
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL; \
print('All imports OK')"

# Download RealVisXL V4.0 (uncensored SDXL model)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('SG161222/RealVisXL_V4.0', \
    local_dir='/app/models/RealVisXL_V4.0', \
    ignore_patterns=['*.bin', '*.ckpt', '*non_ema*'])"

# Download IP-Adapter FaceID Plus V2 weights for SDXL
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('h94/IP-Adapter-FaceID', \
    filename='ip-adapter-faceid-plusv2_sdxl.bin', \
    local_dir='/app/models/ip-adapter-faceid')"

# Download CLIP ViT-H/14 image encoder (used by IP-Adapter Plus)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', \
    local_dir='/app/models/CLIP-ViT-H-14')"

# Download InsightFace antelopev2 model for face embedding extraction
RUN mkdir -p /app/models/insightface/models/antelopev2 && \
    python -c "\
from huggingface_hub import hf_hub_download; \
files = ['1k3d68.onnx', '2d106det.onnx', 'genderage.onnx', 'glintr100.onnx', 'scrfd_10g_bnkps.onnx']; \
[hf_hub_download('DIAMONIK7777/antelopev2', filename=f, \
    local_dir='/app/models/insightface/models/antelopev2') for f in files]"

COPY handler.py /app/handler.py

CMD ["python", "handler.py"]
