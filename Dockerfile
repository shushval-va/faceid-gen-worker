FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    runpod brotlicffi \
    diffusers==0.25.0 \
    transformers==4.36.2 \
    accelerate==0.26.1 \
    safetensors \
    insightface \
    onnxruntime-gpu==1.16.2 \
    opencv-python \
    huggingface_hub==0.25.0

# Clone IP-Adapter code (provides FaceID adapter classes)
RUN git clone https://github.com/tencent-ailab/IP-Adapter.git /app/IP-Adapter

WORKDIR /app

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
