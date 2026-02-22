FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DOWNLOAD_TIMEOUT=600
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget g++ libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps for Flux + PuLID
RUN pip install --no-cache-dir \
    runpod brotlicffi \
    diffusers==0.30.0 \
    transformers==4.43.3 \
    accelerate \
    safetensors \
    insightface \
    onnxruntime-gpu==1.20.0 \
    opencv-python \
    huggingface_hub==0.25.0 \
    hf_transfer \
    einops \
    timm \
    ftfy \
    facexlib \
    sentencepiece \
    optimum-quanto==0.2.4 \
    torchsde

# Clone PuLID repo (Flux + identity pipeline)
RUN git clone --depth 1 https://github.com/ToTheBeginning/PuLID.git /app/PuLID \
    && rm -rf /app/PuLID/.git /app/PuLID/example_inputs /app/PuLID/docs

# Verify all imports at build time
RUN python -c "\
import sys; sys.path.insert(0, '/app/PuLID'); \
import torch; print(f'torch {torch.__version__}'); \
from flux.model import Flux; \
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack; \
from flux.util import configs, load_ae, load_clip, load_t5; \
from pulid.pipeline_flux import PuLIDPipeline; \
from einops import rearrange; \
import cv2; import insightface; \
print('All imports OK')"

# --- Model downloads (split into layers for Docker cache) ---

# Layer 1: Flux Dev fp8 (~12GB) — largest download, cached first
RUN python -c "\
from huggingface_hub import hf_hub_download; \
print('--- Flux Dev fp8 ---'); \
hf_hub_download('XLabs-AI/flux-dev-fp8', 'flux-dev-fp8.safetensors', \
    local_dir='/app/models'); \
hf_hub_download('XLabs-AI/flux-dev-fp8', 'flux_dev_quantization_map.json', \
    local_dir='/app/models'); \
print('Flux Dev fp8 OK')"

# Layer 2: Flux VAE (gated model — needs HF_TOKEN with accepted license)
ARG HF_TOKEN=""
RUN python -c "\
import os; \
token = '${HF_TOKEN}' or os.environ.get('HF_TOKEN') or None; \
from huggingface_hub import hf_hub_download; \
print('--- Flux VAE ---'); \
hf_hub_download('black-forest-labs/FLUX.1-dev', 'ae.safetensors', \
    local_dir='/app/models', token=token if token else None); \
print('VAE OK')"

# Layer 3: T5 + CLIP text encoders (~10GB total)
RUN python -c "\
import sys; sys.path.insert(0, '/app/PuLID'); \
from flux.util import load_t5, load_clip; \
print('--- T5 encoder ---'); \
t5 = load_t5('cpu', max_length=128); \
print('--- CLIP encoder ---'); \
clip = load_clip('cpu'); \
print('Text encoders OK')"

# Layer 4: PuLID model + antelopev2 (public mirror) + facexlib + EVA-CLIP
# NOTE: PuLID hardcodes DIAMONIK7777/antelopev2 (private), so we pre-download
# from a public mirror to the path PuLID expects (models/antelopev2/ from CWD).
RUN python -c "\
from huggingface_hub import hf_hub_download, snapshot_download; \
print('--- PuLID model ---'); \
hf_hub_download('guozinan/PuLID', 'pulid_flux_v0.9.1.safetensors', \
    local_dir='models'); \
print('--- antelopev2 (public mirror) ---'); \
snapshot_download('kidyu/antelopev2-for-InstantID-ComfyUI', \
    local_dir='models/antelopev2'); \
print('PuLID + antelopev2 OK')"

# Pre-download facexlib detection/parsing models
RUN python -c "\
from facexlib.utils.face_restoration_helper import FaceRestoreHelper; \
from facexlib.parsing import init_parsing_model; \
helper = FaceRestoreHelper(upscale_factor=1, face_size=512, crop_ratio=(1,1), \
    det_model='retinaface_resnet50', save_ext='png', device='cpu'); \
helper.face_parse = init_parsing_model(model_name='bisenet', device='cpu'); \
print('facexlib OK')"

# Pre-download EVA-CLIP model (used by PuLID for face features)
RUN python -c "\
import sys; sys.path.insert(0, '/app/PuLID'); \
from eva_clip import create_model_and_transforms; \
model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', \
    force_custom_clip=True); \
print('EVA-CLIP OK')"

# Layer 5: NSFW LoRA (~687MB)
RUN python -c "\
from huggingface_hub import hf_hub_download; \
print('--- NSFW LoRA ---'); \
hf_hub_download('enhanceaiteam/Flux-Uncensored-V2', 'lora.safetensors', \
    local_dir='/app/models'); \
print('NSFW LoRA OK')" \
    && mv /app/models/lora.safetensors /app/models/nsfw_lora.safetensors \
    || echo "WARNING: NSFW LoRA download failed (may need HF auth for sensitive content)"

# Prevent runtime downloads — all models must be pre-cached above
ENV HF_HUB_OFFLINE=1

COPY handler.py /app/handler.py

CMD ["python", "handler.py"]
