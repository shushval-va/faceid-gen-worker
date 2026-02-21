"""RunPod serverless handler for face-preserving image generation.

Uses SDXL (RealVisXL) + IP-Adapter FaceID Plus V2 to generate images
with a specific person's face from a reference photo.
No NSFW filters.
"""

import base64
import io
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

import runpod

sys.path.insert(0, "/app/IP-Adapter")

from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# Model loading (runs once at worker startup)
# ---------------------------------------------------------------------------
MODELS_DIR = "/app/models"

print("Loading SDXL pipeline (RealVisXL V4.0)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    f"{MODELS_DIR}/RealVisXL_V4.0",
    torch_dtype=torch.float16,
)
pipe.scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

print("Loading IP-Adapter FaceID Plus V2 for SDXL...")
ip_model = IPAdapterFaceIDPlusXL(
    pipe,
    f"{MODELS_DIR}/CLIP-ViT-H-14",
    f"{MODELS_DIR}/ip-adapter-faceid/ip-adapter-faceid-plusv2_sdxl.bin",
    device="cuda",
    torch_dtype=torch.float16,
)

print("Loading InsightFace (antelopev2)...")
face_app = FaceAnalysis(
    name="antelopev2",
    root=f"{MODELS_DIR}/insightface",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

print("All models loaded.")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event):
    """
    Input JSON:
      face_image: base64 encoded face reference photo (required)
      prompt: text prompt describing the desired image (required)
      negative_prompt: text for negative prompt (optional)
      face_scale: IP-Adapter face influence 0.0-1.5 (default 0.7)
      structure_scale: face structure preservation scale (default 1.0)
      guidance_scale: classifier-free guidance (default 7.5)
      num_steps: denoising steps (default 30)
      width: output width (default 768)
      height: output height (default 1024)
      seed: random seed (default 42)
    """
    try:
        inp = event["input"]

        # Decode face image
        face_pil = Image.open(
            io.BytesIO(base64.b64decode(inp["face_image"]))
        ).convert("RGB")

        # Extract face embedding with InsightFace
        face_cv = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)
        faces = face_app.get(face_cv)
        if not faces:
            return {"error": "No face detected in the provided image"}

        faceid_embeds = torch.tensor(
            faces[0].normed_embedding, dtype=torch.float32
        ).unsqueeze(0)

        # Generation parameters
        prompt = inp.get(
            "prompt",
            "beautiful woman, professional photo, high quality, detailed",
        )
        negative_prompt = inp.get(
            "negative_prompt",
            "monochrome, lowres, bad anatomy, worst quality, low quality, "
            "deformed, blurry, watermark, text",
        )

        images = ip_model.generate(
            face_image=face_pil,
            faceid_embeds=faceid_embeds,
            prompt=prompt,
            negative_prompt=negative_prompt,
            scale=float(inp.get("face_scale", 0.7)),
            s_scale=float(inp.get("structure_scale", 1.0)),
            shortcut=True,
            num_samples=1,
            seed=int(inp.get("seed", 42)),
            guidance_scale=float(inp.get("guidance_scale", 7.5)),
            num_inference_steps=int(inp.get("num_steps", 30)),
            width=int(inp.get("width", 768)),
            height=int(inp.get("height", 1024)),
        )

        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode()}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
