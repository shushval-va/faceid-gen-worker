"""
RunPod Serverless Handler: Image Generation with InstantID Face Identity.

Pipeline: InstantID (face identity baked into SDXL generation)
1. Extract face embedding + keypoints from reference image (antelopev2)
2. Generate image with InstantID ControlNet + IP-Adapter
3. Optional: GFPGAN face restoration
"""

import base64
import io
import os
import sys

sys.path.insert(0, "/app/InstantID")

import cv2
import numpy as np
import runpod
import torch
from diffusers import DPMSolverMultistepScheduler
from diffusers.models import ControlNetModel
from gfpgan import GFPGANer
from insightface.app import FaceAnalysis
from PIL import Image
from pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

MODEL_DIR = "/app/models"

pipe = None
face_app = None
restorer = None


def load_models():
    global pipe, face_app, restorer

    # InstantID ControlNet
    controlnet = ControlNetModel.from_pretrained(
        f"{MODEL_DIR}/InstantID/ControlNetModel",
        torch_dtype=torch.float16,
    )

    # SDXL + InstantID pipeline (RealVisXL V5.0)
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        f"{MODEL_DIR}/RealVisXL_V5.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    # Load InstantID IP-Adapter
    pipe.load_ip_adapter_instantid(f"{MODEL_DIR}/InstantID/ip-adapter.bin")

    # InsightFace antelopev2 for face detection/embedding
    face_app = FaceAnalysis(
        name="antelopev2",
        root=f"{MODEL_DIR}/insightface",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # GFPGAN face restorer
    restorer = GFPGANer(
        model_path=f"{MODEL_DIR}/GFPGANv1.4.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


def decode_image(b64_string):
    return Image.open(io.BytesIO(base64.b64decode(b64_string))).convert("RGB")


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def handler(event):
    inp = event["input"]

    face_img = decode_image(inp["face_image"])

    prompt = inp.get(
        "prompt",
        "beautiful woman, professional photo, natural lighting, detailed skin",
    )
    negative = inp.get(
        "negative_prompt",
        "bad hands, bad anatomy, ugly, deformed, face asymmetry, "
        "deformed eyes, deformed mouth, low quality, blurry, "
        "deformed fingers, extra fingers, missing fingers",
    )
    width = inp.get("width", 768)
    height = inp.get("height", 1024)
    steps = inp.get("num_steps", 30)
    guidance = inp.get("guidance_scale", 5.0)
    seed = inp.get("seed", -1)
    do_restore = inp.get("restore_face", True)
    identity_scale = inp.get("identity_scale", 0.8)
    controlnet_scale = inp.get("controlnet_scale", 0.8)

    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

    # --- Step 1: Extract face identity from reference ---
    face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    faces = face_app.get(face_cv)

    if not faces:
        return {"image": "", "error": "No face detected in reference image"}

    # Use largest face
    face_info = sorted(
        faces,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
    )[-1]

    face_emb = face_info["embedding"]
    face_kps = draw_kps(face_img, face_info["kps"])

    # --- Step 2: Generate image with InstantID ---
    pipe.set_ip_adapter_scale(identity_scale)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=controlnet_scale,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    # --- Step 3: Optional face restoration ---
    if do_restore:
        img_cv = pil_to_cv2(image)
        _, _, restored = restorer.enhance(img_cv, paste_back=True)
        image = cv2_to_pil(restored)

    return {"image": encode_image(image), "face_preserved": True}


load_models()
runpod.serverless.start({"handler": handler})
