"""
RunPod Serverless Handler: NSFW Image Generation with Face Swap.

Two-step pipeline:
1. Generate photorealistic image with SDXL (RealVisXL V5.0)
2. Swap face from reference onto generated image (inswapper_128)
3. Restore face quality (GFPGAN)
"""

import base64
import io
import os

import cv2
import numpy as np
import runpod
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from gfpgan import GFPGANer
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model as get_insightface_model
from PIL import Image

MODEL_DIR = "/app/models"

pipe = None
face_app = None
swapper = None
restorer = None


def load_models():
    global pipe, face_app, swapper, restorer

    # SDXL pipeline (RealVisXL V5.0)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        f"{MODEL_DIR}/RealVisXL_V5.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    # InsightFace for face detection/recognition
    face_app = FaceAnalysis(
        name="buffalo_l",
        root=f"{MODEL_DIR}/insightface",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # inswapper_128 face swap model
    swapper = get_insightface_model(
        f"{MODEL_DIR}/inswapper_128.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

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
    guidance = inp.get("guidance_scale", 7.0)
    seed = inp.get("seed", -1)
    do_restore = inp.get("restore_face", True)

    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

    # --- Step 1: Generate base image with SDXL ---
    gen_image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    # --- Step 2: Face swap ---
    gen_cv = pil_to_cv2(gen_image)
    face_cv = pil_to_cv2(face_img)

    target_faces = face_app.get(gen_cv)
    source_faces = face_app.get(face_cv)

    if not source_faces:
        return {
            "image": encode_image(gen_image),
            "face_swapped": False,
            "error": "No face detected in source image",
        }

    if not target_faces:
        return {
            "image": encode_image(gen_image),
            "face_swapped": False,
            "error": "No face detected in generated image",
        }

    # Pick largest face in generated image
    target_faces = sorted(
        target_faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )

    swapped = swapper.get(gen_cv, target_faces[0], source_faces[0], paste_back=True)

    # --- Step 3: Face restoration ---
    if do_restore:
        _, _, swapped = restorer.enhance(swapped, paste_back=True)

    final = cv2_to_pil(swapped)

    return {"image": encode_image(final), "face_swapped": True}


load_models()
runpod.serverless.start({"handler": handler})
