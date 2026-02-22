"""
RunPod Serverless Handler: Flux + PuLID identity-preserving image generation.

Pipeline:
1. Extract face ID embedding from face_image (PuLID: antelopev2 + EVA-CLIP)
2. Encode prompt with T5 + CLIP text encoders
3. Denoise with Flux DiT (fp8) + PuLID ID injection
4. Decode latents with Flux VAE
"""

import base64
import gc
import io
import json
import os
import sys
import time

sys.path.insert(0, "/app/PuLID")

import numpy as np
import runpod
import torch
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file as load_sft

from flux.model import Flux
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, load_ae, load_clip, load_t5
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long

MODEL_DIR = "/app/models"

model = None
ae = None
t5 = None
clip = None
pulid_model = None


def download_gated_models():
    """Download models that require HF_TOKEN (gated/sensitive content).
    Skips if files already exist. Called once at startup.
    """
    from huggingface_hub import hf_hub_download

    vae_path = os.path.join(MODEL_DIR, "ae.safetensors")
    lora_path = os.path.join(MODEL_DIR, "nsfw_lora.safetensors")

    if not os.path.exists(vae_path):
        print("Downloading Flux VAE (gated, requires HF_TOKEN)...")
        hf_hub_download(
            "black-forest-labs/FLUX.1-dev", "ae.safetensors",
            local_dir=MODEL_DIR,
        )
        print("Flux VAE OK")
    else:
        print("Flux VAE already cached")

    if not os.path.exists(lora_path):
        print("Downloading NSFW LoRA (sensitive, requires HF_TOKEN)...")
        try:
            hf_hub_download(
                "enhanceaiteam/Flux-Uncensored-V2", "lora.safetensors",
                local_dir=MODEL_DIR,
            )
            os.rename(
                os.path.join(MODEL_DIR, "lora.safetensors"), lora_path,
            )
            print("NSFW LoRA OK")
        except Exception as e:
            print(f"WARNING: NSFW LoRA download failed: {e}")
    else:
        print("NSFW LoRA already cached")


def merge_lora(state_dict, lora_path, scale=1.0):
    """Merge LoRA weights into Flux state dict (before requantization).

    Supports both BFL format (lora_down/lora_up) and
    diffusers format (lora_A/lora_B) key naming.
    """
    lora_sd = load_sft(lora_path, device="cpu")

    lora_keys_sample = list(lora_sd.keys())[:5]
    model_keys_sample = list(state_dict.keys())[:5]
    print(f"LoRA keys sample: {lora_keys_sample}")
    print(f"Model keys sample: {model_keys_sample}")

    down_keys = [
        k for k in lora_sd
        if k.endswith(".lora_down.weight") or k.endswith(".lora_A.weight")
    ]
    print(f"LoRA: found {len(down_keys)} down-projection layers")

    applied, skipped = 0, []
    for down_key in down_keys:
        up_key = down_key.replace("lora_down", "lora_up").replace("lora_A", "lora_B")
        if up_key not in lora_sd:
            skipped.append(("no_up", down_key))
            continue

        base_key = (
            down_key
            .replace(".lora_down.weight", ".weight")
            .replace(".lora_A.weight", ".weight")
        )
        if base_key not in state_dict:
            skipped.append(("no_base", base_key))
            continue

        down = lora_sd[down_key].float()
        up = lora_sd[up_key].float()
        delta = (up @ down) * scale
        state_dict[base_key] = state_dict[base_key].float() + delta
        applied += 1

    print(f"LoRA: {applied} layers merged, {len(skipped)} skipped")
    if skipped[:10]:
        print(f"  skipped (first 10): {skipped[:10]}")
    del lora_sd
    gc.collect()
    return state_dict


def load_flux_fp8(name="flux-dev", lora_path=None, lora_scale=1.0):
    """Load Flux DiT in fp8 with optional LoRA merge."""
    from huggingface_hub import hf_hub_download
    from optimum.quanto import requantize

    fp8_path = os.path.join(MODEL_DIR, "flux-dev-fp8.safetensors")
    map_path = os.path.join(MODEL_DIR, "flux_dev_quantization_map.json")

    if not os.path.exists(fp8_path):
        fp8_path = hf_hub_download(
            "XLabs-AI/flux-dev-fp8", "flux-dev-fp8.safetensors", local_dir=MODEL_DIR
        )
    if not os.path.exists(map_path):
        map_path = hf_hub_download(
            "XLabs-AI/flux-dev-fp8", "flux_dev_quantization_map.json",
            local_dir=MODEL_DIR,
        )

    print("Loading fp8 state dict...")
    sd = load_sft(fp8_path, device="cpu")

    if lora_path and os.path.exists(lora_path):
        print(f"Merging LoRA from {lora_path} (scale={lora_scale})...")
        sd = merge_lora(sd, lora_path, scale=lora_scale)

    print("Creating Flux model + requantizing to fp8...")
    flux_model = Flux(configs[name].params).to(torch.bfloat16)
    with open(map_path) as f:
        qmap = json.load(f)
    requantize(flux_model, sd, qmap, device="cpu")

    del sd
    gc.collect()
    return flux_model


def load_models():
    """Initialize all models. Called once at worker startup."""
    global model, ae, t5, clip, pulid_model
    device = "cuda"

    lora_path = os.path.join(MODEL_DIR, "nsfw_lora.safetensors")
    model = load_flux_fp8(
        "flux-dev",
        lora_path=lora_path if os.path.exists(lora_path) else None,
        lora_scale=1.0,
    )
    model.eval()

    print("Loading T5 + CLIP text encoders...")
    t5_enc = load_t5(device, max_length=128)
    clip_enc = load_clip(device)
    t5 = t5_enc
    clip = clip_enc

    print("Loading Flux VAE...")
    ae = load_ae("flux-dev", device="cpu")

    print("Initializing PuLID pipeline...")
    pulid_model = PuLIDPipeline(
        model, device="cpu", weight_dtype=torch.bfloat16, onnx_provider="gpu",
    )
    pulid_model.face_helper.face_det.mean_tensor = (
        pulid_model.face_helper.face_det.mean_tensor.to(device)
    )
    pulid_model.face_helper.face_det.device = torch.device(device)
    pulid_model.face_helper.device = torch.device(device)
    pulid_model.device = torch.device(device)
    pulid_model.load_pretrain(version="v0.9.1")

    print("All models loaded OK.")


def encode_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def handler(event):
    inp = event["input"]

    face_b64 = inp.get("face_image")
    if not face_b64:
        return {"error": "face_image is required"}

    face_pil = Image.open(io.BytesIO(base64.b64decode(face_b64))).convert("RGB")
    face_np = np.array(face_pil)
    face_np = resize_numpy_image_long(face_np, 1024)

    prompt = inp.get("prompt", "portrait, color, cinematic, professional photo")
    neg_prompt = inp.get(
        "negative_prompt",
        "bad quality, worst quality, text, signature, watermark, extra limbs",
    )
    width = inp.get("width", 896)
    height = inp.get("height", 1152)
    num_steps = inp.get("num_steps", 20)
    start_step = inp.get("start_step", 4)
    guidance = inp.get("guidance_scale", 4.0)
    seed = inp.get("seed", -1)
    id_weight = inp.get("id_weight", 1.0)
    true_cfg = inp.get("true_cfg", 1.0)
    max_seq_len = inp.get("max_sequence_length", 128)

    if seed == -1:
        seed = int(torch.Generator(device="cpu").seed())

    device = torch.device("cuda")
    t0 = time.perf_counter()

    use_true_cfg = abs(true_cfg - 1.0) > 1e-2

    with torch.inference_mode():
        # --- Step 1: Prepare noise + timesteps ---
        x = get_noise(1, height, width, device=device, dtype=torch.bfloat16, seed=seed)
        timesteps = get_schedule(num_steps, x.shape[-1] * x.shape[-2] // 4, shift=True)

        # --- Step 2: Encode prompt (T5 + CLIP on GPU) ---
        t5.max_length = max_seq_len
        t5.to(device)
        clip.to(device)
        inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=prompt)
        inp_neg = (
            prepare(t5=t5, clip=clip, img=x, prompt=neg_prompt)
            if use_true_cfg else None
        )
        t5.cpu()
        clip.cpu()
        torch.cuda.empty_cache()

        # --- Step 3: Extract face ID embedding (PuLID) ---
        pulid_model.components_to_device(device)
        try:
            id_emb, uncond_id_emb = pulid_model.get_id_embedding(
                face_np, cal_uncond=use_true_cfg,
            )
        except RuntimeError as e:
            pulid_model.components_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
            return {"error": f"Face detection failed: {e}"}

        pulid_model.components_to_device(torch.device("cpu"))
        torch.cuda.empty_cache()

        # --- Step 4: Denoise (Flux DiT on GPU) ---
        model.to(device)
        x = denoise(
            model,
            **inp_cond,
            timesteps=timesteps,
            guidance=guidance,
            id=id_emb,
            id_weight=id_weight,
            start_step=start_step,
            uncond_id=uncond_id_emb,
            true_cfg=true_cfg,
            timestep_to_start_cfg=1,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
        )
        model.cpu()
        torch.cuda.empty_cache()

        # --- Step 5: Decode latents (VAE on GPU) ---
        ae.decoder.to(device)
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = ae.decode(x)
        ae.decoder.cpu()
        torch.cuda.empty_cache()

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    t1 = time.perf_counter()
    print(f"Generated in {t1 - t0:.1f}s (seed={seed})")

    return {"image": encode_image(img), "seed": seed}


download_gated_models()
load_models()
runpod.serverless.start({"handler": handler})
