import gc
import logging
import os
import time

import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from safetensors.torch import load_file

from blackwell_utils import patch_flux2_with_blackwell

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Flux2BlackwellNVFP4")

# --- Paths ---
REPO_NVFP4 = os.path.expanduser(
    "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev-NVFP4/"
    "snapshots/142b87e70bc3006937b7093d89ff287b5f59f071"
)
# We use the bnb-4bit repo for components OTHER than the transformer (VAE, T5, etc.)
REPO_BASE = "diffusers/FLUX.2-dev-bnb-4bit"
NVFP4_SAFETENSORS = os.path.join(REPO_NVFP4, "flux2-dev-nvfp4-mixed.safetensors")


def main() -> None:
    """Native Blackwell NVFP4 Inference."""

    # 1. Load the base pipeline (excluding the transformer to save time/VRAM initially)
    print(">>> Loading Flux 2 Pipeline components (VAE, T5, CLIP)...")
    pipe = Flux2Pipeline.from_pretrained(
        REPO_BASE,
        transformer=None,  # We will load and patch our own
        torch_dtype=torch.bfloat16,
    )

    # 2. Load the NVFP4 weights from safetensors
    print(f">>> Loading NVFP4 weights from {NVFP4_SAFETENSORS}...")
    checkpoint = load_file(NVFP4_SAFETENSORS)

    # 3. Create the transformer and patch it
    # We need the config, let's pull it from the base repo
    print(">>> Initializing Transformer architecture (meta device)...")
    transformer_config = Flux2Transformer2DModel.load_config(REPO_BASE, subfolder="transformer")
    with torch.device("meta"):
        transformer = Flux2Transformer2DModel.from_config(transformer_config)

    # Patch the meta-model with BlackwellLinear and load the quantized weights
    print(">>> Patching Transformer with BlackwellLinear and loading weights...")
    transformer = patch_flux2_with_blackwell(transformer, checkpoint)

    # Move to GPU
    print(">>> Moving Transformer to GPU (RTX 5090)...")
    transformer.to(device="cuda", dtype=torch.bfloat16)
    pipe.transformer = transformer

    # 4. Enable CPU Offload
    logger.info("Enabling CPU offload...")
    pipe.enable_model_cpu_offload()

    # Cleanup
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Generate Image
    prompt = "A majestic dragon sitting on a pile of gold, highly detailed digital art, cinematic lighting"
    logger.info("Generating image with prompt: %s", prompt)

    generator = torch.Generator(device="cuda").manual_seed(42)

    # Warmup
    logger.info("Running warmup inference...")
    with torch.inference_mode():
        _ = pipe(
            prompt="warmup",
            num_inference_steps=1,
            height=512,
            width=512,
        )

    # Actual timed run
    logger.info("Running timed inference...")
    start_time = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=50,
            guidance_scale=4.0,
            height=1024,
            width=1024,
        ).images[0]
    elapsed = time.perf_counter() - start_time

    output_path = "flux2_blackwell_native_nvfp4.png"
    image.save(output_path)
    logger.info("Image saved to %s (%.1f seconds)", output_path, elapsed)

    # Final VRAM check
    vram_gb = torch.cuda.memory_peak_allocated() / 1024**3
    logger.info("Peak VRAM usage: %.2f GB", vram_gb)


if __name__ == "__main__":
    main()
