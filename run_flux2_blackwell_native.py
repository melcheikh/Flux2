"""
Flux 2 Blackwell NVFP4 — Using Diffusers native from_single_file() loading.

Replaces the broken manual BlackwellLinear implementation with Diffusers'
native support for BFL's NVFP4 safetensors files.

Fallback: if from_single_file() doesn't support NVFP4 format natively,
falls back to the proven BNB 4-bit quantized model.
"""

import gc
import logging
import os
import time

import torch
from diffusers import AutoModel, Flux2Pipeline, Flux2Transformer2DModel
from transformers import Mistral3ForConditionalGeneration

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Flux2Blackwell")

# --- Paths ---
REPO_NVFP4 = os.path.expanduser(
    "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev-NVFP4/"
    "snapshots/142b87e70bc3006937b7093d89ff287b5f59f071"
)
REPO_BNB4BIT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--diffusers--FLUX.2-dev-bnb-4bit/"
    "snapshots/c30ad107542e63f222f864a8de510204394fb18a"
)
NVFP4_SAFETENSORS = os.path.join(REPO_NVFP4, "flux2-dev-nvfp4-mixed.safetensors")


def load_transformer_nvfp4() -> Flux2Transformer2DModel | None:
    """Attempt to load transformer via Diffusers from_single_file (NVFP4)."""
    if not os.path.isfile(NVFP4_SAFETENSORS):
        logger.warning("NVFP4 safetensors file not found at %s", NVFP4_SAFETENSORS)
        return None

    try:
        logger.info("Attempting from_single_file() with NVFP4 weights...")
        transformer = Flux2Transformer2DModel.from_single_file(
            NVFP4_SAFETENSORS,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Successfully loaded NVFP4 transformer via from_single_file()!")
        return transformer
    except Exception as exc:
        logger.warning("from_single_file() failed for NVFP4: %s", exc)
        logger.info("Will fall back to BNB 4-bit model.")
        return None


def load_transformer_bnb4bit() -> Flux2Transformer2DModel:
    """Load transformer from pre-quantized BNB 4-bit repo (proven path)."""
    logger.info("Loading BNB 4-bit Transformer from %s...", REPO_BNB4BIT)
    transformer = AutoModel.from_pretrained(
        REPO_BNB4BIT,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    logger.info("BNB 4-bit Transformer loaded successfully.")
    return transformer


def load_text_encoder() -> Mistral3ForConditionalGeneration:
    """Load Mistral 3 text encoder from BNB 4-bit repo."""
    logger.info("Loading Mistral 3 Text Encoder from %s...", REPO_BNB4BIT)
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        REPO_BNB4BIT,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    logger.info("Text encoder loaded.")
    return text_encoder


def main() -> None:
    """Main inference pipeline."""
    # 1. Load Transformer — try NVFP4 first, fall back to BNB 4-bit
    transformer = load_transformer_nvfp4()
    if transformer is None:
        transformer = load_transformer_bnb4bit()

    # 2. Load Text Encoder
    text_encoder = load_text_encoder()

    # 3. Assemble Pipeline
    logger.info("Assembling Flux 2 Pipeline...")
    pipe = Flux2Pipeline.from_pretrained(
        REPO_BNB4BIT,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )

    # 4. Enable CPU Offload to fit within 24GB VRAM
    logger.info("Enabling CPU offload...")
    pipe.enable_model_cpu_offload()

    # Free references
    del transformer, text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    vram_gb = torch.cuda.memory_allocated() / 1024**3
    logger.info("Pipeline ready. VRAM: %.2f GB / 24.00 GB", vram_gb)

    # 5. Generate Image
    prompt = "A majestic dragon sitting on a pile of gold, highly detailed digital art, cinematic lighting"
    logger.info("Generating image with prompt: %s", prompt)

    generator = torch.Generator(device="cuda").manual_seed(42)

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

    output_path = "flux2_blackwell_hybrid.png"
    image.save(output_path)
    logger.info("Image saved to %s (%.1f seconds)", output_path, elapsed)


if __name__ == "__main__":
    main()
