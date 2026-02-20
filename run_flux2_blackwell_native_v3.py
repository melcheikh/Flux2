import argparse
import gc
import logging
import os
import random
import time

import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from blackwell_utils import patch_flux2_with_blackwell

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Flux2BlackwellNVFP4")

REPO_BASE = "diffusers/FLUX.2-dev-bnb-4bit"
DEFAULT_NVFP4_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev-NVFP4/" "flux2-dev-nvfp4.safetensors"
)
TEXT_ENCODER_REPO = "Comfy-Org/flux2-dev"
TEXT_ENCODER_FILE = "split_files/text_encoders/mistral_3_small_flux2_fp4_mixed.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux 2 Blackwell NVFP4 (no fallback).")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "A majestic dragon sitting on a pile of gold, highly detailed digital art, cinematic lighting"
        ),
        help="Text prompt for image generation",
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale (default: 4.0)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--seed", type=int, default=None, help="Base seed. If set, uses seed+i.")
    parser.add_argument("--count", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="flux2_blackwell_nvfp4",
        help="Prefix for output filenames",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument(
        "--offload",
        type=str,
        default="model",
        choices=["model", "sequential", "none"],
        help="Offload strategy: model (safe), sequential, none",
    )
    parser.add_argument(
        "--nvfp4_path",
        type=str,
        default=DEFAULT_NVFP4_PATH,
        help="Absolute path to flux2-dev-nvfp4.safetensors",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Optional path to text encoder safetensors (overrides hub download)",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 matmul for extra speed (may slightly affect quality)",
    )
    return parser.parse_args()


def resolve_seed(base_seed: int | None, index: int) -> int:
    if base_seed is None:
        return random.randint(0, 2**32 - 1)
    return base_seed + index


def normalize_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("model.transformer."):
            new_key = new_key[len("model.transformer.") :]
        elif new_key.startswith("transformer."):
            new_key = new_key[len("transformer.") :]
        elif new_key.startswith("model."):
            new_key = new_key[len("model.") :]

        if new_key.endswith(".weight_2"):
            base_key = new_key[: -len(".weight_2")] + ".weight"
            if base_key not in state_dict and base_key not in normalized:
                new_key = base_key

        normalized[new_key] = value
    return normalized


def build_state_dict_for_load(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return state_dict


def resolve_text_encoder_path(path_arg: str | None) -> str:
    if path_arg:
        resolved = os.path.expanduser(path_arg)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Text encoder safetensors file not found at {resolved}")
        return resolved

    return hf_hub_download(repo_id=TEXT_ENCODER_REPO, filename=TEXT_ENCODER_FILE)


def load_text_encoder_weights(pipe: Flux2Pipeline, text_encoder_path: str) -> None:
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        logger.info("Text encoder: not present")
        return

    logger.info("Loading text encoder weights from %s...", text_encoder_path)
    state_dict = load_file(text_encoder_path)
    incompatible = text_encoder.load_state_dict(state_dict, strict=False)
    logger.info(
        "Text encoder state loaded. Missing keys: %s | Unexpected keys: %s",
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )


def log_text_encoder_info(pipe: Flux2Pipeline) -> None:
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        logger.info("Text encoder: not present")
        return

    try:
        first_param = next(text_encoder.parameters())
    except StopIteration:
        logger.info("Text encoder: no parameters")
        return

    dtype = first_param.dtype
    device = first_param.device
    numel = sum(p.numel() for p in text_encoder.parameters())
    if dtype.is_floating_point:
        bytes_per = torch.finfo(dtype).bits // 8
    else:
        bytes_per = torch.iinfo(dtype).bits // 8
    size_gb = numel * bytes_per / (1024**3)

    logger.info(
        "Text encoder: %s | dtype=%s | device=%s | params=%s | approx_size=%.2f GB",
        text_encoder.__class__.__name__,
        dtype,
        device,
        numel,
        size_gb,
    )


def main() -> None:
    args = parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    nvfp4_path = os.path.expanduser(args.nvfp4_path)
    if not os.path.isfile(nvfp4_path):
        raise FileNotFoundError(
            f"NVFP4 safetensors file not found at {nvfp4_path}. " "Pass --nvfp4_path to override."
        )

    text_encoder_path = resolve_text_encoder_path(args.text_encoder_path)

    logger.info("Pipeline base repo: %s", REPO_BASE)
    logger.info("NVFP4 safetensors path: %s", nvfp4_path)
    logger.info("Text encoder safetensors path: %s", text_encoder_path)

    logger.info("Loading Flux 2 Pipeline components (VAE, T5, CLIP)...")
    pipe = Flux2Pipeline.from_pretrained(
        REPO_BASE,
        transformer=None,
        torch_dtype=torch.bfloat16,
    )

    if args.offload == "sequential":
        logger.info("Enabling sequential CPU offload (early)...")
        pipe.enable_sequential_cpu_offload()
    elif args.offload == "model":
        logger.info("Enabling model CPU offload (early)...")
        pipe.enable_model_cpu_offload()

    load_text_encoder_weights(pipe, text_encoder_path)
    log_text_encoder_info(pipe)

    torch.cuda.empty_cache()

    logger.info("Loading NVFP4 weights from %s...", nvfp4_path)
    checkpoint = normalize_checkpoint_keys(load_file(nvfp4_path))

    logger.info("Initializing Transformer architecture (meta)...")
    transformer_config = Flux2Transformer2DModel.load_config(REPO_BASE, subfolder="transformer")
    with torch.device("meta"):
        transformer = Flux2Transformer2DModel.from_config(transformer_config)

    logger.info("Materializing empty Transformer on CPU...")
    transformer = transformer.to_empty(device="cpu")

    logger.info("Patching Transformer with BlackwellLinear and loading weights...")
    transformer = patch_flux2_with_blackwell(transformer, checkpoint)

    logger.info("Loading remaining NVFP4 weights into Transformer...")
    state_dict_for_load = build_state_dict_for_load(checkpoint)
    incompatible = transformer.load_state_dict(state_dict_for_load, strict=False, assign=True)
    logger.info(
        "NVFP4 state loaded. Missing keys: %s | Unexpected keys: %s",
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )

    logger.info("Moving Transformer to %s...", args.device)
    transformer.to(device=args.device, dtype=torch.bfloat16)
    pipe.transformer = transformer

    if args.offload == "none":
        logger.info("Keeping pipeline on %s (no offload).", args.device)
        pipe = pipe.to(args.device)

    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Generating %s image(s)...", args.count)
    logger.info("Prompt: %s", args.prompt)

    # Force text encoder to stay on CPU to avoid OOM
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.to("cpu")
        if hasattr(pipe.text_encoder, "_hf_hook"):
            try:
                pipe.text_encoder._hf_hook.offload = True
                pipe.text_encoder._hf_hook.execution_device = torch.device("cpu")
            except Exception:
                pass

    total_start = time.perf_counter()
    for index in range(args.count):
        current_seed = resolve_seed(args.seed, index)
        filename = f"{args.output_prefix}_{index + 1}.png"
        output_path = os.path.abspath(filename)
        logger.info(
            "[Image %s/%s] seed=%s -> %s",
            index + 1,
            args.count,
            current_seed,
            output_path,
        )

        generator = torch.Generator(device=args.device).manual_seed(current_seed)

        # Encode prompt on CPU to avoid GPU OOM from text encoder
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=args.prompt,
            device=torch.device("cpu"),
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        prompt_embeds = prompt_embeds.to(device=args.device)

        with torch.inference_mode():
            image = pipe(
                prompt_embeds=prompt_embeds,
                generator=generator,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                height=args.height,
                width=args.width,
            ).images[0]

        image.save(output_path)

    total_elapsed = time.perf_counter() - total_start
    logger.info("Total time: %.1f seconds", total_elapsed)


if __name__ == "__main__":
    main()
