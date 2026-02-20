import argparse
import os
import random
import time

import torch
from diffusers import AutoModel, Flux2Pipeline
from transformers import Mistral3ForConditionalGeneration

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux 2 NF4 fast-ish batch run with safe offload.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Realistic macro photograph of a hermit crab using a soda can as its shell, "
            "partially emerging from the can, captured with sharp detail and natural colors, "
            "on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean "
            "waves in the background. The can has the text `BFL + Diffusers` on it and it has a "
            "color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."
        ),
        help="Text prompt for image generation",
    )
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps (default: 28)")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale (default: 4.0)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--seed", type=int, default=None, help="Base seed. If set, uses seed+i.")
    parser.add_argument("--count", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="flux2_fast_batch",
        help="Prefix for output filenames",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument(
        "--offload",
        type=str,
        default="sequential",
        choices=["sequential", "model", "none"],
        help="Offload strategy: sequential (fastest-safe), model (slow), none (try full GPU)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "fast"],
        help="Preset configuration: default or fast",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 matmul for extra speed (may slightly affect quality)",
    )
    return parser.parse_args()

def load_text_encoder(
    repo_id: str,
    dtype: torch.dtype,
    low_cpu_mem_usage: bool,
    device_map: str | None,
) -> Mistral3ForConditionalGeneration:
    return Mistral3ForConditionalGeneration.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

def load_transformer(
    repo_id: str,
    dtype: torch.dtype,
    low_cpu_mem_usage: bool,
    device_map: str | None,
) -> AutoModel:
    return AutoModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

def build_pipeline(
    repo_id: str,
    text_encoder,
    transformer,
    dtype: torch.dtype,
    low_cpu_mem_usage: bool = True,
) -> Flux2Pipeline:
    return Flux2Pipeline.from_pretrained(
        repo_id,
        text_encoder=text_encoder,
        transformer=transformer,
        torch_dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

def rebuild_for_model_offload(repo_id: str, dtype: torch.dtype) -> tuple[Flux2Pipeline, str]:
    print("Re-loading components for model CPU offload (no meta tensors).")
    text_encoder = load_text_encoder(
        repo_id,
        dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    dit = load_transformer(
        repo_id,
        dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    pipe = build_pipeline(repo_id, text_encoder, dit, dtype, low_cpu_mem_usage=True)
    pipe.enable_model_cpu_offload()
    return pipe, "model"

def resolve_seed(base_seed: int | None, index: int) -> int:
    if base_seed is None:
        return random.randint(0, 2**32 - 1)
    return base_seed + index

def main() -> None:
    args = parse_args()

    if args.preset == "fast":
        args.steps = 24
        args.guidance = 3.5
        args.width = 896
        args.height = 896
        args.tf32 = True
        if args.offload == "sequential":
            args.offload = "model"

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
    dtype = torch.bfloat16

    print("Initializing Flux 2 NF4 Pipeline (FAST, safe offload)...")

    print(f"Loading Text Encoder from {repo_id} (CPU first)...")
    text_encoder = load_text_encoder(
        repo_id,
        dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    print(f"Loading Transformer (DiT) from {repo_id} (CPU first)...")
    dit = load_transformer(
        repo_id,
        dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    print("Assembling Pipeline...")
    pipe = build_pipeline(repo_id, text_encoder, dit, dtype)

    offload_mode = args.offload
    if args.offload == "none":
        try:
            print("Attempting full GPU residency...")
            pipe = pipe.to(args.device)
            offload_mode = "none"
        except torch.cuda.OutOfMemoryError:
            print("OOM -> falling back to sequential CPU offload")
            torch.cuda.empty_cache()
            try:
                pipe.enable_sequential_cpu_offload()
                offload_mode = "sequential"
            except TypeError as exc:
                print(f"Sequential offload failed ({exc}). Falling back to model CPU offload.")
                pipe, offload_mode = rebuild_for_model_offload(repo_id, dtype)
    elif args.offload == "model":
        print("Using model CPU offload (slower, safest).")
        pipe, offload_mode = rebuild_for_model_offload(repo_id, dtype)
    else:
        print("Using sequential CPU offload (recommended).")
        try:
            pipe.enable_sequential_cpu_offload()
            offload_mode = "sequential"
        except TypeError as exc:
            print(f"Sequential offload failed ({exc}). Falling back to model CPU offload.")
            pipe, offload_mode = rebuild_for_model_offload(repo_id, dtype)

    print(f"Starting batch generation ({args.count} images)...")
    print(f"Prompt: {args.prompt[:100]}...")

    for index in range(args.count):
        current_seed = resolve_seed(args.seed, index)
        filename = f"{args.output_prefix}_{index + 1:03d}_seed{current_seed}.png"
        print(f"\n[Image {index + 1}/{args.count}] seed={{current_seed}} -> {{filename}}")

        generator = torch.Generator(device=args.device).manual_seed(current_seed)
        t0 = time.time()
        with torch.inference_mode():
            image = pipe(
                prompt=args.prompt,
                generator=generator,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                height=args.height,
                width=args.width,
            ).images[0]

        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

        elapsed = time.time() - t0
        print(f"Tiempo transcurrido: {elapsed}")
        image.save(filename)

    print("Batch complete.")

if __name__ == "__main__":
    main()
