import argparse
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import AutoModel, Flux2Pipeline
from transformers import Mistral3ForConditionalGeneration

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux 2 NF4 fast-ish run with safe offload.")
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
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (default: 42)")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument(
        "--offload",
        type=str,
        default="sequential",
        choices=["sequential", "model", "none"],
        help="Offload strategy: sequential (fastest-safe), model (slow), none (try full GPU)",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 matmul for extra speed (may slightly affect quality)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
    dtype = torch.bfloat16

    print("Initializing Flux 2 NF4 Pipeline (FAST, safe offload)...")

    print(f"Loading Text Encoder from {repo_id} (CPU first)...")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
        device_map="cpu",
    )

    print(f"Loading Transformer (DiT) from {repo_id} (CPU first)...")
    dit = AutoModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        torch_dtype=dtype,
        device_map="cpu",
    )

    print("Assembling Pipeline...")
    pipe = Flux2Pipeline.from_pretrained(
        repo_id,
        text_encoder=text_encoder,
        transformer=dit,
        torch_dtype=dtype,
    )

    offload_mode = args.offload
    if args.offload == "none":
        try:
            print("Attempting full GPU residency...")
            pipe = pipe.to(args.device)
            offload_mode = "none"
        except torch.cuda.OutOfMemoryError:
            print("OOM -> falling back to sequential CPU offload")
            torch.cuda.empty_cache()
            pipe.enable_sequential_cpu_offload()
            offload_mode = "sequential"
    elif args.offload == "model":
        print("Using model CPU offload (slower, safest).")
        pipe.enable_model_cpu_offload()
        offload_mode = "model"
    else:
        print("Using sequential CPU offload (recommended).")
        try:
            pipe.enable_sequential_cpu_offload()
        except TypeError as exc:
            print(f"Sequential offload failed ({exc}). Falling back to model CPU offload.")
            pipe.enable_model_cpu_offload()
            offload_mode = "model"
        else:
            offload_mode = "sequential"

    print(f"Generating (offload={{offload_mode}}, steps={{args.steps}})...")
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

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
    print(f"Done in {{elapsed:.1f}}s")

    output_path = "flux2_fast.png"
    image.save(output_path)
    print(f"Saved to {{output_path}}")

if __name__ == "__main__":
    main()