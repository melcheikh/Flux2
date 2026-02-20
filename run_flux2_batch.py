import argparse
import random

import torch
from diffusers import AutoModel, Flux2Pipeline
from transformers import Mistral3ForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Flux 2 NF4 Batch Image Generation")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL + Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom.",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width (default: 1024)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height (default: 1024)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=4.0,
        help="Guidance scale (default: 4.0)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed. If provided, subsequent images use seed+1. If not, random seeds are used.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="flux2_batch",
        help="Prefix for output filenames",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
    device = "cuda"
    dtype = torch.bfloat16

    print("Initializing Flux 2 NF4 Pipeline for Batch Generation...")

    print(f"Loading Text Encoder from {repo_id}...")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading Transformer (DiT) from {repo_id}...")
    dit = AutoModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("Assembling Pipeline...")
    pipe = Flux2Pipeline.from_pretrained(
        repo_id, text_encoder=text_encoder, transformer=dit, torch_dtype=dtype
    )

    print("Enabling CPU Offload...")
    pipe.enable_model_cpu_offload()

    print(f"Starting generation of {args.count} images...")
    print(f"Prompt: {args.prompt[:100]}...")

    for i in range(args.count):
        if args.seed is not None:
            current_seed = args.seed + i
        else:
            current_seed = random.randint(0, 2**32 - 1)

        print(f"\n[Image {i + 1}/{args.count}] Using seed: {current_seed}")

        generator = torch.Generator(device=device).manual_seed(current_seed)

        image = pipe(
            prompt=args.prompt,
            generator=generator,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
        ).images[0]

        output_filename = f"{args.output_prefix}_{i + 1}_seed_{current_seed}.png"
        image.save(output_filename)
        print(f"Saved to {output_filename}")

    print("\nBatch processing complete!")


if __name__ == "__main__":
    main()
