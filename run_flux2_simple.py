import torch
from diffusers import AutoModel, Flux2Pipeline
from transformers import Mistral3ForConditionalGeneration

# Official 4-bit loading script from flux2/docs/flux2_dev_hf.md
# Fits in ~20GB VRAM (perfect for RTX 5090's 24GB)

print("Initializing Flux 2 NF4 Pipeline (Official Method)...")

repo_id = "diffusers/FLUX.2-dev-bnb-4bit"  # quantized text-encoder and DiT. VAE still in bf16
device = "cuda"
dtype = torch.bfloat16

print(f"Loading Text Encoder from {repo_id}...")
text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    repo_id,
    subfolder="text_encoder",
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Load to CPU first as per docs
)

print(f"Loading Transformer (DiT) from {repo_id}...")
dit = AutoModel.from_pretrained(
    repo_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Load to CPU first as per docs
)

print("Assembling Pipeline...")
pipe = Flux2Pipeline.from_pretrained(repo_id, text_encoder=text_encoder, transformer=dit, torch_dtype=dtype)

print("Enabling CPU Offload...")
pipe.enable_model_cpu_offload()

print("Generating Image...")
prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL + Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

# Using fixed seed for reproducibility
generator = torch.Generator(device=device).manual_seed(42)

image = pipe(
    prompt=prompt,
    generator=generator,
    num_inference_steps=50,
    guidance_scale=4,
    height=1024,  # Explicitly setting resolution to standard Flux
    width=1024,
).images[0]

output_filename = "flux2_official_nf4_output.png"
image.save(output_filename)
print(f"Success! Image saved to {output_filename}")
