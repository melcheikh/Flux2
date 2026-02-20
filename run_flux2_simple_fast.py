import os
import time
import torch
from diffusers import AutoModel, Flux2Pipeline
from transformers import Mistral3ForConditionalGeneration

repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
dtype = torch.bfloat16

# A veces ayuda a evitar fragmentación VRAM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("Initializing Flux 2 NF4 Pipeline (FAST)...")

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
pipe = Flux2Pipeline.from_pretrained(repo_id, text_encoder=text_encoder, transformer=dit, torch_dtype=dtype)

# Intentar GPU completo (rápido). Si no entra, recién ahí offload.
try:
    print("Moving pipeline to GPU...")
    pipe = pipe.to(device)
    offload = "none"
except torch.cuda.OutOfMemoryError:
    print("OOM -> using sequential CPU offload (slower but better than model_cpu_offload)")
    torch.cuda.empty_cache()
    pipe.enable_sequential_cpu_offload()
    offload = "sequential"

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell..."
generator = torch.Generator(device=device).manual_seed(42)

num_steps = 28  # gran mejora vs 50

print(f"Generating (offload={offload}, steps={num_steps})...")
t0 = time.time()
with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        generator=generator,
        num_inference_steps=num_steps,
        guidance_scale=4.0,
        height=1024,
        width=1024,
    ).images[0]
torch.cuda.synchronize()
print(f"Done in {time.time() - t0:.1f}s")

image.save("flux2_fast.png")
