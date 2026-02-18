# Models Used in Flux 2 NF4 Setup

This repository contains scripts optimized for running Flux 2 with 4-bit Normal Float (NF4) quantization, enabling inference on consumer GPUs with ~24GB VRAM.

## Primary Model

| Component | Model Name | HuggingFace Link | Description |
|-----------|------------|------------------|-------------|
| **Quantized Weights** | `diffusers/FLUX.2-dev-bnb-4bit` | [Link](https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit) | Contains the Transformer (DiT) and Text Encoder quantized in 4-bit (bnf4) for low VRAM usage. |
| **Base Configuration** | `black-forest-labs/FLUX.1-dev` | [Link](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Used as the base for Scheduler, VAE, and Tokenizer configurations (compatible with Flux 2 for these components). |

## Dependencies

- **Transformers**: `Mistral3ForConditionalGeneration` (Text Encoder)
- **Diffusers**: `Flux2Pipeline` (Inference Pipeline)

## Usage

The script `run_flux2_simple.py` automatically downloads the necessary shards from the `diffusers/FLUX.2-dev-bnb-4bit` repository using `huggingface_hub` snapshot download or direct pipeline loading.

**Note**: You must accept the license agreement on the [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) model card page before access is granted via your HuggingFace token.
