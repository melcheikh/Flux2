import torch
from safetensors.torch import load_file
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_weights(path):
    logger.info(f"Loading checkpoint from {path}...")
    sd = load_file(path)

    # Pick a representative quantized layer
    # e.g., double_blocks.0.img_mlp.0.weight
    prefix = "double_blocks.0.img_mlp.0."

    keys = [k for k in sd.keys() if k.startswith(prefix)]
    logger.info(f"Keys for {prefix}: {keys}")

    if not keys:
        logger.error("No keys found for prefix.")
        return

    weight = sd.get(f"{prefix}weight")
    scale = sd.get(f"{prefix}weight_scale")
    scale_2 = sd.get(f"{prefix}weight_scale_2")

    logger.info(f"Weight: {weight.dtype}, {weight.shape}")
    if scale is not None:
        logger.info(f"Scale (Block): {scale.dtype}, {scale.shape}")
        # Inspect range of FP8 scales
        # float8_e4m3fn range is approx 1e-9 to 448
        s_f32 = scale.to(torch.float32)
        logger.info(f"  Min: {s_f32.min().item()}, Max: {s_f32.max().item()}, Mean: {s_f32.mean().item()}")

    if scale_2 is not None:
        logger.info(f"Scale 2 (Global): {scale_2.dtype}, {scale_2.shape}")
        logger.info(f"  Value: {scale_2.item()}")

    # Analyze Weight bits
    # Check if weights are mostly zero or have a distribution
    w_uint8 = weight.to(torch.uint8)
    logger.info(f"Weight Histogram (0-255): {torch.histc(w_uint8.float(), bins=16, min=0, max=255)}")

    # Test dequantization logic on ONE block
    # NVFP4 E2M1: S(1) E(2) M(1)
    # Binary: SEEM
    # 0: 0.0
    # 1: 0.5 (2^-1 * 1.0)
    # ...
    # Blackwell hardware might expect the two 4-bit values in a specific order.

    # Compare with a BF16 layer if available to see if the mapping is correct
    # final_layer.linear.weight is BF16
    bf16_weight = sd.get("final_layer.linear.weight")
    if bf16_weight is not None:
        logger.info(f"BF16 Weight Sample: {bf16_weight.view(-1)[:5]}")


if __name__ == "__main__":
    path = "/home/martinelcheikh/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev-NVFP4/snapshots/142b87e70bc3006937b7093d89ff287b5f59f071/flux2-dev-nvfp4-mixed.safetensors"
    diagnose_weights(path)
