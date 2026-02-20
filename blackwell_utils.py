import logging
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

try:
    from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda

    HAS_NUNCHAKU = True
except ImportError:
    HAS_NUNCHAKU = False

logger = logging.getLogger("BlackwellUtils")


class BlackwellLinear(nn.Module):
    """
    Native Blackwell NVFP4 Linear Layer using torch._scaled_mm.
    Optimized for BFL FLUX.2-dev-NVFP4 weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # BFL weights: (out, in // 2) uint8
        # BFL scales: (out, in // 16) float8_e4m3fn
        self.register_buffer("qweight", torch.empty((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer(
            "weight_scale", torch.empty((out_features, in_features // 16), dtype=torch.float8_e4m3fn)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

    def quantize_activations(self, x: torch.Tensor):
        if HAS_NUNCHAKU:
            # Use nunchaku's fast CUDA kernel for dynamic FP4 quantization
            # It returns (q_input, scales, lora_act_out)
            # Blackwell expects scale_a to be (M, K // 16) float8_e4m3fn
            # Nunchaku's oscales is (K // 16, M_pad)
            M, K = x.shape
            pad = 16
            M_pad = ((M + pad - 1) // pad) * pad

            lora_down = x.new_empty((K, 0))
            lora_act_out = x.new_empty((M_pad, 0), dtype=torch.float32)

            qact, act_scale, _ = svdq_quantize_w4a4_act_fuse_lora_cuda(
                x,
                lora_down=lora_down,
                lora_act_out=lora_act_out,
                fp4=True,
                pad_size=pad,  # Blackwell works best with 16-aligned blocks
            )
            # torch._scaled_mm expects act_scale to be (M, K // 16)
            return qact, act_scale.t()
        else:
            # Slower fallback for activation quantization if nunchaku is missing
            M, K = x.shape
            x_blocked = x.view(M, K // 16, 16)
            scales = x_blocked.abs().max(dim=2).values / 6.0
            scales = scales.clamp(min=1e-12).to(torch.float8_e4m3fn)

            # This is a very rough approximation, real bit-packing is handled by _scaled_mm
            # if we pass proper dtypes. However, we need to pass uint8 to _scaled_mm.
            # For now, we MUST have nunchaku for the fast path.
            raise RuntimeError("Nunchaku is required for fast Blackwell NVFP4 activation quantization.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # 1. Dynamic Quantization
        qact, act_scale = self.quantize_activations(x)

        # 2. Blackwell hardware-accelerated MatMul
        # torch._scaled_mm(A, B.t(), ...)
        # A (Quantized Activation): (M, K) -> VIEW as float4_e2m1fn_x2
        # B (Quantized Weight): (N, K) -> VIEW as float4_e2m1fn_x2
        # Note: BFL weights are stored as (N, K//2) uint8, which matches Blackwell's B.t() expectation

        res = torch._scaled_mm(
            qact.view(torch.float4_e2m1fn_x2),
            self.qweight.view(torch.float4_e2m1fn_x2),
            scale_a=act_scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
        )

        # Trim padding if any (nunchaku pads to 16/256)
        if res.shape[0] > x.shape[0]:
            res = res[: x.shape[0]]

        if self.bias is not None:
            res = res + self.bias

        return res.view(*orig_shape[:-1], self.out_features)


def _resolve_state_key(state_dict: Dict[str, torch.Tensor], key: str, prefixes: Iterable[str]) -> str | None:
    for prefix in prefixes:
        candidate = f"{prefix}{key}"
        if candidate in state_dict:
            return candidate
    return None


def _nvfp4_key_for_module(name: str) -> Optional[Tuple[str, Optional[int]]]:
    if name.startswith("transformer_blocks."):
        remainder = name[len("transformer_blocks.") :]
        block_id, _, suffix = remainder.partition(".")
        if not block_id.isdigit():
            return None
        if suffix == "attn.to_q":
            return f"double_blocks.{block_id}.img_attn.qkv", 0
        if suffix == "attn.to_k":
            return f"double_blocks.{block_id}.img_attn.qkv", 1
        if suffix == "attn.to_v":
            return f"double_blocks.{block_id}.img_attn.qkv", 2
        if suffix == "attn.to_out.0":
            return f"double_blocks.{block_id}.img_attn.proj", None
        if suffix == "ff.linear_in":
            return f"double_blocks.{block_id}.img_mlp.0", None
        if suffix == "ff.linear_out":
            return f"double_blocks.{block_id}.img_mlp.2", None
        if suffix == "ff_context.linear_in":
            return f"double_blocks.{block_id}.txt_mlp.0", None
        if suffix == "ff_context.linear_out":
            return f"double_blocks.{block_id}.txt_mlp.2", None
        return None

    if name.startswith("single_transformer_blocks."):
        remainder = name[len("single_transformer_blocks.") :]
        block_id, _, suffix = remainder.partition(".")
        if not block_id.isdigit():
            return None
        if suffix == "attn.to_qkv_mlp_proj":
            return f"single_blocks.{block_id}.linear1", None
        if suffix == "attn.to_out":
            return f"single_blocks.{block_id}.linear2", None
    return None


def patch_flux2_with_blackwell(model, state_dict: Dict[str, torch.Tensor]):
    """
    Replaces Linear layers in the transformer with BlackwellLinear and loads NVFP4 weights.
    """
    prefixes = ("", "transformer.", "model.")

    # Identify modules to patch
    modules_to_patch = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        mapping = _nvfp4_key_for_module(name)
        if mapping is None:
            continue
        base_key, slice_index = mapping

        weight_key = _resolve_state_key(state_dict, f"{base_key}.weight", prefixes)
        if weight_key is None:
            continue
        if state_dict[weight_key].dtype != torch.uint8:
            continue

        scale_key = _resolve_state_key(state_dict, f"{base_key}.weight_scale", prefixes)
        if scale_key is None:
            logger.warning("Missing weight_scale for %s; skipping.", name)
            continue
        if state_dict[scale_key].dtype != torch.float8_e4m3fn:
            logger.warning("Unexpected weight_scale dtype for %s: %s", name, state_dict[scale_key].dtype)
            continue

        modules_to_patch.append((name, module, weight_key, scale_key, slice_index))

    logger.info("Patching %s layers with Blackwell NVFP4...", len(modules_to_patch))

    for name, module, weight_key, scale_key, slice_index in modules_to_patch:
        # 1. Create BlackwellLinear
        # Use same bias existence and in/out features
        new_module = BlackwellLinear(
            module.in_features, module.out_features, bias=module.bias is not None
        ).to(device=model.device if hasattr(model, "device") else "cpu", dtype=torch.bfloat16)

        # 2. Load weights and scales
        weight = state_dict[weight_key]
        scale = state_dict[scale_key]
        if slice_index is not None:
            start = slice_index * module.out_features
            end = start + module.out_features
            weight = weight[start:end]
            scale = scale[start:end]

        new_module.qweight.copy_(weight)
        new_module.weight_scale.copy_(scale)
        if module.bias is not None:
            bias_key = _resolve_state_key(state_dict, f"{name}.bias", prefixes)
            if bias_key is not None:
                new_module.bias.data.copy_(state_dict[bias_key])
            else:
                new_module.bias.data.zero_()

        # 3. Replace module
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, new_module)

    logger.info("Flux 2 Transformer patched successfully.")
    return model
