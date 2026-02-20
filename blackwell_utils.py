import torch
import torch.nn as nn
from typing import Dict
import logging

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
            qact, act_scale, _ = svdq_quantize_w4a4_act_fuse_lora_cuda(
                x,
                fp4=True,
                pad_size=16,  # Blackwell works best with 16-aligned blocks
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


def patch_flux2_with_blackwell(model, state_dict: Dict[str, torch.Tensor]):
    """
    Replaces Linear layers in the transformer with BlackwellLinear and loads NVFP4 weights.
    """
    # Identify modules to patch
    modules_to_patch = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Only patch if we have corresponding quantized weights in state_dict
            if f"{name}.weight" in state_dict and state_dict[f"{name}.weight"].dtype == torch.uint8:
                modules_to_patch.append((name, module))

    logger.info(f"Patching {len(modules_to_patch)} layers with Blackwell NVFP4...")

    for name, module in modules_to_patch:
        # 1. Create BlackwellLinear
        # Use same bias existence and in/out features
        new_module = BlackwellLinear(
            module.in_features, module.out_features, bias=module.bias is not None
        ).to(device=model.device if hasattr(model, "device") else "cpu", dtype=torch.bfloat16)

        # 2. Load weights and scales
        new_module.qweight.copy_(state_dict[f"{name}.weight"])
        new_module.weight_scale.copy_(state_dict[f"{name}.weight_scale"])
        if module.bias is not None:
            if f"{name}.bias" in state_dict:
                new_module.bias.data.copy_(state_dict[f"{name}.bias"])
            else:
                new_module.bias.data.zero_()

        # 3. Replace module
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, new_module)

    logger.info("Flux 2 Transformer patched successfully.")
    return model
