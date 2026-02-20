import torch


def test_scaled_mm():
    print(f"Torch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Blackwell SM 12.0 check
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {major}.{minor}")

    if major < 12:
        print("NOT A BLACKWELL GPU. Native NVFP4 _scaled_mm may not work as expected.")

    try:
        # Dummy NVFP4-like setup
        # bfl weights are (out, in // 2) packed uint8
        # torch._scaled_mm expects float4_e2m1fn_x2 (which is packed uint8)

        M, N, K = 128, 128, 128

        # A, B must be uint8 (packed fp4)
        A = torch.randint(0, 255, (M, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)
        B = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda").view(torch.float4_e2m1fn_x2)

        # Scales must be float8_e4m3fn
        # BFL weight_scale is (out, in // 16)
        scale_a = torch.ones((M, K // 16), dtype=torch.float8_e4m3fn, device="cuda")
        scale_b = torch.ones((N, K // 16), dtype=torch.float8_e4m3fn, device="cuda")

        print("Attempting torch._scaled_mm...")
        # Note: _scaled_mm signature changed in nightly.
        # Check signature or use common nightly pattern.
        # result = torch._scaled_mm(mat1, mat2, scale_a, scale_b, ...)

        res = torch._scaled_mm(A, B.t(), scale_a, scale_b, out_dtype=torch.bfloat16)
        print(f"SUCCESS! Result shape: {res.shape}")

    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    test_scaled_mm()
