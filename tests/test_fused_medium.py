"""Medium test to find the size threshold where fused kernel breaks."""
import torch
import sys
sys.path.insert(0, '/root/.cache/huggingface/Consumer-DeepGEMM')
from consumer_deep_gemm.gemm import _dequant_fp4_block, _e8m0_to_float
from consumer_deep_gemm.triton_moe import triton_fused_fp4_matmul_nt

torch.manual_seed(42)

for M, N, K in [
    (1, 32, 64),
    (1, 32, 128),
    (1, 32, 256),
    (1, 32, 512),
    (1, 32, 1024),
    (1, 32, 2048),
    (1, 32, 4096),
    (1, 32, 7168),
    (1, 128, 7168),
    (1, 4096, 7168),
    (64, 32, 64),
    (64, 32, 7168),
    (64, 4096, 7168),
]:
    K_packed = K // 2
    a_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b_packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device='cuda')
    b_scale = torch.randint(110, 140, (N, K // 32), dtype=torch.uint8, device='cuda')
    b_scale_f32 = _e8m0_to_float(b_scale)

    b_deq = _dequant_fp4_block(b_packed, b_scale_f32, block_k=32)
    ref = torch.mm(a_bf16.float(), b_deq.float().t()).to(torch.bfloat16)

    out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    triton_fused_fp4_matmul_nt(a_bf16, b_packed, b_scale, out)

    diff = (ref.float() - out.float()).abs()
    nz = ref.float().abs() > 1e-6
    rel_err = (diff[nz] / ref.float().abs()[nz]).max().item() if nz.any() else 0.0
    status = "PASS" if rel_err < 0.02 else "FAIL"
    print(f"[{M:4d},{N:5d},{K:5d}] max_diff={diff.max().item():10.2f}  rel_err={rel_err:.6f}  {status}")
