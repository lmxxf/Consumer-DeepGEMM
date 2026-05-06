"""Tiny test to debug fused kernel."""
import torch
import sys
sys.path.insert(0, '/root/.cache/huggingface/Consumer-DeepGEMM')
from consumer_deep_gemm.gemm import _dequant_fp4_block, _e8m0_to_float
from consumer_deep_gemm.triton_moe import triton_fused_fp4_matmul_nt

torch.manual_seed(42)

# Tiny: M=1, N=1, K=64 -> single output value
M, N, K = 1, 1, 64
K_packed = K // 2

# Simple data: A = all 1.0, B = known pattern
a_bf16 = torch.ones(M, K, device='cuda', dtype=torch.bfloat16)

# All zeros except first byte = 0x21 -> low=1 (=0.5), high=2 (=1.0)
b_packed = torch.zeros(N, K_packed, dtype=torch.uint8, device='cuda')
b_packed[0, 0] = 0x21  # high nibble=2 (1.0), low nibble=1 (0.5)

# Scale = 127 -> 2^0 = 1.0
b_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device='cuda')
b_scale_f32 = _e8m0_to_float(b_scale)

# Reference
b_deq = _dequant_fp4_block(b_packed, b_scale_f32, block_k=32)
print(f"b_deq first 4 values: {b_deq[0, :4].tolist()}")
# Should be [0.5, 1.0, 0.0, 0.0, ...]
ref = torch.mm(a_bf16.float(), b_deq.float().t()).to(torch.bfloat16)
print(f"ref result: {ref.item()}")
# Should be 0.5 + 1.0 = 1.5 (only first two values non-zero)

# Fused
out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
triton_fused_fp4_matmul_nt(a_bf16, b_packed, b_scale, out)
print(f"fused result: {out.item()}")
print(f"diff: {abs(ref.item() - out.item())}")

# Medium: M=1, N=32, K=64
print("\n--- M=1, N=32, K=64 ---")
M, N, K = 1, 32, 64
K_packed = K // 2
a_bf16 = torch.ones(M, K, device='cuda', dtype=torch.bfloat16)
b_packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device='cuda')
b_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device='cuda')
b_scale_f32 = _e8m0_to_float(b_scale)

b_deq = _dequant_fp4_block(b_packed, b_scale_f32, block_k=32)
ref = torch.mm(a_bf16.float(), b_deq.float().t()).to(torch.bfloat16)
out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
triton_fused_fp4_matmul_nt(a_bf16, b_packed, b_scale, out)
diff = (ref.float() - out.float()).abs()
print(f"max_diff={diff.max().item():.6f}")
print(f"ref[:5]={ref[0,:5].tolist()}")
print(f"out[:5]={out[0,:5].tolist()}")
