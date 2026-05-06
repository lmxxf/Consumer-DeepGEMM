"""Debug shapes in grouped GEMM."""
import torch
import sys
sys.path.insert(0, '/root/.cache/huggingface/Consumer-DeepGEMM')
from consumer_deep_gemm.gemm import _e8m0_to_float

M_sum, K, N, num_experts, top_k = 384, 7168, 4096, 256, 6

a_tensor = torch.randn(M_sum, K, device='cuda').to(torch.float8_e4m3fn)
a_scale = torch.ones(M_sum, K // 128, device='cuda', dtype=torch.float32)

b_packed = torch.randint(0, 256, (num_experts, N, K // 2), dtype=torch.uint8, device='cuda')
b_scale = torch.randint(110, 140, (num_experts, N, K // 32), dtype=torch.uint8, device='cuda')
b_scale_f32 = _e8m0_to_float(b_scale)

print(f"a_tensor: {a_tensor.shape} {a_tensor.dtype}")
print(f"a_scale:  {a_scale.shape} {a_scale.dtype}")
print(f"b_packed: {b_packed.shape} {b_packed.dtype}")
print(f"b_scale:  {b_scale_f32.shape} {b_scale_f32.dtype}")
print(f"b_packed[0]: {b_packed[0].shape}  -> K_packed={K//2}, K_full={K}")
print(f"After dequant b[0] should be: [{N}, {K}]")
print(f"NT matmul: a_group [{top_k}, {K}] x b_deq [{N}, {K}].T = [{top_k}, {N}]")
print(f"need_transpose: K_full={K} != k={K} -> {K != K}")
