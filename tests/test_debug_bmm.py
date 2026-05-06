"""Debug bmm shapes in grouped GEMM."""
import torch
import sys
sys.path.insert(0, '/root/.cache/huggingface/Consumer-DeepGEMM')
from consumer_deep_gemm.gemm import (
    _dequant_fp4_block, _dequant_fp8_block, _e8m0_to_float,
    _m_grouped_fp8_fp4_dequant_mm_nt, _select_group_weight_nt,
)
from consumer_deep_gemm.triton_moe import triton_dequant_fp4

torch.manual_seed(42)
M_sum, K, N, num_experts, top_k = 384, 7168, 4096, 256, 6
K_packed = K // 2

a_tensor = torch.randn(M_sum, K, device='cuda').to(torch.float8_e4m3fn)
a_scale = torch.ones(M_sum, K // 128, device='cuda', dtype=torch.float32)
a_deq = _dequant_fp8_block(a_tensor, a_scale)

b_packed = torch.randint(0, 256, (num_experts, N, K_packed), dtype=torch.uint8, device='cuda')
b_scale = torch.randint(110, 140, (num_experts, N, K // 32), dtype=torch.uint8, device='cuda')
b_scale_f32 = _e8m0_to_float(b_scale)

m_indices = torch.full((M_sum,), -1, dtype=torch.int32, device='cuda')
rows_per_expert = M_sum // top_k
expert_ids = torch.randperm(num_experts)[:top_k]
for i, eid in enumerate(expert_ids):
    m_indices[i * rows_per_expert:(i+1) * rows_per_expert] = eid.item()

# Reference: single expert
gid = expert_ids[0].item()
rows = (m_indices == gid).nonzero(as_tuple=False).flatten()
print(f"Expert {gid}: rows={rows.shape}")

# Reference dequant
ref_deq = _dequant_fp4_block(b_packed[gid], b_scale_f32[gid], block_k=32)
print(f"ref_deq shape: {ref_deq.shape}")  # should be [N, K]
ref_deq_nt = _select_group_weight_nt(ref_deq, K)
print(f"ref_deq_nt shape: {ref_deq_nt.shape}")  # should be [N, K]

# Triton dequant
tri_deq = triton_dequant_fp4(b_packed[gid], b_scale_f32[gid])
print(f"tri_deq shape: {tri_deq.shape}")  # should be [N, K]

# Check dequant match
diff = (ref_deq.float() - tri_deq.float()).abs().max().item()
print(f"dequant match: max_diff={diff}")

# Single expert matmul
a_g = a_deq.index_select(0, rows).to(torch.float32)
print(f"a_g shape: {a_g.shape}")  # [64, 7168]
print(f"tri_deq shape for mm: {tri_deq.shape}")  # [4096, 7168]

ref_result = torch.mm(a_g, ref_deq_nt.to(torch.float32).t())
tri_result = torch.mm(a_g, tri_deq.to(torch.float32).t())
print(f"ref_result shape: {ref_result.shape}")
print(f"tri_result shape: {tri_result.shape}")
diff2 = (ref_result - tri_result).abs().max().item()
print(f"single expert mm match: max_diff={diff2}")

# Now test the issue: b_scale is float32 but triton expects uint8
# The E8M0 conversion might be wrong
b_scale_u8_ref = b_scale  # already uint8 in test
from consumer_deep_gemm.triton_moe import _ensure_e8m0_scales
b_scale_u8 = _ensure_e8m0_scales(b_scale_f32)
print(f"\nb_scale_f32[0,0,:5]: {b_scale_f32[0,0,:5]}")
print(f"b_scale_u8[0,0,:5]:  {b_scale_u8[0,0,:5]}")
print(f"b_scale[0,0,:5]:     {b_scale[0,0,:5]}")
diff3 = (b_scale_u8.float() - b_scale.float()).abs().max().item()
print(f"E8M0 roundtrip diff: {diff3}")
