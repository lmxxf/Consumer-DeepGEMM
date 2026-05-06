"""Debug full grouped GEMM mismatch."""
import torch
import sys
sys.path.insert(0, '/root/.cache/huggingface/Consumer-DeepGEMM')
from consumer_deep_gemm.gemm import (
    _dequant_fp8_block, _e8m0_to_float,
    _m_grouped_fp8_fp4_dequant_mm_nt,
)
from consumer_deep_gemm.triton_moe import (
    m_grouped_fp8_fp4_gemm_nt_contiguous_triton,
    triton_dequant_fp4, _ensure_e8m0_scales,
)

torch.manual_seed(42)
M_sum, K, N, num_experts, top_k = 384, 7168, 4096, 256, 6

a_tensor = torch.randn(M_sum, K, device='cuda').to(torch.float8_e4m3fn)
a_scale = torch.ones(M_sum, K // 128, device='cuda', dtype=torch.float32)
b_packed = torch.randint(0, 256, (num_experts, N, K // 2), dtype=torch.uint8, device='cuda')
b_scale = torch.randint(110, 140, (num_experts, N, K // 32), dtype=torch.uint8, device='cuda')
b_scale_f32 = _e8m0_to_float(b_scale)

m_indices = torch.full((M_sum,), -1, dtype=torch.int32, device='cuda')
rows_per_expert = M_sum // top_k
expert_ids = torch.randperm(num_experts)[:top_k]
for i, eid in enumerate(expert_ids):
    m_indices[i * rows_per_expert:(i+1) * rows_per_expert] = eid.item()

a_tuple = (a_tensor, a_scale)
b_tuple = (b_packed, b_scale_f32)

d_ref = torch.empty(M_sum, N, dtype=torch.bfloat16, device='cuda')
d_tri = torch.empty(M_sum, N, dtype=torch.bfloat16, device='cuda')

# Reference
_m_grouped_fp8_fp4_dequant_mm_nt(a_tuple, b_tuple, d_ref, m_indices)

# Manual step-by-step Triton version (bypass the function)
a_deq = _dequant_fp8_block(a_tensor, a_scale)
b_scale_u8 = _ensure_e8m0_scales(b_scale_f32)
active_groups = m_indices[m_indices >= 0].unique()
print(f"Active groups: {active_groups.tolist()}")

d_tri.zero_()
for gid_t in active_groups:
    gid = gid_t.item()
    rows = (m_indices == gid).nonzero(as_tuple=False).flatten()
    if rows.numel() == 0:
        continue

    gs = b_scale_u8[gid]
    deq = triton_dequant_fp4(b_packed[gid], gs)
    print(f"Expert {gid}: deq shape={deq.shape}, rows={rows.numel()}")

    a_group = a_deq.index_select(0, rows).to(torch.float32)
    result = torch.mm(a_group, deq.to(torch.float32).t())
    d_tri.index_copy_(0, rows, result.to(d_tri.dtype))

diff = (d_ref.float() - d_tri.float()).abs()
print(f"\nManual loop: max_diff={diff.max().item():.6f}")

# Now test the actual function
d_tri2 = torch.empty(M_sum, N, dtype=torch.bfloat16, device='cuda')
m_grouped_fp8_fp4_gemm_nt_contiguous_triton(a_tuple, b_tuple, d_tri2, m_indices)
diff2 = (d_ref.float() - d_tri2.float()).abs()
print(f"Function:    max_diff={diff2.max().item():.6f}")

# Find where the error is
if diff2.max().item() > 1.0:
    bad_rows = (diff2.max(dim=1).values > 1.0).nonzero().flatten()
    print(f"Bad rows: {bad_rows[:10].tolist()}")
    for r in bad_rows[:3]:
        r = r.item()
        gid = m_indices[r].item()
        print(f"  row {r}, expert {gid}: ref={d_ref[r,:5]}, tri={d_tri2[r,:5]}")
