"""Minimal bmm debug: find where the diff comes from."""
import torch
import sys
sys.path.insert(0, '/root/.cache/huggingface/Consumer-DeepGEMM')
from consumer_deep_gemm.gemm import _dequant_fp8_block, _e8m0_to_float
from consumer_deep_gemm.triton_moe import triton_dequant_fp4, _ensure_e8m0_scales

torch.manual_seed(42)
M_sum, K, N, num_experts, top_k = 384, 7168, 4096, 256, 6

a_tensor = torch.randn(M_sum, K, device='cuda').to(torch.float8_e4m3fn)
a_scale = torch.ones(M_sum, K // 128, device='cuda', dtype=torch.float32)
a_deq = _dequant_fp8_block(a_tensor, a_scale)

b_packed = torch.randint(0, 256, (num_experts, N, K // 2), dtype=torch.uint8, device='cuda')
b_scale = torch.randint(110, 140, (num_experts, N, K // 32), dtype=torch.uint8, device='cuda')
b_scale_f32 = _e8m0_to_float(b_scale)
b_scale_u8 = _ensure_e8m0_scales(b_scale_f32)

m_indices = torch.full((M_sum,), -1, dtype=torch.int32, device='cuda')
rows_per = M_sum // top_k
expert_ids = torch.randperm(num_experts)[:top_k]
for i, eid in enumerate(expert_ids):
    m_indices[i * rows_per:(i+1) * rows_per] = eid.item()

active_groups = m_indices[m_indices >= 0].unique()
print(f"active_groups: {active_groups.tolist()}")

# === Loop version (reference) ===
d_loop = torch.zeros(M_sum, N, dtype=torch.bfloat16, device='cuda')
group_rows_list = []
group_ids_list = []
for gid_t in active_groups:
    gid = gid_t.item()
    rows = (m_indices == gid).nonzero(as_tuple=False).flatten()
    if rows.numel() == 0:
        continue
    group_rows_list.append(rows)
    group_ids_list.append(gid)
    gs = b_scale_u8[gid]
    b_deq = triton_dequant_fp4(b_packed[gid], gs)
    a_g = a_deq.index_select(0, rows).to(torch.float32)
    res = torch.mm(a_g, b_deq.to(torch.float32).t())
    d_loop.index_copy_(0, rows, res.to(d_loop.dtype))

# === BMM version ===
n_groups = len(group_ids_list)
max_rows = max(r.numel() for r in group_rows_list)
print(f"n_groups={n_groups}, max_rows={max_rows}")

# Dequant all active weights
b_deq_list = []
for gid in group_ids_list:
    gs = b_scale_u8[gid]
    deq = triton_dequant_fp4(b_packed[gid], gs)
    b_deq_list.append(deq)

# Build batched A: [n_groups, max_rows, K]
a_batched = torch.zeros(n_groups, max_rows, K, dtype=torch.float32, device='cuda')
for i, rows in enumerate(group_rows_list):
    a_batched[i, :rows.numel()] = a_deq.index_select(0, rows).to(torch.float32)

# Build batched B: [n_groups, N, K]
b_batched = torch.stack(b_deq_list, dim=0).to(torch.float32)
print(f"a_batched: {a_batched.shape}, b_batched: {b_batched.shape}")

# BMM: [n_groups, max_rows, K] x [n_groups, K, N] -> [n_groups, max_rows, N]
result_batched = torch.bmm(a_batched, b_batched.transpose(1, 2))
print(f"result_batched: {result_batched.shape}")

# Scatter
d_bmm = torch.zeros(M_sum, N, dtype=torch.bfloat16, device='cuda')
for i, rows in enumerate(group_rows_list):
    d_bmm.index_copy_(0, rows, result_batched[i, :rows.numel()].to(d_bmm.dtype))

# Compare
diff = (d_loop.float() - d_bmm.float()).abs()
print(f"\nmax_diff={diff.max().item():.6f}")
if diff.max().item() > 0.01:
    bad_row = diff.max(dim=1).values.argmax().item()
    bad_col = diff[bad_row].argmax().item()
    print(f"bad_row={bad_row}, bad_col={bad_col}")
    print(f"  loop: {d_loop[bad_row, bad_col].item()}")
    print(f"  bmm:  {d_bmm[bad_row, bad_col].item()}")
    gid = m_indices[bad_row].item()
    print(f"  expert={gid}")
    # Check per-group
    for i, (gid2, rows) in enumerate(zip(group_ids_list, group_rows_list)):
        if gid2 == gid:
            row_in_group = (rows == bad_row).nonzero().item()
            print(f"  group_idx={i}, row_in_group={row_in_group}")
            # Compare single-expert mm vs bmm slice
            a_g = a_deq.index_select(0, rows).to(torch.float32)
            b_g = b_deq_list[i].to(torch.float32)
            ref_single = torch.mm(a_g, b_g.t())
            bmm_single = result_batched[i, :rows.numel()]
            d2 = (ref_single - bmm_single).abs()
            print(f"  single mm vs bmm slice: max_diff={d2.max().item():.6f}")
else:
    print("PASS!")
