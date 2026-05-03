import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import consumer_deep_gemm as dg
from consumer_deep_gemm.gemm import _m_grouped_fp8_fp4_fallback_nt

M, K, N, G = 128, 256, 128, 1
a = torch.ones((M, K), dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
b = torch.full((G, N, K // 2), 0x22, dtype=torch.uint8).view(torch.int8).cuda()
a_s = torch.ones((M, K // 128), dtype=torch.float32).cuda()
a_s[:, 0] = 2.0
a_s[:, 1] = 4.0
b_s = torch.ones((G, N, K // 32), dtype=torch.float32).cuda()
d = torch.zeros((M, N), dtype=torch.bfloat16).cuda()
dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
    (a, a_s), (b, b_s), d, torch.zeros(M, dtype=torch.int32).cuda()
)
torch.cuda.synchronize()

d_ref = torch.zeros((M, N), dtype=torch.bfloat16).cuda()
_m_grouped_fp8_fp4_fallback_nt(
    (a, a_s), (b, b_s), d_ref, torch.zeros(M, dtype=torch.int32).cuda()
)

print(f"nat d[0,0]={d[0, 0].item():.1f}")
print(f"ref d[0,0]={d_ref[0, 0].item():.1f}")
nan_count = d.isnan().sum().item()
inf_count = d.isinf().sum().item()
print(f"nan={nan_count} inf={inf_count}")
diff = (d.float() - d_ref.float()).abs()
valid = ~d.isnan() & ~d.isinf()
if valid.any():
    max_diff_val = diff[valid].max().item()
    print(f"max valid diff={max_diff_val:.1f}")
else:
    print("all nan/inf!")

# Test with sfb cache disabled to isolate SFA GPU reorder
from consumer_deep_gemm import gemm
gemm._sfb_reorder_cache.clear()

# Monkey-patch to disable SFB cache
original_get = gemm._get_or_reorder_sfb
def no_cache_sfb(b_scale):
    b_scale_e8m0 = gemm._float_scale_to_e8m0(b_scale)
    return b_scale_e8m0, False  # not pre-reordered, let C++ do it
gemm._get_or_reorder_sfb = no_cache_sfb

d2 = torch.zeros((M, N), dtype=torch.bfloat16).cuda()
dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
    (a, a_s), (b, b_s), d2, torch.zeros(M, dtype=torch.int32).cuda()
)
torch.cuda.synchronize()
diff2 = (d2.float() - d_ref.float()).abs()
print(f"\nSFB cache disabled (C++ does SFB reorder):")
print(f"  max diff={diff2.max().item():.1f}")
print(f"  d2[0,:3]={d2[0, :3].tolist()}")

# Restore
gemm._get_or_reorder_sfb = original_get
