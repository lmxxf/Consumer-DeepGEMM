import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from consumer_deep_gemm.gemm import _reorder_scale_to_cutlass_sf_atom

N, k_blocks = 128, 8
scale = torch.arange(N * k_blocks, dtype=torch.uint8).reshape(N, k_blocks)

py = _reorder_scale_to_cutlass_sf_atom(scale, N, k_blocks)

# C++ reorder via the function (manually replicate the formula)
cpp = torch.full_like(py, 127)
for r in range(N):
    for c in range(k_blocks):
        mt = r // 128
        m_in_tile = r % 128
        m_32 = m_in_tile // 32
        m_in_32 = m_in_tile % 32
        kt = c // 4
        k_in_4 = c % 4
        k_tiles = (k_blocks + 3) // 4
        atom_size = 32 * 4 * 4
        tile_offset = (mt * k_tiles + kt) * atom_size
        in_tile = m_in_32 * 16 + m_32 * 4 + k_in_4
        cpp[tile_offset + in_tile] = scale[r, c]

match = torch.equal(py, cpp)
print(f"Python vs C++ reorder match: {match}")
if not match:
    diff_idx = (py != cpp).nonzero(as_tuple=False).flatten()
    print(f"Mismatch at {diff_idx.numel()} positions out of {py.numel()}")
    for i in diff_idx[:10].tolist():
        print(f"  offset {i}: py={py[i].item()} cpp={cpp[i].item()}")
