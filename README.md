# Consumer-DeepGEMM

DeepGEMM-compatible API implemented with CUTLASS SM120 templates for consumer-grade Blackwell GPUs (RTX 5090, DGX Spark GB10).

## Why

DeepSeek's [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) only supports data-center GPUs: SM90 (Hopper H100) and SM100 (Blackwell B200). Consumer Blackwell (SM120/SM121) uses a completely different MMA instruction set (`mma.sync.aligned` vs GMMA/UMMA), so DeepGEMM's kernels cannot run on these GPUs.

Current workarounds (like [jasl's Triton fallback](https://github.com/jasl/vllm)) work but leave significant performance on the table. NVIDIA's CUTLASS 3.x already ships optimized SM120 templates (Examples 79/80/87) that use hardware-native FP4 MMA instructions — this project wraps them in DeepGEMM's API so vLLM can use them as a drop-in replacement.

## Architecture

```
vLLM calls DeepGEMM Python API (unchanged)
    ↓
Consumer-DeepGEMM (this project)
    ↓
CUTLASS SM120 templates (NVIDIA official, optimized for GeForce/Spark)
    ↓
SM120 mma.sync.aligned.block_scale hardware instructions
```

## Target Hardware

- NVIDIA DGX Spark (GB10, SM121)
- NVIDIA RTX 5090/5080 (SM120)
- Any future consumer Blackwell GPU

## API Coverage

Implements the DeepGEMM functions used by vLLM for DeepSeek V4 inference:

| Function | Status | CUTLASS Source |
|----------|--------|---------------|
| `fp8_gemm_nt` | planned | Example 87 (blockwise FP8) |
| `m_grouped_fp8_fp4_gemm_nt_contiguous` | planned | Example 79d (grouped FP4) |
| `m_grouped_fp8_gemm_nt_contiguous` | planned | Example 79d |
| `fp8_m_grouped_gemm_nt_masked` | planned | Example 79d |
| `tf32_hc_prenorm_gemm` | planned | Custom (CUTLASS BF16/TF32 GEMM + fused prenorm) |
| `fp8_fp4_mqa_logits` | planned | Example 77 (MLA) |
| `fp8_fp4_paged_mqa_logits` | planned | Example 77 (MLA) |
| `fp8_einsum` | planned | CUTLASS batched GEMM |
| `transform_sf_into_required_layout` | planned | Scale factor layout transform |

## Building

```bash
pip install -e .
```

Default install uses the Python fallback only. To build the CUTLASS native
extension inside the DGX Spark/vLLM container:

```bash
CONSUMER_DEEP_GEMM_BUILD_CUDA=1 \
CONSUMER_DEEP_GEMM_CUDA_ARCH=120a \
CUTLASS_PATH=/home/lmxxf/work/deepseek-v4-flash-deployment/DeepGEMM/third-party/cutlass \
pip install -e .
```

For native extension smoke-test:

```bash
python - <<'PY'
import consumer_deep_gemm as dg
print(dg.native_build_info())
PY
```

Inside the `vllm-node-sm120` container, use the one-shot installer:

```bash
cd /work/Consumer-DeepGEMM
./scripts/install_in_vllm_container.sh
```

That does three things:

- installs this package in editable mode, including the top-level `deep_gemm`
  compatibility package
- builds `consumer_deep_gemm._C` for `sm_120a`
- writes `vllm.third_party.deep_gemm` as a shim to `consumer_deep_gemm`, because
  DeepSeek V4 MegaMoE currently imports that hard-coded vendored path
- patches installed vLLM's DeepGEMM support checks to allow SM120/SM121 instead
  of only SM90/SM100

Host-side Docker smoke-test:

```bash
docker run --rm \
  -v /home/lmxxf/work/deepseek-v4-flash-deployment:/work \
  -w /work/Consumer-DeepGEMM \
  vllm-node-sm120:latest \
  bash -lc './scripts/install_in_vllm_container.sh'
```

Expected probe output:

```text
deep_gemm: {'available': True, 'cutlass_sm120_probe': True, 'arch': 'sm_121a'}
vllm.third_party.deep_gemm: {'available': True, 'cutlass_sm120_probe': True, 'arch': 'sm_121a'}
```

The installer handles Docker bind-mount ownership by adding the project path to
Git's `safe.directory` list before `pip install -e .`.

Requires:
- CUDA 12.8+
- PyTorch 2.11+ with SM120 support
- CUTLASS (bundled as submodule)

## Usage with vLLM

```python
# Consumer-DeepGEMM is a drop-in replacement for deep_gemm
import consumer_deep_gemm as deep_gemm
```

## License

Apache 2.0 (same as DeepGEMM and CUTLASS)
