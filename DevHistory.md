# Consumer-DeepGEMM 开发记录

用 CUTLASS SM120 模板实现 DeepGEMM 兼容 API，给消费级 Blackwell GPU（DGX Spark, RTX 5090）用。

---

## 背景

### 为什么需要这个项目

DeepSeek V4 Flash (280B) 在 DGX Spark 双机上跑通了（见 deepseek-v4-flash-deployment），但用的是 jasl/vllm fork 的 Triton fallback kernel。实测速度 ~3.9 tok/s（含 prefill），体感慢。

想提速 → 调查了 DeepGEMM → 发现不能简单改。

### SM120 是三不像

三种 Blackwell 用了完全不同的矩阵乘法硬件指令：

| 架构 | MMA 指令 | 适用场景 |
|------|---------|---------|
| SM90 (Hopper H100) | GMMA (warp group, 64×N) | 数据中心 |
| SM100 (DC Blackwell B200) | UMMA (tcgen05, 新指令) | 数据中心 |
| SM120 (桌面 Blackwell 5090/Spark) | mma.sync.aligned (warp, 16×8) | 消费级 |

SM120 的 MMA 更接近 SM80 (Ampere)，但加了 FP4/FP8 新数据类型。DeepGEMM 的 SM90/SM100 kernel 完全不能用在 SM120 上——不是改个 if 的事，是指令集不同。

### 三条路的比较

| 方案 | 做了什么 | 性能预期 |
|------|---------|---------|
| jasl Triton fallback（现状） | Triton 重写所有 DeepGEMM kernel | 基线 ~3.9 tok/s |
| 改造 DeepGEMM 加 SM120 | 从头写 SM120 kernel 实现 | 工作量太大，放弃 |
| **Consumer-DeepGEMM（本项目）** | 用 CUTLASS SM120 模板封装 DeepGEMM API | 预期快很多 |

### CUTLASS 已有的 SM120 模板

NVIDIA 的 CUTLASS 3.x 已经给 SM120（GeForce/桌面 Blackwell）写好了完整模板：

- **Example 79**: `blackwell_geforce_gemm` — FP4 GEMM + Grouped GEMM (MoE)
- **Example 80**: `blackwell_geforce_sparse_gemm` — 稀疏 GEMM
- **Example 87**: `blackwell_geforce_gemm_blockwise` — blockwise scaling FP8 GEMM
- **Example 77**: `blackwell_fmha` — MLA 注意力（含 `77_blackwell_mla.cu`）

NVIDIA 官方针对 GeForce 硬件调过的 kernel，用 `mma.sync.aligned.block_scale` 硬件原生 FP4 指令。

### 架构设计

不 fork DeepGEMM——SM90/SM100 的代码一行都不需要。独立项目，实现相同的 Python API：

```
vLLM 调用 deep_gemm.fp8_gemm_nt(...)
    ↓
import consumer_deep_gemm as deep_gemm  （替换）
    ↓
Consumer-DeepGEMM 调用 CUTLASS SM120 模板
    ↓
SM120 mma.sync.aligned.block_scale 硬件指令
```

---

## 第一阶段：项目骨架 + PyTorch Fallback（2026-05-02）

### 完成项

- 项目结构搭建
- DeepGEMM 全部 Python API 的 stub 实现（纯 PyTorch fallback）
- 接口分析：vLLM 通过 `_lazy_init()` 用 `getattr` 动态加载 DeepGEMM 函数

### API 覆盖

| 模块 | 函数 | fallback 状态 |
|------|------|-------------|
| gemm | fp8_gemm_{nt,nn,tn,tt} | ✅ PyTorch |
| gemm | fp8_fp4_gemm_{nt,nn} | ✅ PyTorch |
| gemm | m_grouped_fp8{,_fp4}_gemm_*_contiguous | ✅ PyTorch |
| gemm | m_grouped_fp8{,_fp4}_gemm_*_masked | ✅ PyTorch |
| gemm | bf16_gemm_{nt,nn,tn,tt} | ✅ PyTorch |
| hyperconnection | tf32_hc_prenorm_gemm | ✅ PyTorch |
| attention | fp8_fp4_mqa_logits | ✅ PyTorch |
| attention | fp8_fp4_paged_mqa_logits | ✅ PyTorch |
| einsum | einsum, fp8_einsum | ✅ PyTorch |
| layout | transform_sf_into_required_layout | ✅ passthrough |
| utils | set/get_num_sms, set/get_tc_util | ✅ |

### 下一步

- Phase 2: 用 CUTLASS SM120 模板替换 PyTorch fallback
  - 优先级 1: FP4 GEMM (Example 79) — MoE 专家层，调用最频繁
  - 优先级 2: MLA attention (Example 77) — 注意力层
  - 优先级 3: HC prenorm GEMM — mHC 层
  - 优先级 4: FP8 GEMM (Example 87) — 其他矩阵乘
- 在 vllm-sm120 容器里编译测试
- 对比 jasl Triton fallback 速度

---

## TileLang 调研备忘（2026-05-02）

DeepSeek V4 官方 inference 代码用 TileLang（北大王磊开发，基于 TVM），不是 DeepGEMM。TileLang 在 MLA 上比 Triton 快 5 倍（H100 实测）。但 TileLang 0.1.8 也不支持 SM120。

TileLang 用户：DeepSeek（生产级）、微软（BitBLAS）、AMD（MI300X 官方博客）、华为（昇腾适配器）。

和 Consumer-DeepGEMM 无关，记在这备查。
