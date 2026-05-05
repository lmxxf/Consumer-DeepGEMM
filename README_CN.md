# Consumer-DeepGEMM（中文说明）

消费级 Blackwell GPU（SM120/SM121）上的 DeepGEMM 兼容替代方案。

## 解决什么问题

DeepSeek V4 Flash 280B 在 DGX Spark / RTX 5090 上用 vLLM 跑推理时，FP4 MoE 路径走的 Marlin kernel 会**静默算错**——不报错，但输出是垃圾。中文短回复看着正常，英文/代码/长输出从第一个 token 开始跑飞。

根因：Marlin 的 FP4 MMA 指令在 SM120+ 上行为不一致，某些专家组合的计算结果是错的。

本项目用 CUTLASS SM120 模板重写了 FP4 MoE 的 GEMM 计算路径，输出正确。

## 文件说明

### 真正干活的部分（替换 Marlin FP4 MoE kernel）

| 路径 | 作用 |
|------|------|
| `csrc/cutlass_mxfp8_mxfp4_probe.cu` | **核心 CUDA kernel**：用 CUTLASS SM120 模板实现 FP8×FP4 grouped GEMM，构造 256 个专家的 pointer array，做 SfAtom scale 重排，真实 launch。**这一个文件修复了垃圾输出。** |
| `csrc/cutlass_sm120_probe.cu` | SM120 硬件探针 |
| `csrc/bindings.cpp` | PyBind11 绑定入口 |
| `consumer_deep_gemm/gemm.py` | Python 入口：`m_grouped_fp8_fp4_gemm_nt_contiguous()`，优先走 native kernel，失败退回 Python dequant |
| `consumer_deep_gemm/native.py` | CUDA 扩展加载器 |
| `consumer_deep_gemm/layout.py` | Scale factor SfAtom 重排逻辑 |
| `scripts/install_in_vllm_container.sh` | vLLM 容器内一键安装 |
| `scripts/patch_vllm_sm120_deep_gemm.py` | Patch vLLM 让 SM120/SM121 走我们的 MoE FP4 路径 |

### 桩文件（只为了 vLLM 的 `import deep_gemm` 不炸）

以下文件**运行时不参与实际计算**。vLLM 代码里有 `import deep_gemm` / `from vllm.third_party import deep_gemm`，如果这些符号不存在就直接 ImportError。这些桩让 import 链活着，实际计算走的是 vLLM 自己的 kernel（SM120 上本来就没问题）。

| 路径 | 说明 |
|------|------|
| `consumer_deep_gemm/attention.py` | MQA/MLA 注意力桩——vLLM 用自己的 attention kernel |
| `consumer_deep_gemm/hyperconnection.py` | HC pre-norm 桩——jasl 的 Triton 处理这个 |
| `consumer_deep_gemm/einsum.py` | FP8 einsum 桩——vLLM 自己的路径 |
| `consumer_deep_gemm/mega.py` | MegaMoE 权重变换桩 |
| `consumer_deep_gemm/utils.py` | SM 数量 / TC 利用率控制桩 |
| `deep_gemm/` | 顶层兼容包，`import deep_gemm` 转发到 `consumer_deep_gemm` |

## 怎么装到 vLLM 里

### 方法一：容器内一键安装

```bash
# 进入 vLLM 容器
docker exec -it <container_name> bash

# 安装
cd /path/to/Consumer-DeepGEMM
./scripts/install_in_vllm_container.sh
```

脚本做四件事：
1. 编译 CUTLASS native 扩展（sm_121a）
2. `pip install .` 安装 Python 包
3. 写入 `vllm.third_party.deep_gemm` 转发 shim
4. Patch vLLM 的 4 个文件，让 SM120/SM121 能走 DeepGEMM MoE 路径

### 方法二：烘进 Docker 镜像

在 Dockerfile 里加：

```dockerfile
COPY Consumer-DeepGEMM /opt/Consumer-DeepGEMM
RUN cd /opt/Consumer-DeepGEMM && ./scripts/install_in_vllm_container.sh
```

### 验证

```bash
python3 -c "
import consumer_deep_gemm as dg
import vllm.third_party.deep_gemm as vdg
print('consumer:', dg.native_build_info())
print('vllm:', vdg.native_build_info())
"
```

期望输出：
```
consumer: {'available': True, 'cutlass_sm120_probe': True, 'cutlass_mxfp8_mxfp4_probe': True, 'arch': 'sm_121a'}
vllm: {'available': True, 'cutlass_sm120_probe': True, 'cutlass_mxfp8_mxfp4_probe': True, 'arch': 'sm_121a'}
```

## 性能说明

当前速度 ~0.8 tok/s（每个 token 需要 120 次 CUTLASS kernel launch + CPU↔GPU 同步）。**这不是日常使用方案**——价值在于证明修复方向正确。日常使用推荐 [jasl 的 Triton fork](https://github.com/jasl/vllm/tree/ds4-sm120)（~14 tok/s）。

## 相关链接

- vLLM issue: [vllm#40928](https://github.com/vllm-project/vllm/issues/40928)
- 部署教程: [deepseek-v4-deployment-on-dgx-spark](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark)
