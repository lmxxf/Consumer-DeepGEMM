import os
from pathlib import Path

from setuptools import setup, find_packages


def get_ext_modules():
    if os.environ.get("CONSUMER_DEEP_GEMM_BUILD_CUDA") != "1":
        return []

    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    from torch.utils import cpp_extension

    if cpp_extension.CUDA_HOME is None and Path(os.environ["CUDA_HOME"]).exists():
        cpp_extension.CUDA_HOME = os.environ["CUDA_HOME"]

    CUDAExtension = cpp_extension.CUDAExtension

    root = Path(__file__).resolve().parent
    cutlass_path = Path(
        os.environ.get(
            "CUTLASS_PATH",
            root.parent / "DeepGEMM" / "third-party" / "cutlass",
        )
    )
    if not cutlass_path.exists():
        raise RuntimeError(
            "CUTLASS headers not found. Set CUTLASS_PATH or clone DeepGEMM "
            "with third-party/cutlass available."
        )

    arch = os.environ.get("CONSUMER_DEEP_GEMM_CUDA_ARCH", "120a")
    return [
        CUDAExtension(
            name="consumer_deep_gemm._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/cutlass_sm120_probe.cu",
                "csrc/cutlass_mxfp8_mxfp4_probe.cu",
            ],
            include_dirs=[
                str(cutlass_path / "include"),
                str(cutlass_path / "tools" / "util" / "include"),
                str(cutlass_path / "examples" / "common"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    f"-arch=sm_{arch}",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                ],
            },
        )
    ]


def get_cmdclass():
    if os.environ.get("CONSUMER_DEEP_GEMM_BUILD_CUDA") != "1":
        return {}

    from torch.utils.cpp_extension import BuildExtension

    return {"build_ext": BuildExtension}

setup(
    name="consumer-deep-gemm",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.11",
    ],
    ext_modules=get_ext_modules(),
    cmdclass=get_cmdclass(),
    author="lmxxf",
    description="DeepGEMM-compatible API for consumer Blackwell GPUs (SM120/SM121)",
    url="https://github.com/lmxxf/Consumer-DeepGEMM",
    license="Apache-2.0",
)
