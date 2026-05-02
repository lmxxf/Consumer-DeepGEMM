from setuptools import setup, find_packages

setup(
    name="consumer-deep-gemm",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.11",
    ],
    author="lmxxf",
    description="DeepGEMM-compatible API for consumer Blackwell GPUs (SM120/SM121)",
    url="https://github.com/lmxxf/Consumer-DeepGEMM",
    license="Apache-2.0",
)
