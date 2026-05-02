#!/usr/bin/env python3
"""Patch installed vLLM to allow Consumer-DeepGEMM on SM120/SM121."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PATCHES = {
    "platforms/cuda.py": [
        (
            "return cls.is_device_capability(90) or cls.is_device_capability_family(100)",
            "return (\n"
            "            cls.is_device_capability(90)\n"
            "            or cls.is_device_capability_family(100)\n"
            "            or cls.is_device_capability_family(120)\n"
            "        )",
        ),
    ],
    "model_executor/layers/fused_moe/experts/deep_gemm_moe.py": [
        (
            "is_deep_gemm_supported()\n"
            "            and current_platform.is_device_capability_family(100)",
            "is_deep_gemm_supported()\n"
            "            and (\n"
            "                current_platform.is_device_capability_family(100)\n"
            "                or current_platform.is_device_capability_family(120)\n"
            "            )",
        ),
    ],
    "model_executor/kernels/linear/scaled_mm/deep_gemm.py": [
        (
            "        if not is_deep_gemm_supported():\n"
            "            return False, \"Currently, only Hopper and Blackwell GPUs are supported.\"\n"
            "        return True, None",
            "        if current_platform.is_device_capability_family(120):\n"
            "            return False, (\n"
            "                \"Consumer-DeepGEMM on SM120/SM121 is only enabled for \"\n"
            "                \"DeepSeek V4 MoE FP4; ordinary FP8 linear should use the \"\n"
            "                \"non-DeepGEMM vLLM kernels.\"\n"
            "            )\n"
            "        if not is_deep_gemm_supported():\n"
            "            return False, \"Currently, only Hopper and Blackwell GPUs are supported.\"\n"
            "        return True, None",
        ),
        (
            "        if config.out_dtype != torch.bfloat16:\n"
            "            return (False, \"Supports only output dtype of bfloat16\")",
            "        if current_platform.is_device_capability_family(120):\n"
            "            return False, (\n"
            "                \"Consumer-DeepGEMM on SM120/SM121 is only enabled for \"\n"
            "                \"DeepSeek V4 MoE FP4; ordinary FP8 linear should use the \"\n"
            "                \"non-DeepGEMM vLLM kernels.\"\n"
            "            )\n"
            "        if config.out_dtype != torch.bfloat16:\n"
            "            return (False, \"Supports only output dtype of bfloat16\")",
        ),
    ],
    "model_executor/warmup/deep_gemm_warmup.py": [
        (
            "from vllm.tracing import instrument\n"
            "from vllm.utils.deep_gemm import (",
            "from vllm.tracing import instrument\n"
            "from vllm.platforms import current_platform\n"
            "from vllm.utils.deep_gemm import (",
        ),
        (
            "    # FIXME: this logic is brittle and incorrect - since we\n"
            "    # could use DeepGEMM with for than just Fp8LinearMethod\n"
            "    block_size = get_mk_alignment_for_contiguous_layout()[0]",
            "    if current_platform.is_device_capability_family(120):\n"
            "        return False\n\n"
            "    # FIXME: this logic is brittle and incorrect - since we\n"
            "    # could use DeepGEMM with for than just Fp8LinearMethod\n"
            "    block_size = get_mk_alignment_for_contiguous_layout()[0]",
        ),
    ],
}


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm is not importable in this Python environment")
    vllm_root = Path(next(iter(spec.submodule_search_locations)))

    changed = []
    for rel_path, replacements in PATCHES.items():
        path = vllm_root / rel_path
        text = path.read_text(encoding="utf-8")
        original = text
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
            elif new in text:
                pass
            else:
                raise SystemExit(f"patch pattern not found in {path}: {old!r}")
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed.append(str(path))

    if changed:
        print("patched:")
        for path in changed:
            print(path)
    else:
        print("vLLM SM120 DeepGEMM patches already present")


if __name__ == "__main__":
    main()
