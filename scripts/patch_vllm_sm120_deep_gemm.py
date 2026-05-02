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
