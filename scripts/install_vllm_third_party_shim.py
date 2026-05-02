#!/usr/bin/env python3
"""Install a vllm.third_party.deep_gemm shim into the active Python env."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SHIM = '''"""Consumer-DeepGEMM shim for vLLM's vendored DeepGEMM import path."""

from consumer_deep_gemm import *  # noqa: F401,F403
'''


def main() -> None:
    spec = importlib.util.find_spec("vllm.third_party")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm.third_party is not importable in this Python environment")

    third_party = Path(next(iter(spec.submodule_search_locations)))
    target = third_party / "deep_gemm.py"
    target.write_text(SHIM, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
