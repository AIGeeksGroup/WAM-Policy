# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for WAM-Policy (arm_libero): subpackages are imported lazily, and only `modules` and
# `utils` of upstream LingBot-VA are vendored.
"""Vendored subset of LingBot-VA (https://github.com/Robbyant/lingbot-va, commit 7c6ffa9).

Subpackages are imported lazily: the CPU simulation workers must not load a transformer or
CUDA extensions merely to import a lightweight module.
"""

from importlib import import_module

__all__ = ["modules", "utils"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
