"""Guarded imports for the optional Layer 3 seams.

Nothing in the offline core may import a third party. The optional layers may,
but only through this function -- so a missing, broken, or import-time-raising
dependency degrades to `None` instead of taking the run down.

`__init__` is not wrapped by the evaluator (local_evaluator.py:306), so an
ImportError raised at Agent construction kills all 200 sessions rather than one.
That is the failure this module exists to prevent.
"""
from __future__ import annotations

import importlib
from types import ModuleType


def try_import(module_name: str) -> ModuleType | None:
    """Import `module_name`, returning None on ANY failure.

    Deliberately catches BaseException-adjacent breadth via Exception: a package
    that raises at import time (a CUDA probe, a missing shared library, a network
    call in its __init__) is exactly the case a narrow `except ImportError` misses.
    """
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def available(module_name: str) -> bool:
    """True when `module_name` can be imported. Never raises."""
    return try_import(module_name) is not None
