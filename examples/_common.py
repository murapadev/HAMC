"""Shared helper: where the examples write their files."""

import os
import sys


def out_dir() -> str:
    """First CLI argument, or ./output/examples."""
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "examples")
    os.makedirs(path, exist_ok=True)
    return path
