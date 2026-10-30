"""Time the solver on growing grids.

    pip install -e .
    python benchmarks/bench.py            # full table
    python benchmarks/bench.py --quick    # skip the largest sizes

Two measurements:
- ``single layer``: one LocalGenerator over an N x N grid of the grove
  palette, i.e. the raw WFC loop (selection, propagation, backtracking).
- ``full map``: generate_map() end to end; the intermediate layer is the
  biggest single grid (``width*subgrid`` x ``height*subgrid`` blocks).

Each row is the median of ``--repeat`` runs with seeds 0, 1, 2, ...
"""

import argparse
import logging
import platform
import statistics
import sys
import time

from hamc.generators.local_generator import LocalGenerator
from hamc.pipeline import generate_map


def _median_time(fn, repeat):
    samples = []
    for seed in range(repeat):
        t0 = time.perf_counter()
        fn(seed)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _single_layer(n):
    import random

    def run(seed):
        gen = LocalGenerator("grove", n, rng=random.Random(seed))
        assert gen.collapse()
    return run


def _full_map(width, height, subgrid, local):
    def run(seed):
        generate_map(width, height, subgrid, local, seed=seed)
    return run


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--quick", action="store_true")
    ns = p.parse_args(argv)
    logging.disable(logging.CRITICAL)

    print(f"# {platform.processor() or platform.machine()} | Python {sys.version.split()[0]}")
    print("| case | cells | time (s) |")
    print("|---|---:|---:|")
    singles = [16, 32, 48] if ns.quick else [16, 32, 48, 64]
    for n in singles:
        t = _median_time(_single_layer(n), ns.repeat)
        print(f"| single layer {n}x{n} | {n * n} | {t:.3f} |", flush=True)
    maps = [(4, 3, 2, 4), (8, 6, 3, 6), (8, 8, 4, 4)]
    if not ns.quick:
        maps.append((16, 12, 4, 6))
    for w, h, s, loc in maps:
        cells = w * h + (w * s) * (h * s) + (w * s * loc) * (h * s * loc)
        t = _median_time(_full_map(w, h, s, loc), ns.repeat)
        print(f"| full map {w}x{h}, subgrid {s}, local {loc} | {cells} | {t:.3f} |", flush=True)


if __name__ == "__main__":
    main()
