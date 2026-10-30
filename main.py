#!/usr/bin/env python3
"""HAMC demo runner.

Generates a map and writes the region, block and tile layers as PNG and JSON.

Usage:
  python main.py --width 4 --height 3 --subgrid 2 --local 4 --output output --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

from hamc.pipeline import GenerationError, generate_map
from hamc.visualization.renderer import MapRenderer


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a HAMC demo level")
    parser.add_argument("--width", type=int, default=4, help="Global width (cells)")
    parser.add_argument("--height", type=int, default=3, help="Global height (cells)")
    parser.add_argument("--subgrid", type=int, default=2, help="Intermediate subgrid size per global cell")
    parser.add_argument("--local", type=int, default=4, help="Local block tile size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation")
    parser.add_argument("--output", type=str, default="output", help="Output directory for images/data")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def generate_demo(width: int, height: int, subgrid: int, local: int, out_dir: str,
                  seed: Optional[int] = None) -> Dict[str, str]:
    hmap = generate_map(width, height, subgrid, local, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    renderer = MapRenderer(tile_size=20, padding=1)
    images = {
        "global_map": renderer.render_global_map(hmap.regions),
        "intermediate_map": renderer.render_intermediate_map(hmap.blocks),
        "final_tilemap": renderer.render_final_map(
            hmap.tilemap, global_size=(height, width),
            intermediate_size=(height * subgrid, width * subgrid)),
        "final_tilemap_clean": renderer.render_complete_tilemap(hmap.tilemap),
    }
    paths: Dict[str, str] = {}
    for name, img in images.items():
        paths[name] = os.path.join(out_dir, f"{name}.png")
        img.save(paths[name])
    for name, grid in (("regions", hmap.regions), ("blocks", hmap.blocks), ("tilemap", hmap.tilemap)):
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(grid, f, indent=2, ensure_ascii=False)
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.DEBUG if ns.debug else logging.WARNING,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    try:
        paths = generate_demo(ns.width, ns.height, ns.subgrid, ns.local, ns.output, ns.seed)
    except GenerationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("\nHAMC demo generated:")
    for k, v in paths.items():
        print(f" - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
