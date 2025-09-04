#!/usr/bin/env python3
"""HAMC Demo Runner

Generates a demo level showing how to use HAMC from the CLI.

It builds:
- Global regions map
- Intermediate blocks map
- Final stitched tilemap (with overlays)

Usage:
  python main.py --width 4 --height 3 --subgrid 2 --local 4 --output output --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List

import numpy as np

from hamc.generators.global_generator import GlobalGenerator
from hamc.generators.intermediate_generator import IntermediateGenerator
from hamc.generators.local_generator import LocalGenerator
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


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def block_neighbors(I: IntermediateGenerator, r: int, c: int) -> Dict[str, str]:
    int_h, int_w = len(I.cells), len(I.cells[0])
    n: Dict[str, str] = {}
    if r > 0:
        n["top"] = I.cells[r - 1][c].collapsed_value
    if r < int_h - 1:
        n["bottom"] = I.cells[r + 1][c].collapsed_value
    if c > 0:
        n["left"] = I.cells[r][c - 1].collapsed_value
    if c < int_w - 1:
        n["right"] = I.cells[r][c + 1].collapsed_value
    return n


def generate_demo(width: int, height: int, subgrid: int, local: int, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    renderer = MapRenderer(tile_size=20, padding=1)
    paths: Dict[str, str] = {}

    # Global layer
    G = GlobalGenerator(width, height)
    G.initialize()
    success_g = G.collapse()
    regions = [[cell.collapsed_value for cell in row] for row in G.cells]
    img_global = renderer.render_global_map(regions)
    paths["global_map"] = os.path.join(out_dir, "global_map.png")
    img_global.save(paths["global_map"])
    if not success_g:
        logging.getLogger(__name__).warning("Global validation failed; image rendered for visualization anyway")

    # Intermediate layer
    I = IntermediateGenerator(G, subgrid_size=subgrid)
    success_i = I.collapse()
    blocks = [[cell.collapsed_value for cell in row] for row in I.cells]
    img_intermediate = renderer.render_intermediate_map(blocks)
    paths["intermediate_map"] = os.path.join(out_dir, "intermediate_map.png")
    img_intermediate.save(paths["intermediate_map"])
    if not success_i:
        logging.getLogger(__name__).warning("Intermediate validation failed; image rendered for visualization anyway")

    # Local layer stitched tilemap
    int_h, int_w = len(I.cells), len(I.cells[0])
    final_h, final_w = int_h * local, int_w * local
    tilemap: List[List[str]] = [[None for _ in range(final_w)] for _ in range(final_h)]  # type: ignore[list-item]

    for br in range(int_h):
        for bc in range(int_w):
            btype = I.cells[br][bc].collapsed_value
            LG = LocalGenerator(btype, local, block_neighbors(I, br, bc))
            LG.collapse()
            local_tiles = [[cell.collapsed_value for cell in row] for row in LG.cells]
            base_r, base_c = br * local, bc * local
            for r in range(local):
                for c in range(local):
                    tilemap[base_r + r][base_c + c] = local_tiles[r][c]

    # Render overlays and clean tilemap
    img_final = renderer.render_final_map(tilemap, global_size=(height, width), intermediate_size=(height * subgrid, width * subgrid))
    paths["final_tilemap"] = os.path.join(out_dir, "final_tilemap.png")
    img_final.save(paths["final_tilemap"])

    img_clean = renderer.render_complete_tilemap(tilemap)
    paths["final_tilemap_clean"] = os.path.join(out_dir, "final_tilemap_clean.png")
    img_clean.save(paths["final_tilemap_clean"])

    # Save simple JSON dumps for reference
    with open(os.path.join(out_dir, "regions.json"), "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "blocks.json"), "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "tilemap.json"), "w", encoding="utf-8") as f:
        json.dump(tilemap, f, indent=2, ensure_ascii=False)

    return paths


def main(argv: List[str] | None = None) -> int:
    ns = parse_args(argv or sys.argv[1:])
    setup_logging(ns.debug)

    # Seed for reproducibility
    if ns.seed is not None:
        random.seed(ns.seed)
        np.random.seed(ns.seed)

    paths = generate_demo(ns.width, ns.height, ns.subgrid, ns.local, ns.output)

    print("\nHAMC demo generated:")
    for k, v in paths.items():
        print(f" - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
