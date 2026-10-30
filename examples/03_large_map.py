"""A large map: 16x10 regions, 4x4 blocks per region, 6x6 tiles per block."""

import os

from _common import out_dir

from hamc import generate_map, render_map

hmap = generate_map(width=16, height=10, subgrid=4, local=6, seed=2024)
rows, cols = len(hmap.tilemap), len(hmap.tilemap[0])
print(f"{cols}x{rows} = {rows * cols} tiles, {len(hmap.blocks[0])}x{len(hmap.blocks)} blocks")
print("timings:", {k: round(v, 2) for k, v in hmap.timings.items()}, "s")
print("local attempts per block:", round(hmap.block_attempts / (len(hmap.blocks) * len(hmap.blocks[0])), 2))
path = os.path.join(out_dir(), "large_map.png")
render_map(hmap, style="topdown", tile_size=4).save(path)
print("->", path)
