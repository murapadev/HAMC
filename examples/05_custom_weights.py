"""Reweight every layer: a dense, forest-heavy map with few roads."""

import os
from collections import Counter

from _common import out_dir

from hamc import generate_map, render_map

hmap = generate_map(
    6, 4, 3, 6, seed=99,
    region_weights={"forest": 0.7, "desert": 0.1, "city": 0.2},
    block_weights={"forest": {"grove": 0.9, "clearing": 0.1, "river": 0.0}},
    tile_weights={"grove": {"Árbol": 0.85, "Hierba": 0.05}},
)
tiles = Counter(t for row in hmap.tilemap for t in row)
total = sum(tiles.values())
print("top tiles:", ", ".join(f"{t} {n / total:.0%}" for t, n in tiles.most_common(5)))
path = os.path.join(out_dir(), "dense_forest.png")
render_map(hmap, style="topdown", tile_size=6).save(path)
print("->", path)
