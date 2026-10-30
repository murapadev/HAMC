"""Export a map as JSON and as images in every style."""

import json
import os

from _common import out_dir

from hamc import generate_map, render_map

out = out_dir()
hmap = generate_map(4, 3, 2, 6, seed=42)

json_path = os.path.join(out, "map.json")
hmap.save_json(json_path)
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
print("JSON keys:", ", ".join(data))
print(f"tilemap[0][:6] = {data['tilemap'][0][:6]}")

for style in ("flat", "topdown", "iso"):
    path = os.path.join(out, f"map_{style}.png")
    img = render_map(hmap, style=style, tile_size=10)
    img.save(path)
    print(f"{style:8} {img.width}x{img.height}px -> {path}")
