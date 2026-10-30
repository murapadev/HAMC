"""Force a commercial downtown in the centre and industry on the outskirts.

Constraints work per layer: the region constraint makes the whole map a
city, and the block constraint picks the district by distance to the
centre. Everything else (roads, parks, housing in between) is left to WFC.
"""

import os

from _common import out_dir

from hamc import generate_map, render_map

W, H, SUB = 5, 4, 3
rows, cols = H * SUB, W * SUB
cy, cx = (rows - 1) / 2, (cols - 1) / 2


def district(r, c, region):
    d = max(abs(r - cy) / cy, abs(c - cx) / cx)  # 0 at the centre, 1 at the edge
    if d < 0.3:
        return {"commercial": 0.85, "road": 0.15}
    if d > 0.75:
        return {"industrial": 0.7, "road": 0.15, "park": 0.15}
    return {"residential": 0.6, "park": 0.15, "road": 0.15, "commercial": 0.1}


hmap = generate_map(W, H, SUB, 6, seed=7,
                    region_weights={"city": 1.0},
                    constrain_regions=lambda r, c: "city" if 0 < r < H - 1 and 0 < c < W - 1 else None,
                    constrain_blocks=lambda r, c, region: district(r, c, region) if region == "city" else None)

for row in hmap.blocks:
    print(" ".join(b[:3] for b in row))
path = os.path.join(out_dir(), "districts.png")
render_map(hmap, style="iso", tile_size=8).save(path)
print("->", path)
