"""A custom constraint: a city with a 3x3-block central park.

The middle region is forced to be a city and the 3x3 blocks at its centre
to be parks. The city needs room around the park for the transition
blocks towards its neighbours, hence subgrid=5.
"""

import os

from _common import out_dir

from hamc import generate_map, render_map

W, H, SUB = 3, 3, 5
centre = (H * SUB) // 2, (W * SUB) // 2

hmap = generate_map(
    W, H, SUB, 5, seed=5,
    constrain_regions=lambda r, c: "city" if (r, c) == (1, 1) else None,
    constrain_blocks=lambda r, c, region: (
        "park" if abs(r - centre[0]) <= 1 and abs(c - centre[1]) <= 1 else None),
)
print("blocks of the central city region:")
for row in hmap.blocks[SUB:2 * SUB]:
    print(" ".join(f"{b:11}" for b in row[SUB:2 * SUB]))
path = os.path.join(out_dir(), "central_park.png")
render_map(hmap, style="iso", tile_size=10).save(path)
print("->", path)
