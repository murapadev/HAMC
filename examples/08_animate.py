"""Animate the collapse with the observer hook and save a GIF.

The observer receives every event of the three layers. We keep a copy of
each layer's state and snapshot it every few events: first the regions,
then the blocks, then the tiles block by block (finished blocks are shown
with flat tile colours, pending ones with their block colour) and finally
the painted map.
"""

import os

from _common import out_dir
from PIL import Image

from hamc import generate_map, render_map
from hamc.config.tile_config import TileConfig
from hamc.visualization.renderer import MapRenderer

W, H, SUB, LOCAL, SEED = 4, 3, 3, 6, 23
rows, cols = H * SUB * LOCAL, W * SUB * LOCAL
SCALE = 6  # pixels per tile in the GIF

regions = [[None] * W for _ in range(H)]
blocks = [[None] * (W * SUB) for _ in range(H * SUB)]
tiles = [[None] * cols for _ in range(rows)]
frames = []
counter = {"n": 0}


def color_of(r, c):
    if tiles[r][c]:
        return TileConfig.get_color(tiles[r][c])
    b = blocks[r // LOCAL][c // LOCAL]
    if b:
        return MapRenderer.BLOCK_COLORS.get(b, (0, 0, 0))
    g = regions[r // (LOCAL * SUB)][c // (LOCAL * SUB)]
    return MapRenderer.REGION_COLORS.get(g, (24, 30, 38)) if g else (24, 30, 38)


def snapshot():
    img = Image.new("RGB", (cols, rows))
    img.putdata([color_of(r, c) for r in range(rows) for c in range(cols)])
    frames.append(img.resize((cols * SCALE, rows * SCALE), Image.NEAREST))


def observer(event):
    kind, level = event["type"], event.get("level")
    if kind == "collapse" and level == "global":
        regions[event["pos"][0]][event["pos"][1]] = event["value"]
        snapshot()
    elif kind == "collapse" and level == "intermediate":
        blocks[event["pos"][0]][event["pos"][1]] = event["value"]
        counter["n"] += 1
        if counter["n"] % 6 == 0:
            snapshot()
    elif kind == "local_block_complete":
        r0, c0 = event["base"]
        for r, row in enumerate(event["tiles"]):
            tiles[r0 + r][c0:c0 + LOCAL] = row
        if (event["br"] * W * SUB + event["bc"]) % 4 == 3:
            snapshot()


hmap = generate_map(W, H, SUB, LOCAL, seed=SEED, observer=observer)
snapshot()
# End on the painted map (same pixel size: SCALE px per tile), held a while.
frames += [render_map(hmap, style="topdown", tile_size=SCALE)] * 25
path = os.path.join(out_dir(), "collapse.gif")
frames[0].save(path, save_all=True, append_images=frames[1:], duration=60, loop=0, optimize=True)
print(f"{len(frames)} frames, {os.path.getsize(path) / 1024:.0f} KiB -> {path}")
