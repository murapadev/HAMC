"""The smallest useful program: generate a map and save it as an image."""

from hamc import generate_map, render_map

hmap = generate_map(width=4, height=3, subgrid=2, local=6, seed=42)
render_map(hmap, style="topdown").save("quickstart.png")
print(f"{len(hmap.tilemap[0])}x{len(hmap.tilemap)} tiles, valid={hmap.valid} -> quickstart.png")
