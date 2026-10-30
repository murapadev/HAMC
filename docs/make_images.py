"""Regenerate the README images in docs/images.

    pip install -e .
    python docs/make_images.py

The hero and the gallery are rendered here; the example images come from
running the scripts in examples/ with docs/images as their output folder.
The dashboard screenshot (dashboard.png) is a browser capture of
`hamc dashboard` after a run with seed 23, 6x4 regions, subgrid 3, local 6.
"""

import os
import subprocess
import sys

from PIL import Image

from hamc import generate_map, render_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")

# (file, generate_map args, render style, tile size)
IMAGES = [
    ("hero.png", dict(width=6, height=4, subgrid=3, local=6, seed=23), "iso", 10),
    ("gallery_topdown_42.png", dict(width=4, height=3, subgrid=2, local=8, seed=42), "topdown", 8),
    ("gallery_iso_4.png", dict(width=5, height=4, subgrid=3, local=6, seed=4), "iso", 7),
    ("gallery_topdown_31.png", dict(width=6, height=4, subgrid=3, local=5, seed=31), "topdown", 6),
]
EXAMPLES = ["03_large_map.py", "04_forced_districts.py", "05_custom_weights.py",
            "06_central_park.py", "08_animate.py"]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, args, style, tile in IMAGES:
        path = os.path.join(OUT, name)
        render_map(generate_map(**args), style=style, tile_size=tile).save(path, optimize=True)
        print(f"{name}: {os.path.getsize(path) // 1024} KiB")
    for script in EXAMPLES:
        subprocess.run([sys.executable, os.path.join(ROOT, "examples", script), OUT], check=True,
                       cwd=os.path.join(ROOT, "examples"))
    # The large map is mostly texture noise: a 256-colour palette makes it ~2.5x smaller.
    large = os.path.join(OUT, "large_map.png")
    Image.open(large).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(large, optimize=True)


if __name__ == "__main__":
    main()
