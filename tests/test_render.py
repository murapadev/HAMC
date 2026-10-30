import contextlib
import io
import os
import tempfile
import unittest

from PIL import Image

from hamc import generate_map, render_map
from hamc.cli import main
from hamc.visualization.painter import build_scene


class TestRenderer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hmap = generate_map(4, 3, 2, 6, seed=11)

    def test_topdown_covers_the_tilemap(self):
        img = render_map(self.hmap, "topdown", tile_size=8)
        self.assertEqual(img.size, (len(self.hmap.tilemap[0]) * 8, len(self.hmap.tilemap) * 8))

    def test_rendering_is_deterministic(self):
        for style in ("topdown", "iso"):
            a = render_map(self.hmap, style, tile_size=8).tobytes()
            self.assertEqual(a, render_map(self.hmap, style, tile_size=8).tobytes(), style)

    def test_city_lots_become_buildings(self):
        lots = sum(row.count(b) for row in self.hmap.blocks for b in ("residential", "commercial", "industrial"))
        scene = build_scene(self.hmap.blocks, self.hmap.tilemap, self.hmap.local, 8, self.hmap.seed)
        self.assertGreaterEqual(len([b for b in scene.buildings if b.kind in ("house", "shop", "factory")]), lots)

    def test_unknown_style(self):
        with self.assertRaises(ValueError):
            render_map(self.hmap, "pixel-art")

    def test_cli_render_size(self):
        with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
            path = os.path.join(out, "map.png")
            self.assertEqual(main(["render", "--seed", "3", "--size", "48x16", "--tile-size", "4",
                                   "-o", path]), 0)
            with Image.open(path) as img:
                self.assertEqual(img.size, (48 * 4, 16 * 4))


if __name__ == "__main__":
    unittest.main()
