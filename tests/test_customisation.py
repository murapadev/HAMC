import unittest

from hamc import generate_map


class TestWeightsAndConstraints(unittest.TestCase):
    def test_region_weights_shift_the_mix(self):
        hmap = generate_map(6, 6, 1, 3, seed=2, region_weights={"forest": 20, "desert": 1, "city": 1})
        regions = [r for row in hmap.regions for r in row]
        self.assertGreater(regions.count("forest"), len(regions) // 2)
        self.assertEqual(set(regions), {"forest", "desert", "city"})

    def test_tile_weight_zero_removes_a_tile(self):
        hmap = generate_map(4, 3, 2, 4, seed=3, region_weights={"forest": 10},
                            tile_weights={"grove": {"Árbol": 0}})
        size = hmap.local
        for br, row in enumerate(hmap.blocks):
            for bc, block in enumerate(row):
                if block == "grove":
                    tiles = [t for r in hmap.tilemap[br * size:(br + 1) * size]
                             for t in r[bc * size:(bc + 1) * size]]
                    self.assertNotIn("Árbol", tiles)

    def test_forced_central_park(self):
        hmap = generate_map(3, 3, 3, 4, seed=4,
                            constrain_regions=lambda r, c: "city" if (r, c) == (1, 1) else None,
                            constrain_blocks=lambda r, c, region: "park" if (r, c) == (4, 4) else None)
        self.assertEqual(hmap.regions[1][1], "city")
        self.assertEqual(hmap.blocks[4][4], "park")
        self.assertTrue(hmap.valid)

    def test_impossible_constraint_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "leaves no option"):
            generate_map(3, 1, 1, 4, seed=1,
                         constrain_regions=lambda r, c: "forest" if c == 0 else None,
                         constrain_blocks=lambda r, c, region: "park" if c == 0 else None)

    def test_unknown_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown region"):
            generate_map(3, 1, 1, 3, region_weights={"jungle": 1})
        with self.assertRaisesRegex(ValueError, "unknown block"):
            generate_map(3, 1, 1, 3, tile_weights={"castle": {"Árbol": 1}})


if __name__ == "__main__":
    unittest.main()
