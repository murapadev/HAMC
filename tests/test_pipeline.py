"""End-to-end checks: a map reported as valid never contains invalid blocks."""

import unittest
from unittest import mock

from hamc.core.validator import PathValidator
from hamc.generators.local_generator import LocalGenerator
from hamc.pipeline import GenerationError, block_neighbors, generate_map


def _block(tilemap, br, bc, size):
    return [row[bc * size:(bc + 1) * size] for row in tilemap[br * size:(br + 1) * size]]


def _block_is_valid(hmap, br, bc):
    btype = hmap.blocks[br][bc]
    tiles = _block(hmap.tilemap, br, bc, hmap.local)
    neighbours = block_neighbors(hmap.blocks, br, bc)
    if btype in ("river", "road"):
        return PathValidator.validate_network(tiles, btype, neighbours)
    if btype == "oasis":
        return PathValidator.validate_oasis(tiles)
    return PathValidator.validate_building(tiles, btype)


class TestEveryBlockValidates(unittest.TestCase):
    def test_special_blocks_pass_their_rules(self):
        for size in (3, 4, 6, 8):
            for btype in ("river", "road", "oasis", "residential", "commercial", "industrial"):
                for neighbours in ({}, {"left": btype, "top": btype}):
                    gen = LocalGenerator(btype, size, neighbours)
                    self.assertTrue(gen.collapse(), f"{btype} size={size} {neighbours}")

    def test_commercial_block_has_shop_windows(self):
        gen = LocalGenerator("commercial", 4)
        self.assertTrue(gen.collapse())
        self.assertGreaterEqual(sum(row.count("Escaparate") for row in gen.tile_grid()), 2)


class TestPipeline(unittest.TestCase):
    def test_valid_maps_only_contain_valid_blocks(self):
        for seed in range(6):
            hmap = generate_map(4, 3, 2, 4, seed=seed)
            self.assertTrue(hmap.valid, hmap.issues)
            rows, cols = len(hmap.blocks), len(hmap.blocks[0])
            self.assertEqual(len(hmap.tilemap), rows * 4)
            for br in range(rows):
                for bc in range(cols):
                    if hmap.blocks[br][bc] in LocalGenerator.SPECIAL_BLOCK_TYPES:
                        self.assertTrue(_block_is_valid(hmap, br, bc),
                                        f"seed={seed} block=({br},{bc}) {hmap.blocks[br][bc]}")

    def test_invalid_block_is_reported_not_hidden(self):
        with mock.patch.object(LocalGenerator, "validate_paths", return_value=False), \
                mock.patch.object(LocalGenerator, "attempts", 2):
            with self.assertRaises(GenerationError):
                generate_map(2, 2, 2, 4, seed=1)
            hmap = generate_map(2, 2, 2, 4, seed=1, strict=False)
        self.assertFalse(hmap.valid)
        self.assertTrue(any("invalid after 2 attempts" in issue for issue in hmap.issues))


class TestTinyGrids(unittest.TestCase):
    def test_region_grid_too_small_for_every_region(self):
        with self.assertRaisesRegex(ValueError, "needs all 3 region types"):
            generate_map(2, 1, 2, 4, seed=1)

    def test_block_too_small_for_buildings(self):
        with self.assertRaisesRegex(ValueError, "local must be >= 3"):
            generate_map(3, 1, 1, 2, seed=1)

    def test_smallest_valid_map_has_every_region(self):
        for seed in range(5):
            hmap = generate_map(3, 1, 1, 3, seed=seed)
            self.assertEqual(sorted(r for row in hmap.regions for r in row), ["city", "desert", "forest"])


if __name__ == "__main__":
    unittest.main()
