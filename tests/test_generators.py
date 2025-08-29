import unittest
from typing import Dict, List
from hamc.generators.local_generator import LocalGenerator
from hamc.generators.global_generator import GlobalGenerator
from hamc.generators.intermediate_generator import IntermediateGenerator
from hamc.core.cell import Cell
from hamc.config.tile_config import TileConfig
from hamc.config.region_config import RegionConfig
from hamc.config.block_config import BlockConfig

class TestCell(unittest.TestCase):
    def setUp(self):
        self.test_values = {"A": 0.5, "B": 0.3, "C": 0.2}
        self.cell = Cell(self.test_values)

    def test_initialization(self):
        self.assertEqual(self.cell.possible, self.test_values)
        self.assertIsNone(self.cell.collapsed_value)

    def test_entropy(self):
        entropy = self.cell.entropy()
        self.assertGreater(entropy, 0)
        
        self.cell.collapse()
        self.assertEqual(self.cell.entropy(), -1)

    def test_collapse(self):
        self.assertTrue(self.cell.collapse())
        self.assertIn(self.cell.collapsed_value, self.test_values.keys())
        self.assertEqual(len(self.cell.possible), 1)

    def test_invalid_initialization(self):
        with self.assertRaises(ValueError):
            Cell({})
        with self.assertRaises(ValueError):
            Cell({"A": -1})

class TestLocalGenerator(unittest.TestCase):
    def setUp(self):
        self.size = 4
        self.generator = LocalGenerator("river", self.size)

    def test_initialization(self):
        self.assertEqual(len(self.generator.cells), self.size)
        self.assertEqual(len(self.generator.cells[0]), self.size)
        self.assertEqual(self.generator.block_type, "river")

    def test_river_path_validation(self):
        # Set up a valid river path
        center = self.size // 2
        for row in range(self.size):
            self.generator.cells[row][center].possible = {"Agua": 1.0}
            self.generator.cells[row][center].collapse()
        
        self.assertTrue(self.generator.validate_paths())

    def test_compatibility_propagation(self):
        center = self.size // 2
        self.generator.cells[0][center].possible = {"Agua": 1.0}
        self.generator.cells[0][center].collapse()
        
        success = self.generator.propagate(0, center)
        self.assertTrue(success)
        
        # Check neighbors have water as possibility
        neighbors = self.generator.get_neighbors(0, center)
        for r, c in neighbors:
            neighbor = self.generator.cells[r][c]
            self.assertIn("Agua", neighbor.possible)

    def test_invalid_path(self):
        # Create discontinuous water path in center column (should fail for river)
        center = self.size // 2
        self.generator.cells[0][center].possible = {"Agua": 1.0}
        self.generator.cells[0][center].collapse()
        self.generator.cells[2][center].possible = {"Agua": 1.0}
        self.generator.cells[2][center].collapse()
        
        self.assertFalse(self.generator.validate_paths())

class TestGlobalGenerator(unittest.TestCase):
    def setUp(self):
        self.width = 3
        self.height = 3
        self.generator = GlobalGenerator(self.width, self.height)

    def test_initialization(self):
        self.generator.initialize()
        self.assertEqual(len(self.generator.cells), self.height)
        self.assertEqual(len(self.generator.cells[0]), self.width)
        
        # Check initial probabilities
        probs = RegionConfig.get_probabilities()
        for row in self.generator.cells:
            for cell in row:
                self.assertEqual(cell.possible, probs)

    def test_collapse_and_validate(self):
        self.generator.initialize()
        success = self.generator.collapse()
        self.assertTrue(success)
        self.assertTrue(self.generator.validate())

    def test_region_compatibility(self):
        self.generator.initialize()
        self.generator.cells[0][0].possible = {"forest": 1.0}
        self.generator.cells[0][0].collapse()
        
        success = self.generator.propagate(0, 0)
        self.assertTrue(success)
        
        # Check neighbors have compatible regions
        for r, c in self.generator.get_neighbors(0, 0):
            cell = self.generator.cells[r][c]
            for region in cell.possible:
                self.assertTrue(RegionConfig.are_compatible("forest", region))

class TestIntermediateGenerator(unittest.TestCase):
    def setUp(self):
        self.global_gen = GlobalGenerator(2, 2)
        self.assertTrue(self.global_gen.initialize())
        
        # Set specific values for deterministic testing
        self.global_gen.cells[0][0].possible = {"forest": 1.0}
        self.global_gen.cells[0][1].possible = {"desert": 1.0}
        self.global_gen.cells[1][0].possible = {"city": 1.0}
        self.global_gen.cells[1][1].possible = {"forest": 1.0}
        
        # Collapse all cells
        for r in range(2):
            for c in range(2):
                self.assertTrue(self.global_gen.cells[r][c].collapse())
                
        self.generator = IntermediateGenerator(self.global_gen, subgrid_size=2)

    def test_initialization(self):
        self.assertEqual(len(self.generator.cells), 4)  # 2x2 with subgrid_size=2
        self.assertEqual(len(self.generator.cells[0]), 4)

    def test_transition_blocks(self):
        # Find a transition zone
        transitions_found = False
        for row in self.generator.cells:
            for cell in row:
                if "scrubland" in cell.possible or "periurban" in cell.possible:
                    transitions_found = True
                    break
        self.assertTrue(transitions_found)

    def test_collapse_and_validate(self):
        success = self.generator.collapse()
        self.assertTrue(success)
        self.assertTrue(self.generator.validate())

def create_test_suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCell))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLocalGenerator))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGlobalGenerator))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntermediateGenerator))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(create_test_suite())
