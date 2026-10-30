"""Regression tests for the audit fixes (see docs/AUDITORIA.md).

Covers the correctness bugs and the dashboard hardening that previously had no
coverage: commercial-block validation typo, compatibility-graph symmetry, the
``collapse()`` re-initialization bug, weighted-choice error reporting, seed
reproducibility and the SSE cost/limit guards.
"""

import random
import unittest

from hamc import dashboard
from hamc.config.block_config import BlockConfig
from hamc.config.tile_config import TileConfig
from hamc.core.entropy import weighted_random_choice
from hamc.core.validator import PathValidator
from hamc.generators.global_generator import GlobalGenerator


class TestCommercialValidation(unittest.TestCase):
    """A-2: ``validate_building`` used the Spanish 'comercial' and never ran."""

    def _grid(self, escaparates=0):
        g = [["Pared" for _ in range(4)] for _ in range(4)]
        placed = 0
        for r in range(4):
            for c in range(4):
                if placed < escaparates:
                    g[r][c] = "Escaparate"
                    placed += 1
        return g

    def test_commercial_without_displays_fails(self):
        self.assertFalse(
            PathValidator.validate_building(self._grid(escaparates=0), "commercial")
        )

    def test_commercial_with_displays_passes(self):
        self.assertTrue(
            PathValidator.validate_building(self._grid(escaparates=2), "commercial")
        )


class TestGraphSymmetry(unittest.TestCase):
    """M-2: are_compatible requires both directions; graphs must be symmetric."""

    def _assert_symmetric(self, graph):
        for node, neighbours in graph.items():
            for other in neighbours:
                self.assertIn(
                    node, graph.get(other, set()),
                    f"asymmetric edge: {node} -> {other} but not back",
                )

    def test_block_graph_symmetric(self):
        self._assert_symmetric(BlockConfig.COMPATIBILITY)

    def test_tile_graph_symmetric(self):
        self._assert_symmetric(TileConfig.COMPATIBILITY)

    def test_oasis_self_adjacent(self):
        self.assertTrue(BlockConfig.are_compatible("oasis", "oasis"))

    def test_village_borders_city(self):
        self.assertTrue(BlockConfig.are_compatible("village", "residential"))


class TestCollapseDoesNotReinitialize(unittest.TestCase):
    """A-1 / M-5: collapse() must not wipe pre-seeded cells."""

    def test_preseeded_cell_survives_collapse(self):
        g = GlobalGenerator(3, 3)
        g.initialize()
        g.cells[0][0].possible = {"desert": 1.0}
        g.cells[0][0].collapse()
        self.assertEqual(g.cells[0][0].collapsed_value, "desert")

        self.assertTrue(g.collapse())
        # If collapse re-initialized the grid, this seed would be gone.
        self.assertEqual(g.cells[0][0].collapsed_value, "desert")

    def test_all_required_regions_present(self):
        # A-1: forced-region placement must actually take effect.
        for seed in range(5):
            g = GlobalGenerator(3, 3, rng=random.Random(seed))
            g.initialize()
            self.assertTrue(g.collapse())
            self.assertTrue(g.validate())  # validate() requires every region type


class TestWeightedChoiceErrors(unittest.TestCase):
    """M-6: weighted_random_choice signals failure instead of hiding it."""

    def test_empty_options(self):
        self.assertEqual(weighted_random_choice({}), "ERROR")

    def test_zero_sum_weights(self):
        self.assertEqual(weighted_random_choice({"a": 0.0, "b": 0.0}), "ERROR")

    def test_logs_on_failure(self):
        with self.assertLogs("hamc.core.entropy", level="WARNING"):
            # A non-numeric weight raises inside the choice and must be logged
            # (M-6: the except branch previously swallowed the error silently).
            self.assertEqual(weighted_random_choice({"a": "not-a-number"}), "ERROR")


class _CapturingBroker:
    """Stand-in for EventBroker that records emitted events synchronously."""

    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)

    def cancelled(self):
        return False

    def close(self):
        pass

    def types(self):
        return [e.get("type") for e in self.events]


class TestDashboardLimits(unittest.TestCase):
    """C-1: parameter clamping and the cost budget on /api/stream."""

    def test_clip_clamps_to_limits(self):
        self.assertEqual(dashboard._clip("width", 999, 4), 8)
        self.assertEqual(dashboard._clip("width", -5, 4), 1)
        self.assertEqual(dashboard._clip("width", "not-a-number", 4), 4)

    def test_oversized_request_is_rejected(self):
        broker = _CapturingBroker()
        # Max clamped params still exceed MAX_TOTAL_CELLS -> must refuse.
        dashboard._run_generation(
            {"width": 8, "height": 8, "subgrid": 4, "local": 8, "delay": 0}, broker
        )
        self.assertIn("run_error", broker.types())
        self.assertNotIn("run_complete", broker.types())

    def test_small_request_completes(self):
        broker = _CapturingBroker()
        dashboard._run_generation(
            {"width": 2, "height": 2, "subgrid": 1, "local": 3, "delay": 0, "seed": 7}, broker
        )
        self.assertIn("run_complete", broker.types())


if __name__ == "__main__":
    unittest.main()
