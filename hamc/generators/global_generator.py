import random
from typing import Callable, Dict, Mapping, Optional, Set

from ..config.region_config import RegionConfig
from ..config.weights import merge_weights
from ..core.cell import Cell
from .base_generator import BaseGenerator, CellRule


class GlobalGenerator(BaseGenerator):
    """Top layer: assigns a region (forest, desert, city) to each coarse cell.

    Every region type must appear at least once. Plain WFC is retried a few
    times; if a region is still missing, it is seeded at a random free cell
    and the rest of the grid is collapsed around it.
    """

    attempts = 5

    def __init__(self, width: int, height: int, rng: Optional[random.Random] = None,
                 weights: Optional[Mapping[str, float]] = None,
                 constraint: Optional[Callable[[int, int], CellRule]] = None):
        required = len(RegionConfig.TYPES)
        if width < 1 or height < 1 or width * height < required:
            raise ValueError(
                f"a {width}x{height} region grid has {max(0, width * height)} cell(s), but every map "
                f"needs all {required} region types ({', '.join(RegionConfig.TYPES)}); "
                f"use at least {required} cells, e.g. width={required} height=1")
        super().__init__(width, height, name="GlobalGenerator", rng=rng)
        self.weights = merge_weights(RegionConfig.get_probabilities(), weights, RegionConfig.TYPES, "region")
        self.constraint = constraint

    def compatibility(self) -> Dict[str, Set[str]]:
        return {name: set(cfg["can_adjacent"]) for name, cfg in RegionConfig.TYPES.items()}

    def initialize(self) -> bool:
        self.cells = [[Cell(dict(self.weights)) for _ in range(self.width)] for _ in range(self.height)]
        if self.constraint:
            for r in range(self.height):
                for c in range(self.width):
                    self._apply_rule(r, c, self.constraint(r, c))
        return True

    def collapse(self) -> bool:
        # Callers (and tests) may pre-seed cells; keep them across retries.
        seeded = {(r, c): cell.collapsed_value
                  for r, row in enumerate(self.cells) for c, cell in enumerate(row)
                  if cell.collapsed_value}

        for _ in range(self.attempts):
            self._reset(seeded)
            if super().collapse():
                return True

        # Plain WFC keeps missing a region: seed one cell of every region
        # that isn't pre-seeded, then collapse the rest around them.
        missing = sorted(set(RegionConfig.TYPES) - set(seeded.values()))
        self.logger.info("No complete map after %d attempts; seeding regions %s", self.attempts, missing)
        self._reset(seeded)
        free = [(r, c) for r in range(self.height) for c in range(self.width) if (r, c) not in seeded]
        self.rng.shuffle(free)
        for region in missing:
            spot = next(((r, c) for r, c in free if region in self.cells[r][c].possible), None)
            if spot is None:
                raise ValueError(f"weights/constraints leave no cell where region {region!r} can go")
            free.remove(spot)
            self._apply_rule(*spot, region)
        return super().collapse()

    def _reset(self, seeded: Dict) -> None:
        self.initialize()
        for (r, c), value in seeded.items():
            self.cells[r][c].possible = {value: 1.0}
            self.cells[r][c].collapsed_value = value

    def _present(self) -> Set[str]:
        return {cell.collapsed_value for row in self.cells for cell in row if cell.collapsed_value}

    def validate(self) -> bool:
        if any(cell.collapsed_value is None for row in self.cells for cell in row):
            self.logger.debug("Not all cells are collapsed")
            return False
        missing = set(RegionConfig.TYPES) - self._present()
        if missing:
            self.logger.debug("Missing required regions: %s", sorted(missing))
            return False
        table = self.compatibility()
        for r in range(self.height):
            for c in range(self.width):
                a = self.cells[r][c].collapsed_value
                for nr, nc in self.get_neighbors(r, c):
                    if self.cells[nr][nc].collapsed_value not in table[a]:
                        return False
        return True
