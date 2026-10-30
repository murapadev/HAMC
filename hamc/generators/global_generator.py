import random
from typing import Dict, Optional, Set

from ..config.region_config import RegionConfig
from ..core.cell import Cell
from .base_generator import BaseGenerator


class GlobalGenerator(BaseGenerator):
    """Top layer: assigns a region (forest, desert, city) to each coarse cell.

    Every region type must appear at least once. Plain WFC is retried a few
    times; if a region is still missing, it is seeded at a random free cell
    and the rest of the grid is collapsed around it.
    """

    attempts = 5

    def __init__(self, width: int, height: int, rng: Optional[random.Random] = None):
        super().__init__(width, height, name="GlobalGenerator", rng=rng)

    def compatibility(self) -> Dict[str, Set[str]]:
        return {name: set(cfg["can_adjacent"]) for name, cfg in RegionConfig.TYPES.items()}

    def initialize(self) -> bool:
        probs = RegionConfig.get_probabilities()
        self.cells = [[Cell(dict(probs)) for _ in range(self.width)] for _ in range(self.height)]
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

        missing = sorted(set(RegionConfig.TYPES) - self._present())
        self.logger.warning("Seeding missing regions after %d attempts: %s", self.attempts, missing)
        self._reset(seeded)
        free = [(r, c) for r in range(self.height) for c in range(self.width) if (r, c) not in seeded]
        self.rng.shuffle(free)
        for region, (r, c) in zip(sorted(RegionConfig.TYPES), free):
            if region not in seeded.values():
                self.cells[r][c].possible = {region: 1.0}
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
