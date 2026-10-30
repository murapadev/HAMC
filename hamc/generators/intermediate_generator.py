from typing import Dict, Optional, Set

from ..config.block_config import BlockConfig
from ..core.cell import Cell
from .base_generator import BaseGenerator
from .global_generator import GlobalGenerator

_STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0))


class IntermediateGenerator(BaseGenerator):
    """Middle layer: splits every region cell into ``subgrid_size``² blocks.

    Blocks on the edge of a region facing a different region may also take
    the transition block between both (scrubland, periurban, village). The
    block adjacency table then decides where a transition is really needed:
    a city block can't touch a desert block, so the solver must put a village
    (or another compatible block) between them.
    """

    attempts = 5

    def __init__(self, global_gen: GlobalGenerator, subgrid_size: int = 3):
        self.subgrid_size = subgrid_size
        self.global_gen = global_gen
        super().__init__(global_gen.width * subgrid_size, global_gen.height * subgrid_size,
                         name="IntermediateGenerator")
        self.initialize()

    def compatibility(self) -> Dict[str, Set[str]]:
        return BlockConfig.COMPATIBILITY

    def region_at(self, r: int, c: int) -> Optional[str]:
        return self.global_gen.cells[r // self.subgrid_size][c // self.subgrid_size].collapsed_value

    def allowed_blocks(self, r: int, c: int) -> Dict[str, float]:
        """Block weights admissible at intermediate cell ``(r, c)``."""
        region = self.region_at(r, c)
        options = BlockConfig.get_blocks_for_region(region)
        gr, gc = r // self.subgrid_size, c // self.subgrid_size
        s = self.subgrid_size
        for dr, dc in _STEPS:
            # Only the cells on the matching edge of the subgrid border the
            # neighbouring region.
            on_edge = ((dc == 1 and c % s == s - 1) or (dc == -1 and c % s == 0) or
                       (dr == 1 and r % s == s - 1) or (dr == -1 and r % s == 0))
            ngr, ngc = gr + dr, gc + dc
            if not on_edge or not (0 <= ngr < self.global_gen.height and 0 <= ngc < self.global_gen.width):
                continue
            other = self.global_gen.cells[ngr][ngc].collapsed_value
            if other and other != region:
                block, weight = BlockConfig.get_transition(region, other)
                if block and weight > 0:
                    options[block] = max(options.get(block, 0.0), weight)
        return options

    def initialize(self) -> bool:
        for row in self.global_gen.cells:
            for cell in row:
                if not cell.collapsed_value:
                    self.logger.error("Global layer is not fully collapsed")
                    return False
        self.cells = [[Cell(self.allowed_blocks(r, c)) for c in range(self.width)]
                      for r in range(self.height)]
        return True

    def collapse(self) -> bool:
        for _ in range(self.attempts):
            if not self.initialize():
                return False
            if super().collapse():
                return True
        return False

    def validate(self) -> bool:
        table = self.compatibility()
        for r in range(self.height):
            for c in range(self.width):
                block = self.cells[r][c].collapsed_value
                if block is None or block not in self.allowed_blocks(r, c):
                    self.logger.debug("Invalid block %s at (%d,%d)", block, r, c)
                    return False
                for nr, nc in self.get_neighbors(r, c):
                    if self.cells[nr][nc].collapsed_value not in table[block]:
                        self.logger.debug("Incompatible blocks at (%d,%d)", r, c)
                        return False
        return True
