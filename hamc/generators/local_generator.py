import random
from typing import Dict, Optional, Set

from ..config.advanced_config import get_generator_config
from ..config.tile_config import TileConfig
from ..core.cell import Cell
from ..core.validator import PathValidator, edge_anchor, network_connections
from .base_generator import BaseGenerator

BUILDING_TYPES = ("residential", "commercial", "industrial")


class LocalGenerator(BaseGenerator):
    """Bottom layer: the tiles inside one block.

    Most blocks start from their tile palette. Rivers, roads, oases and
    buildings start from a layout (a water channel, a street cross, a pond,
    a facade) whose cells carry their own weights; WFC fills in the rest and
    the block is checked by :class:`PathValidator`. A block that fails the
    check is regenerated, up to ``attempts`` times.
    """

    SPECIAL_BLOCK_TYPES = {"river", "road", "oasis", *BUILDING_TYPES}
    attempts = 25

    def __init__(self, block_type: str, size: int, neighbors: Optional[Dict[str, str]] = None,
                 rng: Optional[random.Random] = None):
        super().__init__(size, size, name=f"LocalGenerator_{block_type}", rng=rng)
        self.block_type = block_type
        self.neighbors = neighbors or {}
        self.validator = PathValidator()
        self.attempts_used = 0
        self._allowed: Set[str] = set()
        self.initialize()

    def compatibility(self) -> Dict[str, Set[str]]:
        return TileConfig.COMPATIBILITY

    # -- layouts -------------------------------------------------------------

    def initialize(self) -> bool:
        tiles = TileConfig.get_tiles_for_block(self.block_type)
        if not tiles:
            self.logger.error("No tiles defined for block type %s", self.block_type)
            return False
        self.cells = [[Cell(dict(tiles)) for _ in range(self.width)] for _ in range(self.height)]

        if self.block_type in ("river", "road"):
            self._init_network()
        elif self.block_type == "oasis":
            self._init_oasis()
        elif self.block_type in BUILDING_TYPES:
            self._init_building()
        else:
            # Matching the neighbouring blocks' palettes can over-constrain a
            # small block; the second half of the attempts go without it.
            if self.attempts_used <= self.attempts // 2:
                self._apply_border_constraints()
            self.cells[self.height // 2][self.width // 2].collapse(rng=self.rng)

        self._allowed = {t for row in self.cells for cell in row for t in cell.possible}
        return True

    def _set(self, r: int, c: int, weights: Dict[str, float]) -> None:
        self.cells[r][c].possible = dict(weights)

    def _init_network(self) -> None:
        """Lay a river channel or a street along the connected sides."""
        H, W = self.height, self.width
        cr, cc = H // 2, W // 2
        sides = network_connections(self.block_type, self.neighbors)
        channel = {(cr, cc)}
        for side in sides:
            er, ec = edge_anchor(side, H, W)
            for r in range(min(er, cr), max(er, cr) + 1):
                for c in range(min(ec, cc), max(ec, cc) + 1):
                    channel.add((r, c))

        river = self.block_type == "river"
        crossing = len(sides) > 2 or ({"top", "bottom"} & sides and {"left", "right"} & sides)
        for r in range(H):
            for c in range(W):
                if (r, c) in channel:
                    if river:
                        self._set(r, c, {"Agua": 0.7, "Aguas_profundas": 0.3})
                    elif crossing and (r, c) == (cr, cc):
                        self._set(r, c, {"Asfalto": 0.7, "Paso_cebra": 0.3})
                    else:
                        self._set(r, c, {"Asfalto": 0.65, "Linea": 0.35})
                elif river:
                    self._set(r, c, {"Hierba": 0.45, "Junco": 0.3, "Agua": 0.15, "Lirio": 0.1})
                elif r in (0, H - 1) or c in (0, W - 1):
                    self._set(r, c, {"Acera": 0.55, "Bordillo": 0.3, "Farola": 0.15})
                else:
                    self._set(r, c, {"Acera": 0.6, "Cesped_urbano": 0.4})

    def _init_oasis(self) -> None:
        """Pond in the middle, sand towards the edges."""
        config = get_generator_config()
        H, W = self.height, self.width
        center_r, center_c = H // 2, W // 2
        max_dist = ((H / 2) ** 2 + (W / 2) ** 2) ** 0.5
        for r in range(H):
            for c in range(W):
                if r in (0, H - 1) or c in (0, W - 1):
                    self._set(r, c, {"Arena": config.local_desert_sand_probability,
                                     "Hierba": config.local_desert_grass_probability})
                    continue
                dist = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5 / max_dist
                water = max(config.local_min_water_probability,
                            config.local_water_probability_decay * (1 - dist ** 2))
                sand = 0.7 * (1 - water)
                self._set(r, c, {"Agua": water, "Arena": sand, "Hierba": 1.0 - water - sand})
        self._set(center_r, center_c, {"Agua": 1.0})

    def _init_building(self) -> None:
        """A facade seen from the street.

        Row 0 is the roof, the last row the ground floor with a door in the
        middle, the outer columns are walls and the rest is facade. Commercial
        ground floors are shop windows; industrial walls are metal/concrete.
        """
        H, W, kind = self.height, self.width, self.block_type
        cx = W // 2

        for c in range(W):
            if kind == "industrial":
                roof = {"Metal": 0.5, "Tejado": 0.3, "Chimenea": 0.2}
            elif kind == "residential" and c == cx:
                roof = {"Chimenea": 0.6, "Tejado": 0.4}
            else:
                roof = {"Tejado": 0.8, "Pared": 0.2}
            self._set(0, c, roof)

        wall = {"Metal": 0.5, "Hormigón": 0.5} if kind == "industrial" else {"Pared": 1.0}
        facade = {
            "industrial": {"Hormigón": 0.45, "Metal": 0.35, "Tubería": 0.15, "Ventana": 0.05},
            "commercial": {"Ventana": 0.5, "Escaparate": 0.3, "Pared": 0.2},
            "residential": {"Ventana": 0.55, "Pared": 0.3, "Suelo_madera": 0.15},
        }[kind]
        for r in range(1, H - 1):
            for c in range(W):
                self._set(r, c, wall if c in (0, W - 1) else facade)

        ground = {
            "industrial": {"Hormigón": 0.5, "Metal": 0.5},
            "commercial": {"Escaparate": 0.85, "Ventana": 0.15},
            "residential": {"Pared": 0.5, "Ventana": 0.5},
        }[kind]
        for c in range(W):
            if c == cx:
                self._set(H - 1, c, {"Puerta": 1.0})
            else:
                self._set(H - 1, c, ground)

    def _apply_border_constraints(self) -> None:
        """Keep edge tiles that can touch some tile of the neighbouring block."""
        own = TileConfig.get_tiles_for_block(self.block_type)
        for side, neighbour_type in sorted(self.neighbors.items()):
            theirs = TileConfig.get_tiles_for_block(neighbour_type)
            if not theirs:
                continue
            ok = {t for t in own if any(TileConfig.are_compatible(t, n) for n in theirs)}
            if side == "top":
                edge = [(0, c) for c in range(self.width)]
            elif side == "bottom":
                edge = [(self.height - 1, c) for c in range(self.width)]
            elif side == "left":
                edge = [(r, 0) for r in range(self.height)]
            else:
                edge = [(r, self.width - 1) for r in range(self.height)]
            for r, c in edge:
                cell = self.cells[r][c]
                kept = {t: w for t, w in cell.possible.items() if t in ok}
                if kept:
                    cell.possible = kept

    # -- generation ----------------------------------------------------------

    def collapse(self) -> bool:
        """Generate the block, retrying until it passes validation."""
        for attempt in range(1, self.attempts + 1):
            self.attempts_used = attempt
            if attempt > 1 and not self.initialize():
                return False
            if super().collapse():
                return True
        self.logger.debug("%s block still invalid after %d attempts", self.block_type, self.attempts)
        return False

    def validate_paths(self) -> bool:
        """Structural rules for rivers, roads, oases and buildings."""
        tilemap = self.tile_grid()
        if self.block_type == "river":
            return self.validator.validate_river_path(tilemap, self.neighbors)
        if self.block_type == "road":
            return self.validator.validate_road_path(tilemap, self.neighbors)
        if self.block_type == "oasis":
            return self.validator.validate_oasis(tilemap)
        if self.block_type in BUILDING_TYPES:
            return self.validator.validate_building(tilemap, self.block_type)
        return True

    def validate(self) -> bool:
        for row in self.cells:
            for cell in row:
                if cell.collapsed_value is None or cell.collapsed_value not in self._allowed:
                    return False
        return self.validate_paths()
