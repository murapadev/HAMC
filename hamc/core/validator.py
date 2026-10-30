"""Structural checks for local blocks (rivers, roads, oases and buildings)."""

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..config.advanced_config import get_validator_config

Tilemap = List[List[Optional[str]]]

SIDES = ("top", "bottom", "left", "right")

WATER_TILES = frozenset({"Agua", "Aguas_profundas", "Lirio"})
ROAD_TILES = frozenset({"Asfalto", "Linea", "Paso_cebra"})

# A network block with no matching neighbour still draws a straight segment
# along this axis, so isolated rivers and roads read as such.
DEFAULT_AXIS = {"river": ("top", "bottom"), "road": ("left", "right")}
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def network_connections(block_type: str, neighbors: Optional[Dict[str, str]]) -> Set[str]:
    """Sides of a river/road block that must reach the block edge.

    A side connects when the neighbouring block has the same type. A single
    connection is extended straight through so it never dead-ends, and an
    isolated block falls back to its default axis.
    """
    neighbors = neighbors or {}
    sides = {s for s in SIDES if neighbors.get(s) == block_type}
    if not sides:
        return set(DEFAULT_AXIS[block_type])
    if len(sides) == 1:
        (only,) = sides
        sides.add(OPPOSITE[only])
    return sides


def edge_anchor(side: str, height: int, width: int) -> Tuple[int, int]:
    """Cell in the middle of ``side`` where a network crosses the block edge."""
    return {
        "top": (0, width // 2),
        "bottom": (height - 1, width // 2),
        "left": (height // 2, 0),
        "right": (height // 2, width - 1),
    }[side]


def _connected(tilemap: Tilemap, anchors: Iterable[Tuple[int, int]], passable: Set[str]) -> bool:
    anchors = list(anchors)
    if not anchors or any(tilemap[r][c] not in passable for r, c in anchors):
        return False
    height, width = len(tilemap), len(tilemap[0])
    seen = {anchors[0]}
    queue = deque([anchors[0]])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in seen \
                    and tilemap[nr][nc] in passable:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return all(a in seen for a in anchors)


def _count(tilemap: Tilemap, tile: str) -> int:
    return sum(row.count(tile) for row in tilemap)


class PathValidator:
    """Validates connectivity and composition rules of a finished block."""

    @staticmethod
    def validate_network(tilemap: Tilemap, block_type: str,
                         neighbors: Optional[Dict[str, str]] = None) -> bool:
        """Every connected side must be reachable from every other one."""
        height, width = len(tilemap), len(tilemap[0])
        passable = WATER_TILES if block_type == "river" else ROAD_TILES
        anchors = [edge_anchor(s, height, width)
                   for s in sorted(network_connections(block_type, neighbors))]
        return _connected(tilemap, anchors, set(passable))

    @staticmethod
    def validate_river_path(tilemap: Tilemap, neighbors: Optional[Dict[str, str]] = None) -> bool:
        return PathValidator.validate_network(tilemap, "river", neighbors)

    @staticmethod
    def validate_road_path(tilemap: Tilemap, neighbors: Optional[Dict[str, str]] = None) -> bool:
        return PathValidator.validate_network(tilemap, "road", neighbors)

    @staticmethod
    def validate_oasis(tilemap: Tilemap) -> bool:
        """An oasis needs a central pond surrounded by sand."""
        config = get_validator_config()
        height, width = len(tilemap), len(tilemap[0])
        area = height * width
        if _count(tilemap, "Agua") < area * config.min_water_percentage:
            return False
        if _count(tilemap, "Arena") < area * 0.2:
            return False

        center_r, center_c = height // 2, width // 2
        radius = max(2, min(height, width) // 2 - 1)
        central = peripheral = 0
        for r in range(height):
            for c in range(width):
                if tilemap[r][c] == "Agua":
                    if ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5 <= radius:
                        central += 1
                    else:
                        peripheral += 1
        if central < peripheral:
            return False

        edge_water = sum(
            1 for r in range(height) for c in range(width)
            if tilemap[r][c] == "Agua" and (r in (0, height - 1) or c in (0, width - 1))
        )
        return edge_water <= 1

    @staticmethod
    def validate_building(tilemap: Tilemap, block_type: str) -> bool:
        """Composition rules per building type.

        Requirements scale down on 3x3 blocks, where a facade only has one
        free interior cell.
        """
        small = min(len(tilemap), len(tilemap[0])) <= 3

        if block_type == "residential":
            return _count(tilemap, "Puerta") >= 1 and _count(tilemap, "Ventana") >= (1 if small else 2)

        if block_type == "commercial":
            return _count(tilemap, "Escaparate") >= 2

        if block_type == "industrial":
            sections = PathValidator._find_continuous_sections(tilemap, {"Metal", "Hormigón"})
            return any(len(section) >= 4 for section in sections)

        return True

    @staticmethod
    def _find_continuous_sections(tilemap: Tilemap, valid_tiles: Set[str]) -> List[Set[Tuple[int, int]]]:
        """4-connected components made of ``valid_tiles``."""
        height, width = len(tilemap), len(tilemap[0])
        seen: Set[Tuple[int, int]] = set()
        sections = []
        for r0 in range(height):
            for c0 in range(width):
                if (r0, c0) in seen or tilemap[r0][c0] not in valid_tiles:
                    continue
                section = set()
                stack = [(r0, c0)]
                seen.add((r0, c0))
                while stack:
                    r, c = stack.pop()
                    section.add((r, c))
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in seen \
                                and tilemap[nr][nc] in valid_tiles:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
                sections.append(section)
        return sections
