"""Procedural tile renderer: top-down and isometric views of a HAMC map.

Everything is drawn with Pillow primitives and seeded noise; there are no
image assets. Natural blocks (forest, water, desert, villages, transitions)
are painted tile by tile from the tilemap. City blocks are painted as lots:
each residential, commercial, industrial or park block becomes a lot with a
sidewalk and its own street frontage, so neighbouring lots form a street
grid; road blocks are drawn from their connections to neighbouring roads.

    from hamc import generate_map
    from hamc.visualization.painter import render_map

    render_map(generate_map(seed=42), style="iso").save("city.png")
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from ..core.validator import network_connections

RGB = Tuple[int, int, int]
Grid = Sequence[Sequence[str]]

STYLES = ("topdown", "iso", "flat")
LOTS = ("residential", "commercial", "industrial", "park")
URBAN = (*LOTS, "road")

# -- palette ------------------------------------------------------------------

GRASS = (104, 160, 72)
FOREST_FLOOR = (74, 118, 56)
URBAN_LAWN = (118, 178, 84)
DRY_GRASS = (150, 158, 90)
SAND = (226, 202, 150)
DIRT = (150, 112, 78)
WATER = (66, 146, 204)
DEEP_WATER = (40, 96, 166)
ASPHALT = (66, 68, 74)
SIDEWALK = (196, 193, 186)
CURB = (150, 148, 144)
CONCRETE = (172, 170, 164)
PATH = (204, 180, 132)
STONE = (150, 148, 146)
THATCH = (206, 166, 88)
TREE = (46, 112, 54)
TREE_DARK = (30, 80, 40)
BUSH = (84, 132, 64)

ROOFS = {
    "residential": [(176, 74, 58), (150, 84, 70), (120, 96, 110), (186, 112, 70), (98, 104, 118)],
    "commercial": [(198, 200, 204), (184, 186, 176), (210, 204, 190)],
    "industrial": [(140, 148, 156), (120, 130, 138), (160, 152, 140)],
}
AWNINGS = [(200, 60, 60), (40, 130, 90), (40, 100, 170), (220, 150, 40)]
CONTAINERS = [(186, 70, 50), (40, 110, 170), (60, 140, 90), (210, 160, 50)]

# Ground colour of each tile, also used by the flat style.
GROUND: Dict[str, RGB] = {
    "Hierba": GRASS, "Árbol": FOREST_FLOOR, "Musgo": (78, 122, 60), "Helecho": (70, 134, 76),
    "Hojas": (128, 140, 66), "Tronco": FOREST_FLOOR, "Tocón": GRASS, "Flores": GRASS,
    "Sendero": PATH, "Tierra": DIRT, "Agua": WATER, "Aguas_profundas": DEEP_WATER,
    "Junco": (92, 150, 110), "Lirio": WATER, "Palmera": SAND, "Arena": SAND,
    "Duna": (214, 186, 130), "Roca": SAND, "Cactus": SAND, "Hueso": SAND,
    "Arbusto": DRY_GRASS, "Cemento": CONCRETE, "Cesped_urbano": URBAN_LAWN, "Madera": GRASS,
    "Piedra": (176, 168, 150), "Tejado_pajizo": GRASS, "Pozo": STONE, "Asfalto": ASPHALT,
    "Linea": ASPHALT, "Paso_cebra": ASPHALT, "Acera": SIDEWALK, "Bordillo": CURB,
    "Farola": SIDEWALK,
}
WATERY = {"Agua", "Aguas_profundas", "Lirio"}


def _hash(*values: int) -> int:
    """Small deterministic hash (Python's hash() is salted per process)."""
    h = 2166136261
    for v in values:
        h = ((h ^ (v & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return h


def _shade(color: RGB, factor: float) -> RGB:
    return tuple(max(0, min(255, int(c * factor))) for c in color)  # type: ignore[return-value]


def _mix(a: RGB, b: RGB, t: float) -> RGB:
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


# -- tile textures ------------------------------------------------------------

def _speckle(img: Image.Image, rng: random.Random, base: RGB, amount: float, spread: float) -> None:
    px = img.load()
    w, h = img.size
    for _ in range(int(w * h * amount)):
        x, y = rng.randrange(w), rng.randrange(h)
        px[x, y] = _shade(base, 1 + rng.uniform(-spread, spread)) + (255,)


@lru_cache(maxsize=1024)
def tile_texture(tile: str, size: int, variant: int) -> Image.Image:
    """Ground texture of a tile (RGBA, ``size`` px). Sprites are drawn later."""
    base = GROUND.get(tile, (200, 0, 200))
    rng = random.Random(_hash(sum(map(ord, tile)), size, variant))
    base = _shade(base, 1 + rng.uniform(-0.04, 0.04))
    img = Image.new("RGBA", (size, size), base + (255,))
    d = ImageDraw.Draw(img)
    s = size

    if tile in ("Agua", "Aguas_profundas", "Lirio"):
        light = _shade(base, 1.18)
        for _ in range(max(1, s // 6)):
            x, y = rng.randrange(s), rng.randrange(2, s - 1)
            ln = max(2, s // 4)
            d.arc([x - ln, y - 2, x + ln, y + 2], 200, 340, fill=light)
    elif tile == "Duna":
        for k in range(2):
            y = rng.randrange(s // 4, s)
            d.arc([-s // 2, y - s // 3, s + s // 2, y + s // 3], 200, 340, fill=_shade(base, 0.86))
        _speckle(img, rng, base, 0.15, 0.08)
    elif tile in ("Cemento", "Acera"):
        _speckle(img, rng, base, 0.1, 0.05)
        step = max(3, s // 2)
        for k in range(0, s, step):
            d.line([(k, 0), (k, s)], fill=_shade(base, 0.9))
            d.line([(0, k), (s, k)], fill=_shade(base, 0.9))
    elif tile == "Piedra":
        step = max(3, s // 3)
        for y in range(0, s, step):
            off = (y // step % 2) * step // 2
            for x in range(-off, s, step):
                d.rectangle([x + 1, y + 1, x + step - 1, y + step - 1],
                            fill=_shade(base, 1 + rng.uniform(-0.1, 0.1)))
    elif tile == "Cesped_urbano":
        if variant % 2:
            d.rectangle([0, 0, s, s // 2], fill=_shade(base, 1.05) + (255,))
        _speckle(img, rng, base, 0.12, 0.08)
    elif tile in ("Asfalto", "Linea", "Paso_cebra"):
        _speckle(img, rng, base, 0.2, 0.12)
        if tile == "Linea":
            d.line([(s // 4, s // 2), (3 * s // 4, s // 2)], fill=(232, 210, 110), width=max(1, s // 10))
        if tile == "Paso_cebra":
            for x in range(1, s, max(2, s // 4)):
                d.rectangle([x, 1, x + max(1, s // 8), s - 2], fill=(232, 232, 228))
    elif tile == "Bordillo":
        d.rectangle([0, 0, s, s // 3], fill=SIDEWALK + (255,))
    else:
        spread = 0.16 if tile in ("Hierba", "Árbol", "Musgo", "Helecho", "Hojas", "Flores", "Tocón",
                                  "Junco", "Arbusto", "Tronco") else 0.1
        _speckle(img, rng, base, 0.25, spread)
        if tile in ("Hierba", "Flores", "Arbusto"):
            for _ in range(max(1, s // 5)):
                x, y = rng.randrange(s), rng.randrange(2, s)
                d.line([(x, y), (x + rng.choice((-1, 1)), y - 2)], fill=_shade(base, 1.25))
    return img


# -- scene description --------------------------------------------------------

@dataclass
class Building:
    x0: float
    y0: float
    x1: float
    y1: float
    height: float
    roof: RGB
    wall: RGB
    kind: str
    ridge: Optional[str] = None  # "h" / "v": gable roof ridge direction
    chimneys: List[Tuple[float, float]] = field(default_factory=list)
    rooftop: List[Tuple[float, float, float]] = field(default_factory=list)  # AC units (x, y, size)
    sawtooth: bool = False


@dataclass
class Tree:
    x: float
    y: float
    r: float
    color: RGB
    palm: bool = False


@dataclass
class Scene:
    width: int
    height: int
    tile: int
    ground: Image.Image
    buildings: List[Building]
    trees: List[Tree]


class _Builder:
    def __init__(self, blocks: Grid, tilemap: Grid, local: int, tile: int, seed: int):
        self.blocks = blocks
        self.tilemap = tilemap
        self.local = local
        self.T = tile
        self.seed = seed
        self.rows, self.cols = len(tilemap), len(tilemap[0])
        self.ground = Image.new("RGBA", (self.cols * tile, self.rows * tile), GRASS + (255,))
        self.draw = ImageDraw.Draw(self.ground)
        self.buildings: List[Building] = []
        self.trees: List[Tree] = []

    def rng(self, *key: int) -> random.Random:
        return random.Random(_hash(self.seed, *key))

    def block_type(self, br: int, bc: int) -> Optional[str]:
        if 0 <= br < len(self.blocks) and 0 <= bc < len(self.blocks[0]):
            return self.blocks[br][bc]
        return None

    def block_tiles(self, br: int, bc: int) -> List[List[str]]:
        L = self.local
        return [row[bc * L:(bc + 1) * L] for row in self.tilemap[br * L:(br + 1) * L]]

    def build(self) -> Scene:
        for br in range(len(self.blocks)):
            for bc in range(len(self.blocks[0])):
                kind = self.blocks[br][bc]
                if kind in LOTS:
                    self.lot(br, bc, kind)
                elif kind == "road":
                    self.road(br, bc)
                elif kind == "periurban":
                    self.suburb(br, bc)
                else:
                    self.natural(br, bc)
        self.street_markings()
        self.trees.sort(key=lambda t: t.y)
        return Scene(self.cols * self.T, self.rows * self.T, self.T, self.ground,
                     self.buildings, self.trees)

    # natural blocks: tile by tile ------------------------------------------

    def natural(self, br: int, bc: int) -> None:
        T, L = self.T, self.local
        tiles = self.block_tiles(br, bc)
        for r in range(L):
            for c in range(L):
                gr, gc = br * L + r, bc * L + c
                tile = tiles[r][c]
                x, y = gc * T, gr * T
                self.ground.paste(tile_texture(tile, T, _hash(gr, gc, self.seed) % 4), (x, y))
                if tile in WATERY:
                    self.shore(gr, gc, x, y)
                self.sprite(tile, x, y, self.rng(gr, gc))

    def suburb(self, br: int, bc: int) -> None:
        """Periurban blocks: a lawn plot with a small house, driveway and trees.
        Paved tiles become lawn so the fringe doesn't read as noise; paths and
        sand keep their texture."""
        T, L = self.T, self.local
        tiles = self.block_tiles(br, bc)
        rng = self.rng(br, bc, 17)
        for r in range(L):
            for c in range(L):
                gr, gc = br * L + r, bc * L + c
                tile = tiles[r][c] if tiles[r][c] in ("Sendero", "Arena") else "Cesped_urbano"
                self.ground.paste(tile_texture(tile, T, _hash(gr, gc, self.seed) % 4), (gc * T, gr * T))
        S = L * T
        x0, y0 = bc * S, br * S
        w = S * rng.uniform(0.32, 0.45)
        h = S * rng.uniform(0.28, 0.4)
        hx = x0 + rng.uniform(S * 0.12, S * 0.88 - w)
        hy = y0 + rng.uniform(S * 0.12, S * 0.6 - h)
        drive = max(2, T * 0.6)
        dx = hx + w * 0.7
        self.draw.rectangle([dx, hy + h, dx + drive, y0 + S - 1], fill=CONCRETE)
        self.buildings.append(Building(hx, hy, hx + w, hy + h, T * rng.uniform(0.9, 1.3),
                                       rng.choice(ROOFS["residential"]), (230, 220, 204), "house",
                                       ridge="h" if w >= h else "v"))
        for _ in range(rng.randint(1, 3)):
            tx, ty = x0 + rng.uniform(T, S - T), y0 + rng.uniform(T, S - T)
            if not (hx - T < tx < hx + w + T and hy - T < ty < hy + h + T):
                self.trees.append(Tree(tx, ty, T * rng.uniform(0.45, 0.65), _shade(TREE, rng.uniform(0.95, 1.2))))

    def shore(self, gr: int, gc: int, x: int, y: int) -> None:
        T = self.T
        foam = (178, 214, 232)
        w = max(1, T // 8)
        for dr, dc, box in ((-1, 0, [x, y, x + T - 1, y + w - 1]), (1, 0, [x, y + T - w, x + T - 1, y + T - 1]),
                            (0, -1, [x, y, x + w - 1, y + T - 1]), (0, 1, [x + T - w, y, x + T - 1, y + T - 1])):
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.tilemap[nr][nc] not in WATERY:
                self.draw.rectangle(box, fill=foam)

    def sprite(self, tile: str, x: int, y: int, rng: random.Random) -> None:
        T, d = self.T, self.draw
        cx, cy = x + T / 2 + rng.uniform(-T / 8, T / 8), y + T / 2 + rng.uniform(-T / 8, T / 8)
        if tile == "Árbol":
            self.trees.append(Tree(cx, cy, T * rng.uniform(0.55, 0.72), _shade(TREE, rng.uniform(0.85, 1.1))))
        elif tile == "Palmera":
            self.trees.append(Tree(cx, cy, T * 0.6, (70, 150, 70), palm=True))
        elif tile == "Arbusto":
            r = T * 0.3
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BUSH, outline=_shade(BUSH, 0.7))
        elif tile == "Roca":
            r = T * rng.uniform(0.2, 0.32)
            d.ellipse([cx - r, cy - r * 0.8, cx + r, cy + r * 0.8], fill=(140, 136, 128), outline=(100, 98, 94))
            d.ellipse([cx - r * 0.5, cy - r * 0.6, cx, cy - r * 0.1], fill=(176, 172, 164))
        elif tile == "Cactus":
            w = max(1, T // 8)
            d.rectangle([cx - w, cy - T * 0.3, cx + w, cy + T * 0.3], fill=(70, 130, 70))
            d.rectangle([cx - T * 0.25, cy - T * 0.05, cx - w, cy + w * 0.5], fill=(70, 130, 70))
        elif tile == "Hueso":
            d.line([(cx - T * 0.2, cy), (cx + T * 0.2, cy + T * 0.08)], fill=(240, 236, 220), width=max(1, T // 10))
        elif tile == "Flores":
            for _ in range(4):
                fx, fy = x + rng.uniform(1, T - 2), y + rng.uniform(1, T - 2)
                d.rectangle([fx, fy, fx + max(1, T // 10), fy + max(1, T // 10)],
                            fill=rng.choice([(236, 120, 170), (250, 220, 90), (240, 240, 250), (190, 120, 220)]))
        elif tile == "Lirio":
            r = T * 0.22
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(70, 150, 70))
            d.rectangle([cx, cy - 1, cx + 1, cy], fill=(240, 150, 190))
        elif tile == "Junco":
            for k in range(3):
                kx = x + T * (0.25 + 0.25 * k)
                d.line([(kx, y + T * 0.85), (kx + rng.uniform(-1, 1), y + T * 0.2)], fill=(90, 120, 60))
        elif tile == "Tocón":
            r = T * 0.22
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(120, 84, 52), outline=(84, 58, 36))
        elif tile == "Tronco":
            d.rounded_rectangle([x + T * 0.15, cy - T * 0.12, x + T * 0.85, cy + T * 0.12],
                                radius=max(1, T // 8), fill=(110, 76, 46))
        elif tile == "Helecho":
            for k in range(3):
                d.line([(cx, cy), (cx + rng.uniform(-T / 3, T / 3), cy - rng.uniform(1, T / 3))], fill=(50, 110, 60))
        elif tile == "Madera":
            if rng.random() > 0.35:  # most wood tiles are fences, a few are sheds
                d.line([(x + 1, cy), (x + T - 2, cy)], fill=(120, 84, 50), width=max(1, T // 10))
                return
            self.buildings.append(Building(x + 1, y + 1, x + T - 1, y + T - 1, T * 0.8, (128, 86, 54),
                                           (178, 140, 100), "hut", ridge="h" if rng.random() < 0.5 else "v"))
        elif tile == "Tejado_pajizo":
            self.buildings.append(Building(x + 1, y + 1, x + T - 1, y + T - 1, T * 0.7, THATCH,
                                           (170, 130, 90), "hut", ridge="h" if rng.random() < 0.5 else "v"))
        elif tile == "Pozo":
            r = T * 0.3
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(120, 120, 124), outline=(80, 80, 84))
            d.ellipse([cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55], fill=DEEP_WATER)
        elif tile == "Farola":
            self.lamp(cx, cy)

    def lamp(self, x: float, y: float) -> None:
        r = max(1.0, self.T / 8)
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=(250, 220, 120), outline=(90, 90, 90))

    # city -------------------------------------------------------------------

    def street_width(self) -> Tuple[int, int]:
        """Half street width and sidewalk width, in px."""
        T = self.T
        return max(2, round(T * 0.6)), max(1, round(T * 0.4))

    def lot(self, br: int, bc: int, kind: str) -> None:
        T, L = self.T, self.local
        S = L * T
        x0, y0 = bc * S, br * S
        hw, sw = self.street_width()
        d = self.draw
        d.rectangle([x0, y0, x0 + S - 1, y0 + S - 1], fill=ASPHALT)
        rng = self.rng(br, bc, 7)
        # asphalt grain
        for _ in range(S * S // 18):
            px, py = x0 + rng.randrange(S), y0 + rng.randrange(S)
            d.point((px, py), fill=_shade(ASPHALT, rng.uniform(0.85, 1.2)))
        d.rounded_rectangle([x0 + hw, y0 + hw, x0 + S - 1 - hw, y0 + S - 1 - hw],
                            radius=max(1, sw), fill=SIDEWALK, outline=CURB)
        ix0, iy0, ix1, iy1 = x0 + hw + sw, y0 + hw + sw, x0 + S - 1 - hw - sw, y0 + S - 1 - hw - sw
        tiles = self.block_tiles(br, bc)
        count = {t: sum(row.count(t) for row in tiles) for t in ("Chimenea", "Escaparate", "Ventana", "Tubería")}
        inner = (ix0, iy0, ix1, iy1)
        if kind == "park":
            self.park(inner, tiles, rng)
        elif kind == "residential":
            self.houses(inner, count, rng)
        elif kind == "commercial":
            self.shops(inner, count, rng)
        else:
            self.factory(inner, count, rng)
        # street trees on the sidewalk corners
        if L >= 5:
            for tx, ty in ((x0 + hw + sw / 2, y0 + S / 2), (x0 + S - hw - sw / 2, y0 + S / 2)):
                if rng.random() < 0.5:
                    self.trees.append(Tree(tx, ty, T * 0.35, _shade(TREE, 1.1)))

    def park(self, inner, tiles, rng) -> None:
        ix0, iy0, ix1, iy1 = inner
        d = self.draw
        d.rectangle([ix0, iy0, ix1, iy1], fill=URBAN_LAWN)
        band = max(2, self.T // 2)
        for k, y in enumerate(range(int(iy0), int(iy1), band)):
            if k % 2:
                d.rectangle([ix0, y, ix1, min(iy1, y + band - 1)], fill=_shade(URBAN_LAWN, 1.06))
        L = len(tiles)
        sx, sy = (ix1 - ix0) / L, (iy1 - iy0) / L
        for r in range(L):
            for c in range(L):
                cx, cy = ix0 + (c + 0.5) * sx, iy0 + (r + 0.5) * sy
                t = tiles[r][c]
                if t == "Sendero":
                    rr = max(sx, sy) * 0.62
                    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=PATH)
        for r in range(L):
            for c in range(L):
                cx, cy = ix0 + (c + 0.5) * sx, iy0 + (r + 0.5) * sy
                t = tiles[r][c]
                if t == "Árbol":
                    self.trees.append(Tree(cx, cy, min(sx, sy) * rng.uniform(0.55, 0.75),
                                           _shade(TREE, rng.uniform(0.9, 1.15))))
                elif t == "Flores":
                    for _ in range(5):
                        fx, fy = cx + rng.uniform(-sx / 3, sx / 3), cy + rng.uniform(-sy / 3, sy / 3)
                        d.rectangle([fx, fy, fx + 1, fy + 1],
                                    fill=rng.choice([(236, 120, 170), (250, 220, 90), (240, 240, 250)]))
                elif t == "Farola":
                    self.lamp(cx, cy)

    def houses(self, inner, count, rng) -> None:
        ix0, iy0, ix1, iy1 = inner
        d = self.draw
        d.rectangle([ix0, iy0, ix1, iy1], fill=_shade(GRASS, 1.08))
        w, h = ix1 - ix0, iy1 - iy0
        T = self.T
        n = 2 if (w > 4 * T and rng.random() < 0.6) else 1
        roof = rng.choice(ROOFS["residential"])
        gap = max(2, T * 0.3)
        for k in range(n):
            if n == 1:
                bx0, by0, bx1, by1 = ix0 + w * 0.12, iy0 + h * 0.14, ix1 - w * 0.12, iy1 - h * 0.18
            else:
                half = (w - gap) / 2
                bx0 = ix0 + k * (half + gap) + w * 0.04
                bx1 = bx0 + half - w * 0.08
                by0, by1 = iy0 + h * 0.12, iy1 - h * 0.2
            ridge = "h" if (bx1 - bx0) >= (by1 - by0) else "v"
            b = Building(bx0, by0, bx1, by1, T * rng.uniform(1.0, 1.6), _shade(roof, rng.uniform(0.92, 1.08)),
                         (226, 214, 196), "house", ridge=ridge)
            if count["Chimenea"]:
                b.chimneys.append((bx0 + (bx1 - bx0) * 0.25, by0 + (by1 - by0) * 0.3))
            self.buildings.append(b)
        # garden trees
        for _ in range(rng.randint(0, 2)):
            tx, ty = rng.choice([(ix0 + T * 0.5, iy1 - T * 0.4), (ix1 - T * 0.5, iy1 - T * 0.4)])
            self.trees.append(Tree(tx, ty, T * 0.45, _shade(TREE, 1.15)))
        d.rectangle([ix0, iy0, ix1, iy1], outline=_shade(GRASS, 0.8))

    def shops(self, inner, count, rng) -> None:
        ix0, iy0, ix1, iy1 = inner
        T = self.T
        w = ix1 - ix0
        b = Building(ix0 + 1, iy0 + 1, ix1 - 1, iy1 - T * 0.35, T * rng.uniform(2.2, 4.0),
                     rng.choice(ROOFS["commercial"]), (210, 206, 196), "shop")
        for _ in range(min(4, 1 + count["Ventana"] // 4)):
            s = T * 0.4
            b.rooftop.append((rng.uniform(b.x0 + s, b.x1 - 2 * s), rng.uniform(b.y0 + s, b.y1 - 2 * s), s))
        self.buildings.append(b)
        # shop-front awnings on the street side
        n = max(1, min(count["Escaparate"], int(w // (T * 0.9))))
        color = rng.choice(AWNINGS)
        aw = w / n
        for k in range(n):
            ax0 = ix0 + k * aw + 1
            stripes = max(2, int(aw // 3))
            for s in range(stripes):
                sx0 = ax0 + s * (aw - 2) / stripes
                self.draw.rectangle([sx0, iy1 - T * 0.35, sx0 + (aw - 2) / stripes, iy1],
                                    fill=color if s % 2 == 0 else (236, 236, 230))

    def factory(self, inner, count, rng) -> None:
        ix0, iy0, ix1, iy1 = inner
        T = self.T
        d = self.draw
        d.rectangle([ix0, iy0, ix1, iy1], fill=CONCRETE)
        for y in range(int(iy0), int(iy1), max(3, T)):
            d.line([(ix0, y), (ix1, y)], fill=_shade(CONCRETE, 0.93))
        w, h = ix1 - ix0, iy1 - iy0
        b = Building(ix0 + 1, iy0 + 1, ix0 + w * 0.72, iy1 - h * 0.12, T * rng.uniform(1.6, 2.4),
                     rng.choice(ROOFS["industrial"]), (150, 152, 150), "factory", sawtooth=rng.random() < 0.6)
        for k in range(min(2, count["Chimenea"] + (1 if count["Tubería"] else 0))):
            b.chimneys.append((b.x0 + (b.x1 - b.x0) * (0.3 + 0.4 * k), b.y0 + (b.y1 - b.y0) * 0.3))
        self.buildings.append(b)
        # containers in the yard
        cx0 = ix0 + w * 0.76
        for k in range(rng.randint(1, 3)):
            cy0 = iy0 + k * T * 0.8 + 2
            if cy0 + T * 0.6 < iy1:
                d.rectangle([cx0, cy0, ix1 - 2, cy0 + T * 0.55], fill=rng.choice(CONTAINERS), outline=(60, 60, 60))
        if count["Tubería"]:
            d.line([(ix0 + 2, iy1 - h * 0.06), (ix1 - 2, iy1 - h * 0.06)], fill=(110, 116, 128), width=max(1, T // 6))

    def road(self, br: int, bc: int) -> None:
        T, L = self.T, self.local
        S = L * T
        x0, y0 = bc * S, br * S
        d = self.draw
        neighbours = {side: self.block_type(br + dr, bc + dc)
                      for side, dr, dc in (("top", -1, 0), ("bottom", 1, 0), ("left", 0, -1), ("right", 0, 1))}
        d.rectangle([x0, y0, x0 + S - 1, y0 + S - 1], fill=ASPHALT)
        rng = self.rng(br, bc, 11)
        for _ in range(S * S // 18):
            px, py = x0 + rng.randrange(S), y0 + rng.randrange(S)
            d.point((px, py), fill=_shade(ASPHALT, rng.uniform(0.85, 1.2)))
        hw, sw = self.street_width()
        # sides that don't open onto another street get a sidewalk
        for side, other in neighbours.items():
            if other in URBAN:
                continue
            box = {"top": [x0, y0, x0 + S - 1, y0 + sw + 1], "bottom": [x0, y0 + S - sw - 2, x0 + S - 1, y0 + S - 1],
                   "left": [x0, y0, x0 + sw + 1, y0 + S - 1], "right": [x0 + S - sw - 2, y0, x0 + S - 1, y0 + S - 1]}[side]
            d.rectangle(box, fill=SIDEWALK, outline=CURB)
        conns = network_connections("road", {k: v for k, v in neighbours.items() if v})
        cx, cy = x0 + S / 2, y0 + S / 2
        dash = max(2, T // 2)
        lw = max(1, T // 10)
        yellow = (236, 206, 92)
        ends = {"top": (cx, y0), "bottom": (cx, y0 + S), "left": (x0, cy), "right": (x0 + S, cy)}
        hub = T * 0.9
        for side in conns:
            ex, ey = ends[side]
            steps = int(abs(ex - cx + ey - cy) // (2 * dash))
            for k in range(steps):
                t0, t1 = k * 2 * dash, k * 2 * dash + dash
                if t1 > S / 2 - hub:
                    break
                fx = lambda t: ex + (cx - ex) * t / (S / 2)  # noqa: E731
                fy = lambda t: ey + (cy - ey) * t / (S / 2)  # noqa: E731
                d.line([(fx(t0), fy(t0)), (fx(t1), fy(t1))], fill=yellow, width=lw)
        if len(conns) >= 3 or (conns & {"top", "bottom"} and conns & {"left", "right"}):
            r = hub * 0.75
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=URBAN_LAWN, outline=CURB)
            self.trees.append(Tree(cx, cy, r * 0.7, TREE))
            for side in conns:
                ex, ey = ends[side]
                mx, my = cx + (ex - cx) * 0.55, cy + (ey - cy) * 0.55
                self.zebra(mx, my, horizontal=side in ("left", "right"), span=hw * 1.6)
        lamp_rng = self.rng(br, bc, 13)
        if lamp_rng.random() < 0.7:
            self.lamp(x0 + sw / 2 + 1, y0 + sw / 2 + 1)

    def zebra(self, x: float, y: float, horizontal: bool, span: float) -> None:
        """Crosswalk centred at (x, y) across a street running horizontally or vertically."""
        T = self.T
        bar = max(1, T // 8)
        depth = max(2, T * 0.35)
        n = max(3, int(span * 2 // (2 * bar)))
        for k in range(n):
            off = -span + k * 2 * bar * (2 * span / (n * 2 * bar))
            if horizontal:
                self.draw.rectangle([x - depth / 2, y + off, x + depth / 2, y + off + bar - 1], fill=(232, 232, 228))
            else:
                self.draw.rectangle([x + off, y - depth / 2, x + off + bar - 1, y + depth / 2], fill=(232, 232, 228))

    def street_markings(self) -> None:
        """Dashed centre lines on streets between two lots, zebras at crossings."""
        T, L = self.T, self.local
        S = L * T
        hw, _ = self.street_width()
        dash = max(2, T // 2)
        lw = max(1, T // 12)
        white = (226, 226, 220)
        rows, cols = len(self.blocks), len(self.blocks[0])
        for br in range(rows):
            for bc in range(cols):
                if self.blocks[br][bc] not in LOTS:
                    continue
                x0, y0 = bc * S, br * S
                if bc + 1 < cols and self.blocks[br][bc + 1] in LOTS:
                    x = x0 + S
                    for y in range(int(y0 + hw * 2), int(y0 + S - hw * 2), 2 * dash):
                        self.draw.line([(x, y), (x, y + dash)], fill=white, width=lw)
                if br + 1 < rows and self.blocks[br + 1][bc] in LOTS:
                    y = y0 + S
                    for x in range(int(x0 + hw * 2), int(x0 + S - hw * 2), 2 * dash):
                        self.draw.line([(x, y), (x + dash, y)], fill=white, width=lw)
        if T < 8:
            return
        for br in range(1, rows):
            for bc in range(1, cols):
                around = [self.blocks[br - 1][bc - 1], self.blocks[br - 1][bc], self.blocks[br][bc - 1],
                          self.blocks[br][bc]]
                if all(b in LOTS for b in around):
                    vx, vy = bc * S, br * S
                    off = hw * 1.6
                    self.zebra(vx + off, vy, horizontal=True, span=hw * 0.9)
                    self.zebra(vx - off, vy, horizontal=True, span=hw * 0.9)
                    self.zebra(vx, vy + off, horizontal=False, span=hw * 0.9)
                    self.zebra(vx, vy - off, horizontal=False, span=hw * 0.9)


def build_scene(blocks: Grid, tilemap: Grid, local: int, tile_size: int = 12, seed: int = 0) -> Scene:
    """Turn the block and tile layers into ground + buildings + trees."""
    return _Builder(blocks, tilemap, local, tile_size, seed).build()


# -- top-down -----------------------------------------------------------------

def _draw_topdown_building(d: ImageDraw.ImageDraw, b: Building, T: int) -> None:
    x0, y0, x1, y1 = b.x0, b.y0, b.x1, b.y1
    edge = _shade(b.roof, 0.7)
    if b.ridge:
        if b.ridge == "h":
            mid = (y0 + y1) / 2
            d.rectangle([x0, y0, x1, mid], fill=_shade(b.roof, 1.12))
            d.rectangle([x0, mid, x1, y1], fill=_shade(b.roof, 0.86))
            d.line([(x0, mid), (x1, mid)], fill=_shade(b.roof, 0.65), width=max(1, T // 10))
        else:
            mid = (x0 + x1) / 2
            d.rectangle([x0, y0, mid, y1], fill=_shade(b.roof, 1.12))
            d.rectangle([mid, y0, x1, y1], fill=_shade(b.roof, 0.86))
            d.line([(mid, y0), (mid, y1)], fill=_shade(b.roof, 0.65), width=max(1, T // 10))
        d.rectangle([x0, y0, x1, y1], outline=edge)
    else:
        d.rectangle([x0, y0, x1, y1], fill=b.roof, outline=edge)
        if b.sawtooth:
            step = max(3, T // 2)
            for x in range(int(x0) + step, int(x1), step):
                d.rectangle([x - step // 3, y0 + 1, x, y1 - 1], fill=_shade(b.roof, 0.78))
        elif b.kind == "shop":
            inset = max(1, T // 6)
            d.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=_shade(b.roof, 0.85))
    for ax, ay, s in b.rooftop:
        d.rectangle([ax, ay, ax + s, ay + s], fill=(150, 154, 160), outline=(96, 98, 104))
    for cx, cy in b.chimneys:
        r = max(1.5, T * 0.2)
        if b.kind == "factory":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(90, 86, 84), outline=(50, 50, 50))
            d.ellipse([cx - r * 0.5, cy - r * 0.5, cx + r * 0.5, cy + r * 0.5], fill=(30, 30, 30))
        else:
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(110, 70, 60), outline=(60, 40, 36))


def _draw_tree(d: ImageDraw.ImageDraw, t: Tree) -> None:
    if t.palm:
        for k in range(6):
            import math
            a = k * math.pi / 3
            d.line([(t.x, t.y), (t.x + math.cos(a) * t.r, t.y + math.sin(a) * t.r)], fill=t.color, width=2)
        d.ellipse([t.x - 1.5, t.y - 1.5, t.x + 1.5, t.y + 1.5], fill=(120, 90, 50))
        return
    r = t.r
    d.ellipse([t.x - r, t.y - r, t.x + r, t.y + r], fill=_shade(t.color, 0.8))
    d.ellipse([t.x - r * 0.9, t.y - r * 0.95, t.x + r * 0.8, t.y + r * 0.7], fill=t.color)
    d.ellipse([t.x - r * 0.55, t.y - r * 0.6, t.x + r * 0.05, t.y - r * 0.05], fill=_shade(t.color, 1.2))


def render_topdown(scene: Scene) -> Image.Image:
    T = scene.tile
    img = scene.ground.copy()
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    for b in scene.buildings:
        off = min(b.height * 0.45, T * 1.5)
        sd.rectangle([b.x0 + off, b.y0 + off, b.x1 + off, b.y1 + off], fill=(0, 0, 0, 90))
    for t in scene.trees:
        off = t.r * 0.45
        sd.ellipse([t.x - t.r + off, t.y - t.r + off, t.x + t.r + off, t.y + t.r + off], fill=(0, 0, 0, 70))
    img = Image.alpha_composite(img, shadow)
    d = ImageDraw.Draw(img)
    for b in scene.buildings:
        _draw_topdown_building(d, b, T)
    for t in scene.trees:
        _draw_tree(d, t)
    return img.convert("RGB")


# -- isometric ----------------------------------------------------------------

def _iso(x: float, y: float, z: float, H: float) -> Tuple[float, float]:
    """Top-down pixel (x, y) at height z -> isometric image coordinates."""
    return x - y + H, (x + y) / 2 - z


def render_iso(scene: Scene) -> Image.Image:
    """2:1 isometric projection: the ground is an affine warp of the top-down
    ground, buildings and trees are extruded on top in back-to-front order."""
    W, H, T = scene.width, scene.height, scene.tile
    top_margin = int(T * 5)
    out_w, out_h = W + H, (W + H) // 2
    ground = scene.ground.transform(
        (out_w, out_h), Image.AFFINE, (0.5, 1, -H / 2, -0.5, 1, H / 2),
        resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))
    img = Image.new("RGBA", (out_w, out_h + top_margin), (24, 30, 38, 255))
    img.alpha_composite(ground, (0, top_margin))
    d = ImageDraw.Draw(img)

    def P(x: float, y: float, z: float = 0.0) -> Tuple[float, float]:
        px, py = _iso(x, y, z, H)
        return px, py + top_margin

    items = [((b.x1 + b.y1), "b", b) for b in scene.buildings] + [((t.x + t.y), "t", t) for t in scene.trees]
    items.sort(key=lambda it: it[0])
    for _, kind, obj in items:
        if kind == "t":
            t: Tree = obj
            bx, by = P(t.x, t.y)
            h = t.r * (2.4 if t.palm else 1.6)
            d.line([(bx, by), (bx, by - h * 0.6)], fill=(100, 70, 44), width=max(1, T // 8))
            cx, cy = bx, by - h * 0.75
            r = t.r * 0.95
            if t.palm:
                import math
                for k in range(6):
                    a = k * math.pi / 3
                    d.line([(cx, cy), (cx + math.cos(a) * r * 1.2, cy + math.sin(a) * r * 0.5 + r * 0.2)],
                           fill=t.color, width=2)
            else:
                d.ellipse([cx - r, cy - r * 0.9, cx + r, cy + r * 0.9], fill=_shade(t.color, 0.82))
                d.ellipse([cx - r * 0.8, cy - r * 0.9, cx + r * 0.6, cy + r * 0.5], fill=t.color)
                d.ellipse([cx - r * 0.5, cy - r * 0.75, cx, cy - r * 0.2], fill=_shade(t.color, 1.2))
            continue
        b: Building = obj
        x0, y0, x1, y1, z = b.x0, b.y0, b.x1, b.y1, b.height
        south = _shade(b.wall, 0.88)
        east = _shade(b.wall, 0.7)
        d.polygon([P(x0, y1), P(x1, y1), P(x1, y1, z), P(x0, y1, z)], fill=south, outline=_shade(south, 0.8))
        d.polygon([P(x1, y0), P(x1, y1), P(x1, y1, z), P(x1, y0, z)], fill=east, outline=_shade(east, 0.8))
        if b.kind in ("shop", "factory", "house") and T >= 8:
            win = (250, 236, 170) if b.kind != "factory" else (120, 150, 170)
            floors = max(1, int(z // (T * 0.8)))
            for f in range(floors):
                zz = (f + 0.35) * z / floors
                for k in range(1, max(2, int((x1 - x0) // (T * 0.7)))):
                    wx = x0 + k * T * 0.7
                    if wx < x1 - T * 0.3:
                        d.line([P(wx, y1, zz), P(wx, y1, zz + T * 0.3)], fill=win, width=max(1, T // 8))
                for k in range(1, max(2, int((y1 - y0) // (T * 0.7)))):
                    wy = y0 + k * T * 0.7
                    if wy < y1 - T * 0.3:
                        d.line([P(x1, wy, zz), P(x1, wy, zz + T * 0.3)], fill=_shade(win, 0.8), width=max(1, T // 8))
        if b.ridge:
            rz = z + min(x1 - x0, y1 - y0) * 0.45
            if b.ridge == "h":
                ym = (y0 + y1) / 2
                d.polygon([P(x0, y0, z), P(x1, y0, z), P(x1, ym, rz), P(x0, ym, rz)], fill=_shade(b.roof, 1.1))
                d.polygon([P(x0, ym, rz), P(x1, ym, rz), P(x1, y1, z), P(x0, y1, z)], fill=_shade(b.roof, 0.9))
                d.polygon([P(x1, y0, z), P(x1, ym, rz), P(x1, y1, z)], fill=east)
            else:
                xm = (x0 + x1) / 2
                d.polygon([P(x0, y0, z), P(xm, y0, rz), P(xm, y1, rz), P(x0, y1, z)], fill=_shade(b.roof, 1.0))
                d.polygon([P(xm, y0, rz), P(x1, y0, z), P(x1, y1, z), P(xm, y1, rz)], fill=_shade(b.roof, 0.82))
                d.polygon([P(x0, y1, z), P(xm, y1, rz), P(x1, y1, z)], fill=south)
        else:
            d.polygon([P(x0, y0, z), P(x1, y0, z), P(x1, y1, z), P(x0, y1, z)], fill=b.roof,
                      outline=_shade(b.roof, 0.75))
            if b.sawtooth:
                step = max(3, T // 2)
                for x in range(int(x0) + step, int(x1), step):
                    d.polygon([P(x - step, y0, z), P(x, y0, z + step * 0.6), P(x, y1, z + step * 0.6),
                               P(x - step, y1, z)], fill=_shade(b.roof, 0.9), outline=_shade(b.roof, 0.7))
            for ax, ay, s in b.rooftop:
                d.polygon([P(ax, ay, z + s * 0.6), P(ax + s, ay, z + s * 0.6), P(ax + s, ay + s, z + s * 0.6),
                           P(ax, ay + s, z + s * 0.6)], fill=(170, 174, 180), outline=(100, 102, 108))
        for cx, cy in b.chimneys:
            r = max(1.5, T * 0.18)
            top = z + (T * 1.6 if b.kind == "factory" else T * 0.9)
            col = (96, 90, 88) if b.kind == "factory" else (120, 74, 62)
            d.polygon([P(cx - r, cy + r, z), P(cx + r, cy + r, z), P(cx + r, cy + r, top), P(cx - r, cy + r, top)],
                      fill=col)
            d.polygon([P(cx + r, cy - r, z), P(cx + r, cy + r, z), P(cx + r, cy + r, top), P(cx + r, cy - r, top)],
                      fill=_shade(col, 0.75))
            d.polygon([P(cx - r, cy - r, top), P(cx + r, cy - r, top), P(cx + r, cy + r, top), P(cx - r, cy + r, top)],
                      fill=_shade(col, 0.6))
    bbox = img.getbbox()
    return img.crop(bbox).convert("RGB") if bbox else img.convert("RGB")


# -- flat ---------------------------------------------------------------------

def render_flat(tilemap: Grid, tile_size: int = 12) -> Image.Image:
    """One flat colour per tile (the colours of TileConfig)."""
    from ..config.tile_config import TileConfig

    rows, cols = len(tilemap), len(tilemap[0])
    img = Image.new("RGB", (cols, rows))
    img.putdata([TileConfig.get_color(t) for row in tilemap for t in row])
    return img.resize((cols * tile_size, rows * tile_size), Image.NEAREST)


def render_map(hmap, style: str = "topdown", tile_size: int = 12) -> Image.Image:
    """Render a :class:`~hamc.pipeline.HAMCMap` (or any object with ``blocks``,
    ``tilemap``, ``local`` and ``seed``) as ``topdown``, ``iso`` or ``flat``."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")
    if tile_size < 4:
        raise ValueError("tile_size must be >= 4")
    if style == "flat":
        return render_flat(hmap.tilemap, tile_size)
    scene = build_scene(hmap.blocks, hmap.tilemap, hmap.local, tile_size, hmap.seed or 0)
    return render_iso(scene) if style == "iso" else render_topdown(scene)
