from typing import Dict, Set, List, Tuple


class TileConfig:
    """Configuration for local-level tiles.

    Each block has a curated palette of tile types with weighted probabilities,
    plus a compatibility graph that constrains which tiles may sit next to each
    other. Tiles are grouped by biome (forest, water, desert, urban, village)
    so the resulting tilemaps read as coherent scenery rather than noise.
    """

    BLOCK_TO_TILES: Dict[str, List[Tuple[str, float]]] = {
        # Forest biome --------------------------------------------------------
        "grove": [
            ("Árbol",   0.55),
            ("Hierba",  0.22),
            ("Musgo",   0.12),
            ("Helecho", 0.08),
            ("Hojas",   0.03),
        ],
        "clearing": [
            ("Hierba",  0.55),
            ("Flores",  0.20),
            ("Sendero", 0.12),
            ("Tocón",   0.08),
            ("Helecho", 0.05),
        ],

        # Water ---------------------------------------------------------------
        "river": [
            ("Agua",            0.55),
            ("Aguas_profundas", 0.20),
            ("Junco",           0.12),
            ("Hierba",          0.10),
            ("Lirio",           0.03),
        ],

        # Desert --------------------------------------------------------------
        "sand": [
            ("Arena",  0.55),
            ("Duna",   0.22),
            ("Roca",   0.13),
            ("Cactus", 0.07),
            ("Hueso",  0.03),
        ],
        "oasis": [
            ("Agua",    0.30),
            ("Arena",   0.30),
            ("Hierba",  0.18),
            ("Palmera", 0.15),
            ("Lirio",   0.07),
        ],

        # City ----------------------------------------------------------------
        "residential": [
            ("Pared",         0.32),
            ("Ventana",       0.28),
            ("Puerta",        0.12),
            ("Tejado",        0.16),
            ("Cesped_urbano", 0.08),
            ("Chimenea",      0.04),
        ],
        "commercial": [
            ("Pared",      0.22),
            ("Ventana",    0.30),
            ("Escaparate", 0.22),
            ("Tejado",     0.14),
            ("Acera",      0.08),
            ("Farola",     0.04),
        ],
        "industrial": [
            ("Metal",    0.34),
            ("Hormigón", 0.34),
            ("Tubería",  0.16),
            ("Chimenea", 0.10),
            ("Tejado",   0.06),
        ],
        "road": [
            ("Asfalto",    0.42),
            ("Linea",      0.20),
            ("Acera",      0.18),
            ("Paso_cebra", 0.08),
            ("Bordillo",   0.08),
            ("Farola",     0.04),
        ],

        # Transitions ---------------------------------------------------------
        "scrubland": [
            ("Arbusto", 0.30),
            ("Hierba",  0.28),
            ("Flores",  0.16),
            ("Tierra",  0.16),
            ("Roca",    0.10),
        ],
        "periurban": [
            ("Asfalto",       0.22),
            ("Cemento",       0.22),
            ("Acera",         0.20),
            ("Cesped_urbano", 0.18),
            ("Arena",         0.10),
            ("Sendero",       0.08),
        ],
        "village": [
            ("Madera",        0.30),
            ("Hierba",        0.22),
            ("Piedra",        0.18),
            ("Tejado_pajizo", 0.14),
            ("Sendero",       0.10),
            ("Pozo",          0.06),
        ],
    }

    # Compatibility graph. The relation is symmetric *in spirit* but we list
    # both directions for clarity and to keep ``are_compatible`` simple.
    # Tiles not appearing in this map cannot be placed next to anything.
    COMPATIBILITY: Dict[str, Set[str]] = {
        # Forest --------------------------------------------------------------
        "Árbol":      {"Árbol", "Hierba", "Musgo", "Helecho", "Hojas", "Tronco"},
        "Hierba":     {"Árbol", "Hierba", "Flores", "Agua", "Arena", "Piedra",
                       "Arbusto", "Musgo", "Helecho", "Sendero", "Tocón",
                       "Tierra", "Cesped_urbano", "Junco", "Palmera"},
        "Flores":     {"Hierba", "Arbusto", "Flores", "Sendero", "Tierra", "Cesped_urbano"},
        "Musgo":      {"Árbol", "Musgo", "Hierba", "Helecho", "Tronco", "Piedra"},
        "Helecho":    {"Árbol", "Hierba", "Helecho", "Musgo", "Hojas", "Tierra"},
        "Hojas":      {"Árbol", "Helecho", "Hojas", "Tronco", "Musgo"},
        "Tronco":     {"Árbol", "Hojas", "Musgo", "Tronco", "Tierra"},
        "Tocón":      {"Hierba", "Sendero", "Tierra", "Tocón", "Flores"},
        "Sendero":    {"Hierba", "Flores", "Sendero", "Tierra", "Tocón",
                       "Arena", "Piedra", "Cesped_urbano", "Madera"},

        # Water ---------------------------------------------------------------
        "Agua":            {"Hierba", "Arena", "Agua", "Aguas_profundas",
                            "Junco", "Lirio", "Palmera"},
        "Aguas_profundas": {"Agua", "Aguas_profundas", "Lirio"},
        "Junco":           {"Agua", "Junco", "Hierba", "Tierra"},
        "Lirio":           {"Agua", "Aguas_profundas", "Lirio"},
        "Palmera":         {"Arena", "Hierba", "Agua", "Palmera"},

        # Desert --------------------------------------------------------------
        "Arena":   {"Arena", "Agua", "Roca", "Hierba", "Asfalto", "Arbusto",
                    "Duna", "Cactus", "Hueso", "Sendero", "Palmera"},
        "Duna":    {"Arena", "Duna", "Roca", "Hueso", "Cactus"},
        "Roca":    {"Arena", "Roca", "Piedra", "Duna", "Musgo", "Tierra"},
        "Cactus":  {"Arena", "Duna", "Cactus", "Tierra"},
        "Hueso":   {"Arena", "Duna", "Hueso", "Tierra"},

        # City --------------------------------------------------------------
        "Pared":        {"Puerta", "Ventana", "Pared", "Escaparate", "Tejado"},
        "Puerta":       {"Pared", "Ventana", "Acera", "Sendero", "Cesped_urbano"},
        "Ventana":      {"Pared", "Ventana", "Escaparate", "Tejado"},
        "Escaparate":   {"Ventana", "Hormigón", "Pared", "Acera"},
        "Tejado":       {"Pared", "Tejado", "Ventana", "Chimenea"},
        "Chimenea":     {"Tejado", "Chimenea", "Hormigón", "Metal"},
        "Suelo_madera": {"Pared", "Madera", "Puerta", "Suelo_madera"},
        "Metal":        {"Hormigón", "Metal", "Asfalto", "Tubería", "Chimenea"},
        "Hormigón":     {"Metal", "Escaparate", "Hormigón", "Cemento",
                         "Tubería", "Chimenea"},
        "Tubería":      {"Metal", "Hormigón", "Tubería", "Cemento"},
        "Asfalto":      {"Arena", "Metal", "Cemento", "Linea", "Acera",
                         "Bordillo", "Paso_cebra"},
        "Linea":        {"Asfalto", "Linea", "Paso_cebra"},
        "Acera":        {"Asfalto", "Acera", "Bordillo", "Cesped_urbano",
                         "Cemento", "Puerta", "Escaparate", "Farola", "Paso_cebra"},
        "Bordillo":     {"Asfalto", "Acera", "Bordillo", "Cesped_urbano"},
        "Paso_cebra":   {"Asfalto", "Linea", "Acera", "Paso_cebra"},
        "Farola":       {"Acera", "Cesped_urbano", "Farola"},

        # Transitions ---------------------------------------------------------
        "Arbusto":        {"Hierba", "Flores", "Arena", "Arbusto", "Tierra", "Cesped_urbano"},
        "Tierra":         {"Hierba", "Tierra", "Arbusto", "Roca", "Sendero",
                           "Tocón", "Helecho", "Hueso", "Cactus", "Junco",
                           "Tronco", "Flores"},
        "Cemento":        {"Asfalto", "Hormigón", "Arena", "Cemento", "Acera"},
        "Cesped_urbano":  {"Acera", "Bordillo", "Cesped_urbano", "Hierba",
                           "Sendero", "Flores", "Arbusto", "Farola", "Puerta"},

        # Village ----------------------------------------------------------
        "Madera":         {"Hierba", "Flores", "Piedra", "Madera", "Sendero",
                           "Suelo_madera", "Tejado_pajizo"},
        "Piedra":         {"Hierba", "Madera", "Piedra", "Sendero", "Musgo",
                           "Pozo", "Roca"},
        "Tejado_pajizo":  {"Madera", "Tejado_pajizo", "Piedra"},
        "Pozo":           {"Piedra", "Sendero", "Hierba", "Pozo"},
    }

    # Visual palette — picked for legibility on a dark canvas with smooth
    # transitions between adjacent biomes.
    COLORS: Dict[str, Tuple[int, int, int]] = {
        # Forest
        "Árbol":      (33, 110, 50),
        "Hierba":     (124, 198, 73),
        "Flores":     (236, 130, 184),
        "Musgo":      (74, 118, 56),
        "Helecho":    (60, 158, 130),
        "Hojas":      (190, 210, 95),
        "Tronco":     (110, 75, 47),
        "Tocón":      (148, 105, 70),
        "Sendero":    (192, 162, 110),

        # Water
        "Agua":            (76, 175, 230),
        "Aguas_profundas": (35, 89, 158),
        "Junco":           (60, 132, 102),
        "Lirio":           (210, 170, 210),
        "Palmera":         (52, 138, 92),

        # Desert
        "Arena":   (232, 207, 154),
        "Duna":    (200, 168, 110),
        "Roca":    (148, 142, 130),
        "Cactus":  (118, 158, 95),
        "Hueso":   (228, 220, 198),

        # City
        "Pared":        (148, 96, 60),
        "Puerta":       (78, 50, 32),
        "Ventana":      (170, 210, 230),
        "Escaparate":   (220, 232, 240),
        "Tejado":       (172, 70, 56),
        "Chimenea":     (90, 88, 88),
        "Suelo_madera": (180, 130, 78),
        "Metal":        (170, 175, 182),
        "Hormigón":     (132, 136, 140),
        "Tubería":      (88, 96, 110),
        "Asfalto":      (62, 62, 70),
        "Linea":        (240, 230, 130),
        "Acera":        (188, 190, 196),
        "Bordillo":     (94, 96, 102),
        "Paso_cebra":   (232, 232, 232),
        "Farola":       (244, 200, 70),

        # Transitions / village
        "Arbusto":         (94, 142, 72),
        "Tierra":          (148, 108, 80),
        "Cemento":         (114, 114, 118),
        "Cesped_urbano":   (130, 188, 96),
        "Madera":          (160, 110, 70),
        "Piedra":          (148, 148, 152),
        "Tejado_pajizo":   (208, 162, 84),
        "Pozo":            (78, 102, 128),

        # System
        "Empty": (16, 22, 30),
        "ERROR": (220, 60, 80),
    }

    @classmethod
    def get_tiles_for_block(cls, block: str) -> Dict[str, float]:
        """Get tile distribution for a block."""
        if block not in cls.BLOCK_TO_TILES:
            return {}
        return {tile: weight for tile, weight in cls.BLOCK_TO_TILES[block]}

    @classmethod
    def are_compatible(cls, tile1: str, tile2: str) -> bool:
        """Check if two tiles can be adjacent. Symmetric in practice."""
        a = cls.COMPATIBILITY.get(tile1)
        b = cls.COMPATIBILITY.get(tile2)
        if a is None or b is None:
            return False
        # The graph is symmetrized at import time (see _symmetrize below).
        return tile2 in a and tile1 in b

    @classmethod
    def get_color(cls, tile: str) -> Tuple[int, int, int]:
        """Get color for visualization."""
        return cls.COLORS.get(tile, cls.COLORS["ERROR"])


def _symmetrize(graph: Dict[str, Set[str]]) -> None:
    """Add the reverse edge for every adjacency so the graph is symmetric.

    ``are_compatible`` requires both directions; the hand-written table had
    one-way edges (e.g. ``Suelo_madera`` listed ``Pared`` but not vice versa)
    that rejected pairs the design intended to allow. Closing the graph keeps
    the data declarative while guaranteeing symmetry.
    """
    for node, neighbours in list(graph.items()):
        for other in list(neighbours):
            graph.setdefault(other, set()).add(node)


_symmetrize(TileConfig.COMPATIBILITY)
