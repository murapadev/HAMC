from typing import Dict, Set, List, Tuple

class TileConfig:
    """Configuration for local level tiles."""
    
    BLOCK_TO_TILES: Dict[str, List[Tuple[str, float]]] = {
        "arboleda":     [("Árbol", 0.7), ("Hierba", 0.3)],
        "claro":        [("Hierba", 0.7), ("Flores", 0.3)],
        "rio":          [("Agua", 0.8), ("Hierba", 0.2)],
        "arena":        [("Arena", 0.8), ("Roca", 0.2)],
        "oasis":        [("Agua", 0.4), ("Arena", 0.4), ("Hierba", 0.2)],
        "residencial":  [("Pared", 0.4), ("Puerta", 0.2), ("Ventana", 0.4)],
        "comercial":    [("Pared", 0.3), ("Ventana", 0.4), ("Escaparate", 0.3)],
        "industrial":   [("Metal", 0.5), ("Hormigón", 0.5)],
        "carretera":    [("Asfalto", 0.7), ("Linea", 0.3)],
        "matorral":     [("Arbusto", 0.5), ("Hierba", 0.3), ("Flores", 0.2)],
        "periurbano":   [("Asfalto", 0.3), ("Cemento", 0.4), ("Arena", 0.3)],
        "aldea":        [("Madera", 0.4), ("Hierba", 0.4), ("Piedra", 0.2)]
    }

    COMPATIBILITY: Dict[str, Set[str]] = {
        "Árbol":       {"Árbol", "Hierba"},
        "Hierba":      {"Árbol", "Hierba", "Flores", "Agua", "Arena", "Piedra", 
                       "Arbusto"},
        "Flores":      {"Hierba", "Arbusto", "Flores"},
        "Agua":        {"Hierba", "Arena", "Agua"},  # Added Agua to be compatible with itself
        "Roca":        {"Arena"},
        "Arena":       {"Arena", "Agua", "Roca", "Hierba", "Asfalto", "Arbusto"},
        "Pared":       {"Puerta", "Ventana", "Pared", "Escaparate"},
        "Puerta":      {"Pared", "Ventana"},
        "Ventana":     {"Pared", "Ventana", "Escaparate"},
        "Escaparate":  {"Ventana", "Hormigón", "Pared"},
        "Metal":       {"Hormigón", "Metal", "Asfalto"},
        "Hormigón":    {"Metal", "Escaparate", "Hormigón", "Cemento"},
        "Arbusto":     {"Hierba", "Flores", "Arena"},
        "Asfalto":     {"Arena", "Metal", "Cemento", "Linea"},
        "Cemento":     {"Asfalto", "Hormigón", "Arena"},
        "Madera":      {"Hierba", "Flores", "Piedra"},
        "Piedra":      {"Hierba", "Madera"},
        "Linea":       {"Asfalto", "Linea"}
    }

    COLORS: Dict[str, Tuple[int, int, int]] = {
        "Árbol":      (34, 139, 34),
        "Hierba":     (124, 252, 0),
        "Flores":     (255, 192, 203),
        "Agua":       (0, 191, 255),
        "Arena":      (238, 214, 175),
        "Roca":       (169, 169, 169),
        "Pared":      (139, 69, 19),
        "Puerta":     (160, 82, 45),
        "Ventana":    (176, 224, 230),
        "Escaparate": (240, 248, 255),
        "Metal":      (192, 192, 192),
        "Hormigón":   (128, 128, 128),
        "Arbusto":    (34, 139, 34),
        "Asfalto":    (70, 70, 70),
        "Cemento":    (100, 100, 100),
        "Madera":     (139, 69, 19),
        "Piedra":     (120, 120, 120),
        "Linea":      (255, 255, 255),
        "Empty":      (0, 0, 0),
        "ERROR":      (255, 0, 0)
    }

    @classmethod
    def get_tiles_for_block(cls, block: str) -> Dict[str, float]:
        """Get tile distribution for a block."""
        if block not in cls.BLOCK_TO_TILES:
            return {}
        return {tile: weight for tile, weight in cls.BLOCK_TO_TILES[block]}

    @classmethod
    def are_compatible(cls, tile1: str, tile2: str) -> bool:
        """Check if two tiles can be adjacent."""
        if tile1 not in cls.COMPATIBILITY or tile2 not in cls.COMPATIBILITY:
            return False
        return (tile2 in cls.COMPATIBILITY[tile1] and 
                tile1 in cls.COMPATIBILITY[tile2])

    @classmethod
    def get_color(cls, tile: str) -> Tuple[int, int, int]:
        """Get color for visualization."""
        return cls.COLORS.get(tile, cls.COLORS["ERROR"])
