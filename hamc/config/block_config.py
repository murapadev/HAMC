from typing import Dict, Set, List, Tuple

class BlockConfig:
    """Configuration for intermediate level blocks."""
    
    # Region to block mappings
    REGION_TO_BLOCKS: Dict[str, List[Tuple[str, float]]] = {
        "bosque": [
            ("arboleda", 0.6),
            ("claro", 0.3),
            ("rio", 0.1)
        ],
        "desierto": [
            ("arena", 0.7),
            ("oasis", 0.2),
            ("rio", 0.1)
        ],
        "ciudad": [
            ("residencial", 0.4),
            ("comercial", 0.3),
            ("industrial", 0.2),
            ("carretera", 0.1)
        ]
    }

    # Transition blocks between regions
    TRANSITIONS: Dict[str, Dict[str, Tuple[str, float]]] = {
        "bosque": {
            "desierto": ("matorral", 0.8),
            "ciudad": ("periurbano", 0.8)
        },
        "desierto": {
            "bosque": ("matorral", 0.8),
            "ciudad": ("aldea", 0.8)
        },
        "ciudad": {
            "bosque": ("periurbano", 0.8),
            "desierto": ("aldea", 0.8)
        }
    }

    # Block compatibility rules
    COMPATIBILITY: Dict[str, Set[str]] = {
        "arboleda": {"arboleda", "claro", "rio", "matorral", "periurbano"},
        "claro": {"arboleda", "claro", "rio", "matorral", "periurbano"},  # Added periurbano
        "rio": {"arboleda", "claro", "arena", "oasis", "rio", "periurbano"},  # Added periurbano
        "arena": {"arena", "oasis", "rio", "matorral", "aldea"},
        "oasis": {"arena", "rio", "matorral"},
        "residencial": {"residencial", "comercial", "carretera", "periurbano"},
        "comercial": {"residencial", "comercial", "industrial", "carretera", "periurbano"},
        "industrial": {"comercial", "industrial", "carretera", "periurbano"},
        "carretera": {"residencial", "comercial", "industrial", "periurbano", "aldea"},
        "matorral": {"arboleda", "claro", "arena", "oasis", "periurbano", "matorral"},  # Added matorral
        "periurbano": {"arboleda", "residencial", "comercial", "industrial", "carretera", "matorral", "claro", "periurbano", "rio"},  # Added rio
        "aldea": {"arena", "carretera"}
    }

    @classmethod
    def get_blocks_for_region(cls, region: str) -> Dict[str, float]:
        """Get block distribution for a region."""
        if region not in cls.REGION_TO_BLOCKS:
            return {}
        return {block: weight for block, weight in cls.REGION_TO_BLOCKS[region]}

    @classmethod
    def get_transition(cls, region1: str, region2: str) -> Tuple[str, float]:
        """Get transition block between two regions."""
        if (region1 in cls.TRANSITIONS and 
            region2 in cls.TRANSITIONS[region1]):
            return cls.TRANSITIONS[region1][region2]
        return ("", 0.0)

    @classmethod
    def are_compatible(cls, block1: str, block2: str) -> bool:
        """Check if two blocks can be adjacent."""
        if block1 not in cls.COMPATIBILITY or block2 not in cls.COMPATIBILITY:
            return False
        # Check compatibility in both directions
        return block2 in cls.COMPATIBILITY[block1] and block1 in cls.COMPATIBILITY[block2]
