from typing import Dict, Set, List, Tuple

class BlockConfig:
    """Configuration for intermediate level blocks."""
    
    # Region to block mappings
    REGION_TO_BLOCKS: Dict[str, List[Tuple[str, float]]] = {
        "forest": [
            ("grove", 0.6),
            ("clearing", 0.3),
            ("river", 0.1)
        ],
        "desert": [
            ("sand", 0.7),
            ("oasis", 0.2),
            ("river", 0.1)
        ],
        "city": [
            ("residential", 0.4),
            ("commercial", 0.3),
            ("industrial", 0.2),
            ("road", 0.1)
        ]
    }

    # Transition blocks between regions
    TRANSITIONS: Dict[str, Dict[str, Tuple[str, float]]] = {
        "forest": {
            "desert": ("scrubland", 0.8),
            "city": ("periurban", 0.8)
        },
        "desert": {
            "forest": ("scrubland", 0.8),
            "city": ("village", 0.8)
        },
        "city": {
            "forest": ("periurban", 0.8),
            "desert": ("village", 0.8)
        }
    }

    # Block compatibility rules
    COMPATIBILITY: Dict[str, Set[str]] = {
        "grove": {"grove", "clearing", "river", "scrubland", "periurban"},
        "clearing": {"grove", "clearing", "river", "scrubland", "periurban"},  # Added periurban
        "river": {"grove", "clearing", "sand", "oasis", "river", "periurban"},  # Added periurban
        "sand": {"sand", "oasis", "river", "scrubland", "village"},
        "oasis": {"sand", "river", "scrubland"},
        "residential": {"residential", "commercial", "road", "periurban"},
        "commercial": {"residential", "commercial", "industrial", "road", "periurban"},
        "industrial": {"commercial", "industrial", "road", "periurban"},
        "road": {"residential", "commercial", "industrial", "periurban", "village"},
        "scrubland": {"grove", "clearing", "sand", "oasis", "periurban", "scrubland"},  # Added scrubland
        "periurban": {"grove", "residential", "commercial", "industrial", "road", "scrubland", "clearing", "periurban", "river"},  # Added river
        "village": {"sand", "road"}
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
