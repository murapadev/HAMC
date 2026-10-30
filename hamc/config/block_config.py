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
            ("residential", 0.35),
            ("commercial", 0.25),
            ("industrial", 0.15),
            ("road", 0.15),
            ("park", 0.1),
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
        "clearing": {"grove", "clearing", "river", "scrubland", "periurban"},
        "river": {"grove", "clearing", "sand", "oasis", "river", "periurban"},
        "sand": {"sand", "oasis", "river", "scrubland", "village"},
        "oasis": {"oasis", "sand", "river", "scrubland"},
        "residential": {"residential", "commercial", "road", "periurban", "village", "park"},
        "commercial": {"residential", "commercial", "industrial", "road", "periurban", "village", "park"},
        "industrial": {"commercial", "industrial", "road", "periurban", "village"},
        "road": {"road", "residential", "commercial", "industrial", "periurban", "village", "park"},
        "park": {"park", "periurban"},
        # Transition blocks also meet each other where three regions touch.
        "scrubland": {"grove", "clearing", "sand", "oasis", "periurban", "scrubland", "river", "village"},
        "periurban": {"grove", "residential", "commercial", "industrial", "road", "scrubland",
                      "clearing", "periurban", "river", "village"},
        # village is the desert<->city transition block, so it must connect to
        # both the desert side (sand/road) and the city blocks it borders.
        "village": {"sand", "road", "residential", "commercial", "industrial", "village"},
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
        # The graph is symmetrized at import time, so a single-direction check
        # is sufficient; keep the bidirectional check as a cheap safety net.
        return block2 in cls.COMPATIBILITY[block1] and block1 in cls.COMPATIBILITY[block2]


def _symmetrize(graph: Dict[str, Set[str]]) -> None:
    """Add the reverse edge for every adjacency so the graph is symmetric.

    The compatibility tables are hand-written and drifted out of symmetry over
    time (e.g. ``village`` listed ``road`` but ``road`` did not list
    ``village``). ``are_compatible`` requires both directions, so any asymmetry
    produced spurious dead-ends and needless backtracking. Closing the graph
    here keeps the data declarative while guaranteeing the invariant.
    """
    for node, neighbours in list(graph.items()):
        for other in list(neighbours):
            graph.setdefault(other, set()).add(node)


_symmetrize(BlockConfig.COMPATIBILITY)
