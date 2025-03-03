from typing import Dict, Set

class RegionConfig:
    """Configuration for global regions."""
    
    TYPES: Dict[str, Dict[str, Set[str] | float]] = {
        "bosque":   {"can_adjacent": {"bosque", "desierto", "ciudad"}, "prob": 0.4},
        "desierto": {"can_adjacent": {"bosque", "desierto", "ciudad"}, "prob": 0.3},
        "ciudad":   {"can_adjacent": {"bosque", "desierto", "ciudad"}, "prob": 0.3}
    }

    @classmethod
    def get_probabilities(cls) -> Dict[str, float]:
        """Get probability distribution for regions."""
        return {region: config["prob"] for region, config in cls.TYPES.items()}

    @classmethod
    def are_compatible(cls, region1: str, region2: str) -> bool:
        """Check if two regions can be adjacent."""
        if region1 not in cls.TYPES or region2 not in cls.TYPES:
            return False
        return (region2 in cls.TYPES[region1]["can_adjacent"] and 
                region1 in cls.TYPES[region2]["can_adjacent"])
