from typing import Dict, Set, Union

class RegionConfig:
    """Configuration for global regions."""
    
    TYPES: Dict[str, Dict[str, Union[Set[str], float]]] = {
        "forest":   {"can_adjacent": {"forest", "desert", "city"}, "prob": 0.4},
        "desert": {"can_adjacent": {"forest", "desert", "city"}, "prob": 0.3},
        "city":   {"can_adjacent": {"forest", "desert", "city"}, "prob": 0.3}
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

        adjacent1 = cls.TYPES[region1].get("can_adjacent")
        adjacent2 = cls.TYPES[region2].get("can_adjacent")

        # Ensure we have sets to work with
        if not isinstance(adjacent1, set) or not isinstance(adjacent2, set):
            return False

        return (region2 in adjacent1 and region1 in adjacent2)
