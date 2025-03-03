from typing import NamedTuple

class GeneratorConfig(NamedTuple):
    min_region_size: int = 3
    max_region_size: int = 6
    min_block_size: int = 2
    max_block_size: int = 4
    min_local_size: int = 3
    max_local_size: int = 6
    
    region_distribution: dict = {
        "bosque": (0.3, 0.5),    # min, max percentage
        "desierto": (0.2, 0.4),
        "ciudad": (0.2, 0.4)
    }
    
    backtrack_limit: int = 100
    cache_size: int = 1024
