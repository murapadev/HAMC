from typing import List, Dict, Set, Optional, Tuple, Union
from ..config.tile_config import TileConfig
from ..config.block_config import BlockConfig
import logging
from ..config.advanced_config import get_validator_config


class PathValidator:
    """Validates path constraints and connectivity requirements."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def validate_river_path(tilemap: List[List[Optional[str]]], 
                          neighbors: Optional[Dict[str, str]] = None) -> bool:
        """Validate vertical river path with neighbor constraints.
        
        Args:
            tilemap: 2D list of tile types (may contain None for uncollapsed cells)
            neighbors: Dictionary of neighbor block types
            
        Returns:
            bool: True if river path is valid
        """
        height, width = len(tilemap), len(tilemap[0])
        center = width // 2
        
        # Check if we need to connect to neighboring rivers
        needs_top = neighbors and neighbors.get('top') == 'river'
        needs_bottom = neighbors and neighbors.get('bottom') == 'river'
        
        # Validate top connection if needed
        if needs_top and (tilemap[0][center] is None or tilemap[0][center] != "Agua"):
            return False
            
        # Validate bottom connection if needed
        if needs_bottom and (tilemap[-1][center] is None or tilemap[-1][center] != "Agua"):
            return False
            
        # For river validation, we require a continuous path in the center column
        # Check that center column has water from top to bottom
        for r in range(height):
            if tilemap[r][center] != "Agua":
                return False
                
        return True

    @staticmethod
    def validate_road_path(tilemap: List[List[Optional[str]]], 
                         neighbors: Optional[Dict[str, str]] = None) -> bool:
        """Validate horizontal road path with neighbor constraints.
        
        Args:
            tilemap: 2D list of tile types (may contain None for uncollapsed cells)
            neighbors: Dictionary of neighbor block types
            
        Returns:
            bool: True if road path is valid
        """
        height, width = len(tilemap), len(tilemap[0])
        center = height // 2
        valid_tiles = {"Asfalto", "Linea"}
        
        # Check if we need to connect to neighboring roads
        needs_left = neighbors and neighbors.get('left') == 'road'
        needs_right = neighbors and neighbors.get('right') == 'road'
        
        # Validate left connection if needed
        if needs_left and (tilemap[center][0] is None or tilemap[center][0] not in valid_tiles):
            return False
            
        # Validate right connection if needed
        if needs_right and (tilemap[center][-1] is None or tilemap[center][-1] not in valid_tiles):
            return False
        
        # Find continuous road path
        visited = [[False]*width for _ in range(height)]
        start_pos = [(r, 0) for r in range(height) if tilemap[r][0] in valid_tiles]
        
        for start_r, start_c in start_pos:
            visited = [[False]*width for _ in range(height)]
            if PathValidator._find_horizontal_path(tilemap, start_r, start_c, visited, valid_tiles):
                return True
                
        return False

    @staticmethod
    def validate_oasis(tilemap: List[List[Optional[str]]]) -> bool:
        """Validate oasis has water and proper transitions.
        
        Args:
            tilemap: 2D list of tile types (may contain None for uncollapsed cells)
            
        Returns:
            bool: True if oasis layout is valid
        """
        config = get_validator_config()
        height, width = len(tilemap), len(tilemap[0])
        water_count = sum(1 for row in tilemap for tile in row if tile == "Agua")
        sand_count = sum(1 for row in tilemap for tile in row if tile == "Arena")
        
        # Oasis should have both water and sand with more relaxed proportions
        min_water = (height * width) * config.min_water_percentage  # Reduced from 0.2
        min_sand = (height * width) * 0.2   # Reduced from 0.3
        
        if water_count < min_water:
            return False
        
        if sand_count < min_sand:
            return False
            
        # Check that water is primarily in the central area
        # and not scattered randomly across the map
        center_r, center_c = height // 2, width // 2
        central_water = 0
        peripheral_water = 0
        
        for r in range(height):
            for c in range(width):
                if tilemap[r][c] == "Agua":
                    dist = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5
                    if dist <= max(2, min(height, width) // 3):
                        central_water += 1
                    else:
                        peripheral_water += 1
        
        # Most water should be central (relaxed constraint)
        if central_water < peripheral_water:
            return False
        
        # Water should not touch ALL map edges, but can touch some
        # Count edges with water
        edges_with_water = 0
        
        # Check top and bottom edges
        for c in range(width):
            if tilemap[0][c] == "Agua":
                edges_with_water += 1
            if tilemap[height-1][c] == "Agua":
                edges_with_water += 1
                
        # Check left and right edges
        for r in range(height):
            if tilemap[r][0] == "Agua":
                edges_with_water += 1
            if tilemap[r][width-1] == "Agua":
                edges_with_water += 1
                
        # Allow water to touch at most one edge
        if edges_with_water > 1:
            return False
            
        return True

    @staticmethod
    def validate_building(tilemap: List[List[Optional[str]]], block_type: str) -> bool:
        """Validate building blocks have proper structure.
        
        Args:
            tilemap: 2D list of tile types (may contain None for uncollapsed cells)
            block_type: Type of building block
            
        Returns:
            bool: True if building layout is valid
        """
        height, width = len(tilemap), len(tilemap[0])
        
        if block_type == "residential":
            # Residential blocks should have doors and windows
            door_count = sum(1 for row in tilemap for tile in row if tile == "Puerta")
            window_count = sum(1 for row in tilemap for tile in row if tile == "Ventana")
            
            if door_count == 0 or window_count < 2:
                return False
                
        elif block_type == "comercial":
            # Commercial blocks need display windows
            display_count = sum(1 for row in tilemap 
                              for tile in row if tile == "Escaparate")
            if display_count < 2:
                return False
                
        elif block_type == "industrial":
            # Industrial blocks need large continuous sections
            metal_sections = PathValidator._find_continuous_sections(
                tilemap, {"Metal", "Hormigón"}
            )
            if not any(len(section) >= 4 for section in metal_sections):
                return False
                
        return True

    @staticmethod
    def _find_vertical_path(tilemap: List[List[Optional[str]]], r: int, c: int, 
                          visited: List[List[bool]]) -> bool:
        """Find continuous vertical path using DFS."""
        if r == len(tilemap) - 1:
            return True
            
        visited[r][c] = True
        height, width = len(tilemap), len(tilemap[0])
        
        # Try moving down, down-left, or down-right
        for dr, dc in [(1, 0), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < height and 0 <= nc < width and
                not visited[nr][nc] and tilemap[nr][nc] == "Agua"):
                if PathValidator._find_vertical_path(tilemap, nr, nc, visited):
                    return True
                    
        return False

    @staticmethod
    def _find_horizontal_path(tilemap: List[List[Optional[str]]], r: int, c: int,
                            visited: List[List[bool]], valid_tiles: Set[str]) -> bool:
        """Find continuous horizontal path using DFS."""
        if c == len(tilemap[0]) - 1:
            return True
            
        visited[r][c] = True
        height, width = len(tilemap), len(tilemap[0])
        
        # Try moving right, up-right, or down-right
        for dr, dc in [(0, 1), (-1, 1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < height and 0 <= nc < width and
                not visited[nr][nc] and tilemap[nr][nc] in valid_tiles):
                if PathValidator._find_horizontal_path(tilemap, nr, nc, visited, valid_tiles):
                    return True
                    
        return False

    @staticmethod
    def _find_continuous_sections(tilemap: List[List[Optional[str]]], 
                                valid_tiles: Set[str]) -> List[Set[Tuple[int, int]]]:
        """Find continuous sections of specified tiles using flood fill."""
        height, width = len(tilemap), len(tilemap[0])
        visited = [[False]*width for _ in range(height)]
        sections = []
        
        def flood_fill(r: int, c: int, section: Set[Tuple[int, int]]) -> None:
            if (not (0 <= r < height and 0 <= c < width) or
                visited[r][c] or tilemap[r][c] not in valid_tiles):
                return
                
            visited[r][c] = True
            section.add((r, c))
            
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                flood_fill(r + dr, c + dc, section)
        
        for r in range(height):
            for c in range(width):
                if not visited[r][c] and tilemap[r][c] in valid_tiles:
                    section = set()
                    flood_fill(r, c, section)
                    sections.append(section)
                    
        return sections