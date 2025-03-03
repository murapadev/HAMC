from typing import List, Dict, Set, Optional, Tuple
from collections import deque
import time
from ..core.cell import Cell
from ..config.tile_config import TileConfig
from ..core.validator import PathValidator
from .base_generator import BaseGenerator

class LocalGenerator(BaseGenerator):
    """Generator for local level (tiles) with path validation."""
    
    SPECIAL_BLOCK_TYPES = {
        "rio", "carretera", "oasis", "residencial", "comercial", "industrial"
    }
    
    def __init__(self, block_type: str, size: int, 
                 neighbors: Optional[Dict[str, str]] = None):
        super().__init__(size, size, name=f"LocalGenerator_{block_type}")
        self.block_type = block_type
        self.neighbors = neighbors or {}
        self.validator = PathValidator()
        self.initialize()

    def initialize(self) -> bool:
        # Get possible tiles for this block type
        tiles = TileConfig.get_tiles_for_block(self.block_type)
        if not tiles:
            self.logger.error(f"No valid tiles found for block type {self.block_type}")
            return False

        # Initialize cells with all possible tiles
        self.cells = [[Cell(dict(tiles)) for _ in range(self.width)] 
                     for _ in range(self.height)]
        
        # Pre-process cells for special block types
        if self.block_type in self.SPECIAL_BLOCK_TYPES:
            return self._init_special_block()
        
        # Apply overlapping constraints based on neighbors
        self._apply_border_constraints()
        
        # Collapse strategic initial cells
        self._collapse_strategic_cells()
        return True

    def _init_special_block(self) -> bool:
        """Initialize special block types with specific constraints."""
        if self.block_type == "rio":
            self._init_river()
        elif self.block_type == "carretera":
            self._init_road()
        elif self.block_type == "oasis":
            self._init_oasis()
        elif self.block_type in {"residencial", "comercial", "industrial"}:
            self._init_building()
        return True

    def _init_river(self) -> None:
        """Initialize river with water in center column."""
        center = self.width // 2
        
        # Force water in center column if connecting to other rivers
        if self.neighbors.get('top') == 'rio':
            self.cells[0][center].possible = {"Agua": 1.0}
        if self.neighbors.get('bottom') == 'rio':
            self.cells[-1][center].possible = {"Agua": 1.0}
            
        # Increase water probability in center column
        for row in range(self.height):
            if not self.cells[row][center].collapsed_value:
                self.cells[row][center].possible = {"Agua": 0.8, "Hierba": 0.2}

    def _init_road(self) -> None:
        """Initialize road with asphalt in center row."""
        center = self.height // 2
        
        # Force road tiles if connecting to other roads
        if self.neighbors.get('left') == 'carretera':
            self.cells[center][0].possible = {"Asfalto": 0.8, "Linea": 0.2}
        if self.neighbors.get('right') == 'carretera':
            self.cells[center][-1].possible = {"Asfalto": 0.8, "Linea": 0.2}
            
        # Increase road probability in center row
        for col in range(self.width):
            if not self.cells[center][col].collapsed_value:
                self.cells[center][col].possible = {"Asfalto": 0.7, "Linea": 0.3}

    def _init_oasis(self) -> None:
        """Initialize oasis with water probability distribution."""
        center_r = self.height // 2
        center_c = self.width // 2
        
        # Create water probability gradient from center
        for r in range(self.height):
            for c in range(self.width):
                # Calculate distance from center (normalized between 0 and 1)
                dist = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5
                max_dist = ((self.height/2) ** 2 + (self.width/2) ** 2) ** 0.5
                norm_dist = dist / max_dist
                
                # Ensure no water at edges
                if r == 0 or r == self.height-1 or c == 0 or c == self.width-1:
                    # Edges should be sand
                    self.cells[r][c].possible = {"Arena": 0.9, "Hierba": 0.1}
                else:
                    # Water probability decreases with distance from center
                    water_prob = max(0, 0.9 * (1 - norm_dist**2))
                    sand_prob = 0.7 * (1 - water_prob)
                    hierba_prob = 1.0 - water_prob - sand_prob
                    
                    self.cells[r][c].possible = {
                        "Agua": water_prob,
                        "Arena": sand_prob,
                        "Hierba": hierba_prob
                    }
        
        # Force water in center
        center_radius = min(1, self.height // 5)
        for r in range(center_r - center_radius, center_r + center_radius + 1):
            for c in range(center_c - center_radius, center_c + center_radius + 1):
                if 0 <= r < self.height and 0 <= c < self.width:
                    dist_to_center = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5
                    if dist_to_center <= center_radius:
                        self.cells[r][c].possible = {"Agua": 1.0}

    def _init_building(self) -> None:
        """Initialize building blocks with structure constraints."""
        # Set border cells to walls
        for r in range(self.height):
            self.cells[r][0].possible = {"Pared": 1.0}
            self.cells[r][-1].possible = {"Pared": 1.0}
        for c in range(self.width):
            self.cells[0][c].possible = {"Pared": 1.0}
            self.cells[-1][c].possible = {"Pared": 1.0}
            
        if self.block_type == "residencial":
            # Add door in middle of one wall
            center = self.width // 2
            self.cells[-1][center].possible = {"Puerta": 1.0}
            
        elif self.block_type == "comercial":
            # Add display windows on ground floor
            for c in range(1, self.width-1):
                self.cells[-1][c].possible = {"Escaparate": 0.7, "Ventana": 0.3}
                
        elif self.block_type == "industrial":
            # Large sections of industrial materials
            for r in range(1, self.height-1):
                for c in range(1, self.width-1):
                    self.cells[r][c].possible = {"Metal": 0.6, "Hormigón": 0.4}

    def _collapse_strategic_cells(self) -> None:
        """Collapse strategic cells based on block type."""
        if self.block_type in ["rio", "carretera"]:
            # Already handled in init methods
            pass
        else:
            # Collapse center cell for other blocks
            center_r = self.height // 2
            center_c = self.width // 2
            self.cells[center_r][center_c].collapse()

    def _apply_border_constraints(self) -> None:
        """Apply constraints based on neighboring blocks."""
        for border in ['top', 'bottom', 'left', 'right']:
            if self.neighbors.get(border):
                self._apply_border_constraint(border)

    def _apply_border_constraint(self, border: str) -> None:
        """Apply constraints based on neighboring blocks."""
        neighbor_type = self.neighbors[border]
        neighbor_tiles = TileConfig.get_tiles_for_block(neighbor_type)
        
        if not neighbor_tiles:
            return

        # Get compatible tiles that can connect with neighbor
        compatible_tiles = set()
        for neighbor_tile in neighbor_tiles.keys():
            for own_tile in TileConfig.get_tiles_for_block(self.block_type).keys():
                if TileConfig.are_compatible(neighbor_tile, own_tile):
                    compatible_tiles.add(own_tile)

        # Apply constraints to border cells
        if border == 'top':
            for c in range(self.width):
                self._filter_cell_possibilities(0, c, compatible_tiles)
        elif border == 'bottom':
            for c in range(self.width):
                self._filter_cell_possibilities(self.height-1, c, compatible_tiles)
        elif border == 'left':
            for r in range(self.height):
                self._filter_cell_possibilities(r, 0, compatible_tiles)
        elif border == 'right':
            for r in range(self.height):
                self._filter_cell_possibilities(r, self.width-1, compatible_tiles)

    def _filter_cell_possibilities(self, row: int, col: int, valid_tiles: Set[str]) -> None:
        """Filter cell possibilities to only include valid tiles."""
        cell = self.cells[row][col]
        new_possible = {}
        for tile, prob in cell.possible.items():
            if tile in valid_tiles:
                new_possible[tile] = prob
        if new_possible:
            cell.possible = new_possible

    def propagate(self, row: int, col: int) -> bool:
        """Propagate tile constraints with proper handling of special tiles."""
        queue = [(row, col)]
        processed = set()
        
        while queue:
            cr, cc = queue.pop(0)
            if (cr, cc) in processed:
                continue
                
            processed.add((cr, cc))
            current = self.cells[cr][cc].collapsed_value
            if not current:
                continue
                
            for nr, nc in self.get_neighbors(cr, cc):
                neighbor = self.cells[nr][nc]
                if neighbor.collapsed_value is not None:
                    continue
                    
                # Special handling for different block types
                if self.block_type == "rio" and current == "Agua":
                    # Vertical neighbor gets high probability of water
                    if abs(nr - cr) == 1:  # Vertical neighbor
                        new_possible = {"Agua": 0.8, "Hierba": 0.2}
                        if new_possible != neighbor.possible:
                            neighbor.possible = new_possible
                            queue.append((nr, nc))
                        continue
                        
                elif self.block_type == "carretera" and current in {"Asfalto", "Linea"}:
                    # Horizontal neighbor gets high probability of matching tile
                    if abs(nc - cc) == 1:  # Horizontal neighbor
                        new_possible = {current: 0.8, "Asfalto" if current == "Linea" else "Linea": 0.2}
                        if new_possible != neighbor.possible:
                            neighbor.possible = new_possible
                            queue.append((nr, nc))
                        continue
                
                elif self.block_type == "oasis":
                    # Special handling for oasis tiles
                    if current == "Agua":
                        # Water should transition to sand or vegetation
                        if neighbor.possible.get("Agua", 0) < 0.5:
                            new_possible = {"Arena": 0.7, "Hierba": 0.3}
                            if new_possible != neighbor.possible:
                                neighbor.possible = new_possible
                                queue.append((nr, nc))
                            continue
                    elif current == "Arena":
                        # Sand can be next to anything
                        continue
                
                # For all other cases, use normal compatibility rules
                new_possible = {}
                for tile, prob in neighbor.possible.items():
                    if TileConfig.are_compatible(current, tile):
                        new_possible[tile] = prob
                        
                # If no compatible tiles, try default compatibility
                if not new_possible:
                    compatible_tiles = TileConfig.COMPATIBILITY.get(current, set())
                    for tile in compatible_tiles:
                        new_possible[tile] = 1.0 / len(compatible_tiles)
                        
                # If still no compatible tiles, propagation failed
                if not new_possible:
                    return False
                    
                # Update neighbor's possibilities if they changed
                if new_possible != neighbor.possible:
                    neighbor.possible = new_possible
                    queue.append((nr, nc))
                    
        return True

    def collapse(self) -> bool:
        """Run wave function collapse algorithm for local level."""
        stack = []
        attempts = 0
        max_attempts = 500  # Prevent infinite loops
        start_time = time.time()
        timeout = 5.0  # Timeout after 5 seconds
        
        while attempts < max_attempts and (time.time() - start_time) < timeout:
            attempts += 1
            if attempts % 100 == 0:
                self.logger.warning(f"Many attempts needed: {attempts}")
            
            # Find cell with minimum entropy
            min_entropy = float('inf')
            target = None
            
            for r in range(self.height):
                for c in range(self.width):
                    entropy = self.cells[r][c].entropy()
                    if 0 < entropy < min_entropy:
                        min_entropy = entropy
                        target = (r, c)
            
            if not target:
                # Before returning success, validate paths if needed
                if self.validate_paths():
                    self.logger.info(f"Collapse successful after {attempts} attempts")
                    return True
                if not stack:
                    self.logger.error("Failed to find valid path configuration")
                    return False
                self._restore(stack.pop())
                continue
            
            r, c = target
            if min_entropy == float('inf'):
                if not stack:
                    self.logger.error("No valid options and no backtrack points")
                    return False
                self._restore(stack.pop())
                continue
            
            stack.append(self._snapshot())
            self.cells[r][c].collapse()
            
            if not self.propagate(r, c):
                self.logger.debug(f"Propagation failed at ({r},{c}), backtracking...")
                self._restore(stack.pop())
                continue
        
        if attempts >= max_attempts:
            self.logger.error(f"Failed to complete after {max_attempts} attempts")
            return False
            
        if (time.time() - start_time) >= timeout:
            self.logger.error(f"Timeout after {timeout} seconds")
            return False
            
        return True

    def validate_paths(self) -> bool:
        """Validate path requirements for special blocks."""
        tilemap = [[cell.collapsed_value for cell in row] for row in self.cells]
        
        if self.block_type == "rio":
            return self.validator.validate_river_path(tilemap, self.neighbors)
            
        elif self.block_type == "carretera":
            return self.validator.validate_road_path(tilemap, self.neighbors)
            
        elif self.block_type == "oasis":
            return self.validator.validate_oasis(tilemap)
            
        elif self.block_type in {"residencial", "comercial", "industrial"}:
            return self.validator.validate_building(tilemap, self.block_type)
            
        return True

    def validate(self) -> bool:
        """Validate final state of local tiles."""
        # Check all cells are collapsed
        if not all(cell.collapsed_value for row in self.cells for cell in row):
            return False

        # Validate tile distribution
        valid_tiles = set(TileConfig.get_tiles_for_block(self.block_type).keys())
        for row in self.cells:
            for cell in row:
                if cell.collapsed_value not in valid_tiles:
                    return False

        # Validate special path requirements
        return self.validate_paths()
