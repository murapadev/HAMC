from typing import List, Dict
from ..core.cell import Cell
from ..config.region_config import RegionConfig
from .base_generator import BaseGenerator

class GlobalGenerator(BaseGenerator):
    """Generator for the global level (regions)."""
    
    def __init__(self, width: int, height: int):
        super().__init__(width, height, name="GlobalGenerator")

    def initialize(self) -> bool:
        """Initialize cells with all possible regions.
        
        Returns:
            bool: True if initialization was successful
        """
        # Get probabilities for all regions
        probs = RegionConfig.get_probabilities()
        
        # Initialize cells with a copy of the probabilities
        self.cells = [[Cell(dict(probs)) for _ in range(self.width)] 
                     for _ in range(self.height)]
        
        return True

    def collapse(self) -> bool:
        """Run wave function collapse algorithm with region type constraints.
        
        Ensures that all required region types appear in the final map.
        """
        stack = []
        while True:
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
                # All cells collapsed, check if all required regions are present
                regions_present = set()
                for r in range(self.height):
                    for c in range(self.width):
                        if self.cells[r][c].collapsed_value:
                            regions_present.add(self.cells[r][c].collapsed_value)
                
                # If any required region is missing, try to fix by backtracking
                # and forcing a cell to be the missing region
                missing_regions = set(RegionConfig.TYPES.keys()) - regions_present
                
                if missing_regions:
                    self.logger.warning(f"Missing required regions: {missing_regions}")
                    
                    # If we have no backtrack points, we can't fix the map
                    if not stack:
                        self.logger.error("Cannot fix missing regions, no backtrack points")
                        return False
                    
                    # Restore to a previous state
                    self._restore(stack.pop())
                    continue
                
                return True  # All cells collapsed and all regions present
            
            r, c = target
            if min_entropy == float('inf'):
                if not stack:
                    return False
                self._restore(stack.pop())
                continue
            
            stack.append(self._snapshot())
            self.cells[r][c].collapse()
            
            if not self.propagate(r, c):
                continue

    def propagate(self, row: int, col: int) -> bool:
        queue = [(row, col)]
        while queue:
            cr, cc = queue.pop(0)
            current = self.cells[cr][cc].collapsed_value
            if not current:
                continue

            for nr, nc in self.get_neighbors(cr, cc):
                neighbor = self.cells[nr][nc]
                if neighbor.collapsed_value is not None:
                    continue

                # Filter possible values based on compatibility
                new_possible = {}
                for reg, prob in neighbor.possible.items():
                    if RegionConfig.are_compatible(current, reg):
                        new_possible[reg] = prob

                if not new_possible:
                    return False

                if len(new_possible) < len(neighbor.possible):
                    neighbor.possible = new_possible
                    queue.append((nr, nc))

        return True

    def validate(self) -> bool:
        """Validate final state of global regions.
        
        Returns:
            bool: True if the map state is valid
        """
        # Check all cells are collapsed
        if not all(cell.collapsed_value for row in self.cells for cell in row):
            self.logger.error("Not all cells are collapsed")
            return False

        # Count regions
        region_counts = {}
        total_cells = self.width * self.height
        
        for row in self.cells:
            for cell in row:
                region = cell.collapsed_value
                region_counts[region] = region_counts.get(region, 0) + 1

        # Check if required regions are present
        for region in RegionConfig.TYPES:
            if region not in region_counts:
                self.logger.error(f"Missing required region: {region}")
                return False

        # Validate region adjacency
        for r in range(self.height):
            for c in range(self.width):
                region = self.cells[r][c].collapsed_value
                
                # Check compatibility with neighbors
                for nr, nc in self.get_neighbors(r, c):
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        neighbor = self.cells[nr][nc].collapsed_value
                        if not RegionConfig.are_compatible(region, neighbor):
                            self.logger.error(
                                f"Incompatible regions: {region} and {neighbor} at ({r},{c})"
                            )
                            return False

        return True
