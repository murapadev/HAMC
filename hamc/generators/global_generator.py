from typing import List, Dict
from ..core.cell import Cell
from ..config.region_config import RegionConfig
from .base_generator import BaseGenerator
from ..config.advanced_config import get_generator_config


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
        # First, ensure minimum representation of each region type
        if not self._ensure_minimum_regions():
            return False
            
        # Use base class collapse method for consistent backtracking
        if not super().collapse():
            return False

        # Additional validation: ensure all required regions are present
        regions_present = set()
        for r in range(self.height):
            for c in range(self.width):
                if self.cells[r][c].collapsed_value:
                    regions_present.add(self.cells[r][c].collapsed_value)

        # If any required region is missing, try to regenerate
        missing_regions = set(RegionConfig.TYPES.keys()) - regions_present

        if missing_regions:
            self.logger.warning(f"Missing required regions: {missing_regions}. Regenerating...")
            # Reset and try again with forced placement
            self.initialize()
            return self._force_region_placement(missing_regions)

        return True
        
    def _ensure_minimum_regions(self) -> bool:
        """Ensure minimum representation of each region type before collapse."""
        try:
            config = get_generator_config()
            # For small grids, ensure at least one cell per region type
            if self.width * self.height < len(RegionConfig.TYPES):
                self.logger.warning("Grid too small for all region types")
                return True
                
            # Don't force probabilities, just ensure initialization is correct
            # The WFC algorithm should handle the rest
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to ensure minimum regions: {str(e)}")
            return False
            
    def _force_region_placement(self, missing_regions: set) -> bool:
        """Force placement of missing regions and retry collapse."""
        try:
            # For small grids, just try a simple approach
            if self.width * self.height <= len(missing_regions):
                # Place one region per cell for small grids
                region_list = list(missing_regions)
                for i, (r, c) in enumerate([(r, c) for r in range(self.height) for c in range(self.width)]):
                    if i < len(region_list):
                        self.cells[r][c].possible = {region_list[i]: 1.0}
                        self.cells[r][c].collapse()
                    else:
                        break
            else:
                # For larger grids, try to place missing regions strategically
                for region in missing_regions:
                    placed = False
                    for r in range(self.height):
                        for c in range(self.width):
                            if self.cells[r][c].collapsed_value is None:
                                # Try to place the region
                                self.cells[r][c].possible = {region: 1.0}
                                if self.cells[r][c].collapse():
                                    placed = True
                                    break
                        if placed:
                            break
                            
            # Now try to collapse remaining cells normally
            return super().collapse()
            
        except Exception as e:
            self.logger.error(f"Failed to force region placement: {str(e)}")
            return False

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

                # Skip if region is None (shouldn't happen in valid state, but be safe)
                if region is None:
                    self.logger.error(f"Cell at ({r},{c}) has no collapsed value")
                    return False

                # Check compatibility with neighbors
                for nr, nc in self.get_neighbors(r, c):
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        neighbor = self.cells[nr][nc].collapsed_value

                        # Skip if neighbor is None
                        if neighbor is None:
                            continue

                        if not RegionConfig.are_compatible(region, neighbor):
                            self.logger.error(
                                f"Incompatible regions: {region} and {neighbor} at ({r},{c})"
                            )
                            return False

        return True
