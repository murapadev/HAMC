from typing import List, Dict, Tuple
import logging
from ..core.cell import Cell
from ..config.block_config import BlockConfig
from .base_generator import BaseGenerator
from .global_generator import GlobalGenerator

class IntermediateGenerator(BaseGenerator):
    """Generator for intermediate level (blocks)."""
    
    def __init__(self, global_gen: GlobalGenerator, subgrid_size: int = 3):
        self.subgrid_size = subgrid_size
        self.global_gen = global_gen  # Set this before calling super()
        super().__init__(
            global_gen.width * subgrid_size,
            global_gen.height * subgrid_size,
            name="IntermediateGenerator"
        )
        self.margin = 1
        self.cells = [[Cell({"Empty": 1.0}) for _ in range(self.width)] 
                     for _ in range(self.height)]
        self.initialize()

    def initialize(self) -> bool:
        """Initialize cells based on global regions and transitions."""
        try:
            # Initialize cells with default blocks for each region
            for gr in range(self.global_gen.height):
                for gc in range(self.global_gen.width):
                    region = self.global_gen.cells[gr][gc].collapsed_value
                    if not region:
                        self.logger.error(f"Global cell ({gr},{gc}) not collapsed")
                        return False
                        
                    blocks = BlockConfig.get_blocks_for_region(region)
                    if not blocks:
                        self.logger.error(f"No blocks found for region {region}")
                        return False
                        
                    # Calculate subgrid boundaries
                    start_row = gr * self.subgrid_size
                    start_col = gc * self.subgrid_size
                    
                    # Initialize cells in subgrid with possible blocks
                    for r in range(start_row, start_row + self.subgrid_size):
                        for c in range(start_col, start_col + self.subgrid_size):
                            if 0 <= r < self.height and 0 <= c < self.width:
                                self.cells[r][c] = Cell(dict(blocks))
            
            # Handle transitions between different regions
            for gr in range(self.global_gen.height):
                for gc in range(self.global_gen.width):
                    region = self.global_gen.cells[gr][gc].collapsed_value
                    if region:
                        self._handle_transitions(gr, gc)
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False

    def _handle_transitions(self, gr: int, gc: int) -> None:
        """Handle transition zones between different regions."""
        region = self.global_gen.cells[gr][gc].collapsed_value
        if not region:
            return
            
        # Check all neighboring regions for transitions
        for dr, dc in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            ngr, ngc = gr + dr, gc + dc
            if not (0 <= ngr < self.global_gen.height and 
                   0 <= ngc < self.global_gen.width):
                continue

            neighbor_region = self.global_gen.cells[ngr][ngc].collapsed_value
            if not neighbor_region or neighbor_region == region:
                continue
                
            # Try transitions in both directions
            trans_block, weight = BlockConfig.get_transition(region, neighbor_region)
            if not trans_block or weight <= 0:
                trans_block, weight = BlockConfig.get_transition(neighbor_region, region)
                
            if trans_block and weight > 0:
                self._mark_transition_cells(gr, gc, dr, dc, trans_block, weight)

    def _mark_transition_cells(self, gr: int, gc: int, dr: int, dc: int, 
                             trans_block: str, weight: float) -> None:
        """Mark cells in transition zone with transition block."""
        start_row = gr * self.subgrid_size
        start_col = gc * self.subgrid_size
        
        # Calculate transition zone boundaries
        if dc == 1:  # Right border
            rows = range(start_row, start_row + self.subgrid_size)
            cols = [start_col + self.subgrid_size - self.margin]
        elif dc == -1:  # Left border
            rows = range(start_row, start_row + self.subgrid_size)
            cols = [start_col]
        elif dr == 1:  # Bottom border
            rows = [start_row + self.subgrid_size - self.margin]
            cols = range(start_col, start_col + self.subgrid_size)
        else:  # Top border
            rows = [start_row]
            cols = range(start_col, start_col + self.subgrid_size)
            
        # Set transition blocks
        for r in rows:
            for c in cols:
                if 0 <= r < self.height and 0 <= c < self.width:
                    self.cells[r][c] = Cell({trans_block: weight})

    def validate(self) -> bool:
        """Validate final state of intermediate blocks."""
        try:
            # Check all cells are collapsed
            for r in range(self.height):
                for c in range(self.width):
                    if self.cells[r][c].collapsed_value is None:
                        self.logger.error(f"Cell at ({r},{c}) is not collapsed")
                        return False

            # Validate block placements and transitions
            for r in range(self.height):
                for c in range(self.width):
                    current_block = self.cells[r][c].collapsed_value
                    gr, gc = r // self.subgrid_size, c // self.subgrid_size
                    region = self.global_gen.cells[gr][gc].collapsed_value
                    
                    # Get allowed blocks for this region
                    valid_blocks = set(BlockConfig.get_blocks_for_region(region).keys())
                    
                    # Check if it's a valid transition block
                    is_transition = False
                    for dr, dc in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                        ngr, ngc = gr + dr, gc + dc
                        if (0 <= ngr < self.global_gen.height and 
                            0 <= ngc < self.global_gen.width):
                            neighbor_region = self.global_gen.cells[ngr][ngc].collapsed_value
                            if neighbor_region and neighbor_region != region:
                                trans1, _ = BlockConfig.get_transition(region, neighbor_region)
                                trans2, _ = BlockConfig.get_transition(neighbor_region, region)
                                if current_block in {trans1, trans2}:
                                    is_transition = True
                                    break
                    
                    if current_block not in valid_blocks and not is_transition:
                        self.logger.error(
                            f"Invalid block {current_block} for region {region} at ({r},{c})"
                        )
                        return False
                        
                    # Check compatibility with neighbors
                    for nr, nc in self.get_neighbors(r, c):
                        if nr >= self.height or nc >= self.width:
                            continue
                            
                        neighbor_block = self.cells[nr][nc].collapsed_value
                        if (neighbor_block and 
                            not BlockConfig.are_compatible(current_block, neighbor_block)):
                            self.logger.error(
                                f"Incompatible blocks {current_block} and {neighbor_block} at ({r},{c})"
                            )
                            return False
                            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return False

    def propagate(self, row: int, col: int) -> bool:
        """Propagate block constraints."""
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
                if nr >= self.height or nc >= self.width:
                    continue
                    
                neighbor = self.cells[nr][nc]
                if neighbor.collapsed_value is not None:
                    continue

                # Filter possible values based on compatibility
                new_possible = {}
                for block, prob in neighbor.possible.items():
                    if BlockConfig.are_compatible(current, block):
                        new_possible[block] = prob

                if not new_possible:
                    return False

                if new_possible != neighbor.possible:
                    neighbor.possible = new_possible
                    queue.append((nr, nc))

        return True

    def collapse(self) -> bool:
        """Run wave function collapse algorithm for intermediate level."""
        # Use base class collapse method for consistent backtracking
        return super().collapse()
