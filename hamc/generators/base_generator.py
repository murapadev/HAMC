from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import logging
from ..core.cell import Cell
from ..core.generator_state import GeneratorState, GeneratorStatus
from ..core.compatibility_cache import CompatibilityCache
from ..core.backtrack_manager import BacktrackManager

class BaseGenerator(ABC):
    """Base class for all wave function collapse generators."""
    
    def __init__(self, width: int, height: int, name: str):
        self.width = width
        self.height = height
        self.cells: List[List[Cell]] = []
        self.state = GeneratorState(name)
        self.cache = CompatibilityCache()
        self.backtrack_manager = BacktrackManager()
        self.logger = logging.getLogger(name)
        
        # Configure logger if not already configured
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def collapse(self) -> bool:
        """Template method for wave function collapse."""
        self.state.update_status(GeneratorStatus.IN_PROGRESS)
        
        try:
            if not self.initialize():
                self.state.update_status(GeneratorStatus.FAILED, "Initialization failed")
                return False

            if not self._run_collapse():
                self.state.update_status(GeneratorStatus.FAILED, "Collapse failed")
                return False

            if not self.validate():
                self.state.update_status(GeneratorStatus.FAILED, "Validation failed")
                return False

            self.state.update_status(GeneratorStatus.COMPLETED)
            return True

        except Exception as e:
            self.state.logger.error(f"Error during collapse: {str(e)}")
            self.state.update_status(GeneratorStatus.FAILED, str(e))
            return False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the generator."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate the final state."""
        pass

    def _run_collapse(self) -> bool:
        """Execute wave function collapse with improved backtracking."""
        self.backtrack_manager.clear()
        
        while True:
            target = self._find_min_entropy_cell()
            
            if not target:
                return True  # All cells collapsed successfully
            
            r, c = target
            current_entropy = self.cells[r][c].entropy()
            
            if current_entropy == float('inf'):
                # No valid options remain, need to backtrack
                point = self.backtrack_manager.pop()
                if not point:
                    return False  # No more backtrack points
                    
                self.state.update_status(GeneratorStatus.BACKTRACKING)
                self._restore(point.state_snapshot)
                continue
            
            # Take snapshot before collapsing
            snapshot = self._snapshot()
            self.backtrack_manager.push(
                self.__class__.__name__.lower(),
                (r, c),
                snapshot,
                current_entropy,
                {"failed_values": self.backtrack_manager.get_failed_values(
                    self.__class__.__name__.lower(), (r, c)
                )}
            )
            
            # Collapse and propagate
            self.state.increment_stat("collapses")
            value = self.cells[r][c].collapse()
            
            if not value or not self.propagate(r, c):
                # Record failure and try next backtrack point
                self.backtrack_manager.record_failure(
                    self.__class__.__name__.lower(),
                    (r, c),
                    value or "propagation_failure"
                )
                continue
            
            # Update metrics
            self.state.update_entropy(current_entropy)
            
        return True

    def _find_min_entropy_cell(self) -> Optional[Tuple[int, int]]:
        """Find cell with minimum entropy, avoiding previously failed values."""
        min_entropy = float('inf')
        target = None
        
        for r in range(self.height):
            for c in range(self.width):
                cell = self.cells[r][c]
                if cell.collapsed_value is not None:
                    continue
                    
                # Filter out failed values for this position
                failed_values = self.backtrack_manager.get_failed_values(
                    self.__class__.__name__.lower(),
                    (r, c)
                )
                
                if failed_values:
                    # Create new possibilities excluding failed values
                    valid_possible = {k: v for k, v in cell.possible.items()
                                   if k not in failed_values}
                    if valid_possible:
                        cell.possible = valid_possible
                
                entropy = cell.entropy()
                if 0 < entropy < min_entropy:
                    min_entropy = entropy
                    target = (r, c)
        
        return target

    @abstractmethod
    def propagate(self, row: int, col: int) -> bool:
        """Propagate constraints from a cell."""
        pass

    def get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Get valid neighbors for a cell."""
        neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbors.append((nr, nc))
        return neighbors

    def _snapshot(self) -> List[List[Tuple[Dict[str, float], str]]]:
        """Take a snapshot of current state for backtracking."""
        snap = []
        for r in range(self.height):
            row_snap = []
            for c in range(self.width):
                cell = self.cells[r][c]
                row_snap.append((dict(cell.possible), cell.collapsed_value))
            snap.append(row_snap)
        return snap

    def _restore(self, snapshot: List[List[Tuple[Dict[str, float], str]]]) -> None:
        """Restore state from a snapshot."""
        for r in range(self.height):
            for c in range(self.width):
                possible, collapsed = snapshot[r][c]
                self.cells[r][c].possible = dict(possible)
                self.cells[r][c].collapsed_value = collapsed
