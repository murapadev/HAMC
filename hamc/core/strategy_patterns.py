# hamc/core/strategy_patterns.py
"""
Strategy Patterns for HAMC
Implements strategy patterns for different generation algorithms and behaviors.
"""

from typing import Dict, List, Any, Optional, Type, TypeVar, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import logging
import math
from dataclasses import dataclass
from enum import Enum

from .cell import Cell
from .generator_state import GeneratorState, GeneratorStatus

T = TypeVar('T')

class AlgorithmType(Enum):
    """Types of generation algorithms."""
    BASIC_WFC = "basic_wfc"
    PRIORITY_WFC = "priority_wfc"
    BACKTRACKING_WFC = "backtracking_wfc"
    PARALLEL_WFC = "parallel_wfc"
    ADAPTIVE_WFC = "adaptive_wfc"


class SelectionStrategy(Enum):
    """Strategies for selecting cells to collapse."""
    RANDOM = "random"
    MIN_ENTROPY = "min_entropy"
    MAX_ENTROPY = "max_entropy"
    PRIORITY_QUEUE = "priority_queue"
    CUSTOM = "custom"


@dataclass
class AlgorithmConfig:
    """Configuration for algorithm execution."""
    algorithm_type: AlgorithmType
    selection_strategy: SelectionStrategy
    max_iterations: int = 10000
    backtrack_limit: int = 1000
    custom_params: Optional[Dict[str, Any]] = None


class StrategyError(Exception):
    """Exception raised when strategy execution fails."""
    pass


@runtime_checkable
class CollapseStrategy(Protocol):
    """Protocol for cell collapse strategies."""
    
    def select_cell(self, cells: List[List[Cell]], state: GeneratorState) -> Optional[tuple[int, int]]:
        """Select a cell to collapse."""
        ...
    
    def should_backtrack(self, state: GeneratorState) -> bool:
        """Determine if backtracking is needed."""
        ...


@runtime_checkable
class PropagationStrategy(Protocol):
    """Protocol for constraint propagation strategies."""
    
    def propagate_constraints(self, cells: List[List[Cell]], x: int, y: int, 
                            collapsed_value: Any, state: GeneratorState) -> bool:
        """Propagate constraints after cell collapse."""
        ...


class BaseStrategy(ABC):
    """Base class for all strategies."""
    
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the strategy."""
        pass


class RandomSelectionStrategy(BaseStrategy):
    """Random cell selection strategy."""
    
    def select_cell(self, cells: List[List[Cell]], state: GeneratorState) -> Optional[tuple[int, int]]:
        """Select a random uncollapsed cell."""
        import random
        
        uncollapsed = []
        for i in range(len(cells)):
            for j in range(len(cells[0])):
                if cells[i][j].collapsed_value is None:  # Check if not collapsed
                    uncollapsed.append((i, j))
        
        if not uncollapsed:
            return None
        
        return random.choice(uncollapsed)
    
    def should_backtrack(self, state: GeneratorState) -> bool:
        """Never backtrack for random strategy."""
        return False
    
    def execute(self, *args, **kwargs):
        """Execute random selection."""
        # This method is not used in the same way as other strategies
        pass


class MinEntropySelectionStrategy(BaseStrategy):
    """Minimum entropy cell selection strategy."""
    
    def select_cell(self, cells: List[List[Cell]], state: GeneratorState) -> Optional[tuple[int, int]]:
        """Select cell with minimum entropy."""
        min_entropy = float('inf')
        selected_cell = None
        
        for i in range(len(cells)):
            for j in range(len(cells[0])):
                if cells[i][j].collapsed_value is None:  # Check if not collapsed
                    cell = cells[i][j]
                    # Calculate entropy manually
                    total_weight = sum(cell.possible.values())
                    if total_weight > 0:
                        entropy = 0
                        for p in cell.possible.values():
                            if p > 0:
                                prob = p / total_weight
                                entropy -= prob * math.log2(prob)  # Correct entropy calculation
                        if entropy < min_entropy:
                            min_entropy = entropy
                            selected_cell = (i, j)
        
        return selected_cell
    
    def should_backtrack(self, state: GeneratorState) -> bool:
        """Backtrack if we have contradictions."""
        return state.status == GeneratorStatus.FAILED
    
    def execute(self, *args, **kwargs):
        """Execute minimum entropy selection."""
        pass


class PriorityQueueSelectionStrategy(BaseStrategy):
    """Priority queue-based cell selection strategy."""
    
    def select_cell(self, cells: List[List[Cell]], state: GeneratorState) -> Optional[tuple[int, int]]:
        """Select cell using priority queue (min entropy first)."""
        import heapq
        
        candidates = []
        for i in range(len(cells)):
            for j in range(len(cells[0])):
                if cells[i][j].collapsed_value is None:  # Check if not collapsed
                    cell = cells[i][j]
                    # Calculate entropy manually
                    total_weight = sum(cell.possible.values())
                    if total_weight > 0:
                        entropy = sum(-p/total_weight * math.log2(p/total_weight) 
                                    for p in cell.possible.values() if p > 0)
                        heapq.heappush(candidates, (entropy, (i, j)))
        
        if not candidates:
            return None
        
        # Return cell with minimum entropy
        _, selected_cell = heapq.heappop(candidates)
        return selected_cell
    
    def should_backtrack(self, state: GeneratorState) -> bool:
        """Backtrack based on state analysis."""
        return state.status == GeneratorStatus.FAILED  # Fixed: STALLED doesn't exist
    
    def execute(self, *args, **kwargs):
        """Execute priority queue selection."""
        pass


class BasicPropagationStrategy(BaseStrategy):
    """Basic constraint propagation strategy."""
    
    def propagate_constraints(self, cells: List[List[Cell]], x: int, y: int, 
                            collapsed_value: Any, state: GeneratorState) -> bool:
        """Propagate constraints to neighboring cells."""
        # Basic implementation - constrain adjacent cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(cells) and 0 <= ny < len(cells[0]):
                neighbor = cells[nx][ny]
                if neighbor.collapsed_value is None:  # Check if not collapsed
                    # Remove incompatible states from neighbor
                    # This is a simplified version
                    compatible_states = self._get_compatible_states(
                        collapsed_value, list(neighbor.possible.keys())  # Use possible.keys()
                    )
                    
                    if not compatible_states:
                        # Contradiction detected
                        return False
                    
                    neighbor.constrain(set(compatible_states))  # Convert to set
        
        return True
    
    def _get_compatible_states(self, collapsed_value: Any, possible_states: List[Any]) -> List[Any]:
        """Get states compatible with the collapsed value."""
        # Simplified compatibility check
        # In a real implementation, this would use pattern compatibility rules
        return possible_states
    
    def execute(self, *args, **kwargs):
        """Execute constraint propagation."""
        pass


class BacktrackingPropagationStrategy(BasicPropagationStrategy):
    """Propagation strategy with backtracking support."""
    
    def __init__(self):
        super().__init__()
        self._backtrack_stack = []
    
    def propagate_constraints(self, cells: List[List[Cell]], x: int, y: int, 
                            collapsed_value: Any, state: GeneratorState) -> bool:
        """Propagate with backtracking support."""
        # Save current state for potential backtracking
        self._save_state(cells, state)
        
        # Perform basic propagation
        success = super().propagate_constraints(cells, x, y, collapsed_value, state)
        
        if not success:
            # Try backtracking
            return self._backtrack(cells, state)
        
        return success
    
    def _save_state(self, cells: List[List[Cell]], state: GeneratorState) -> None:
        """Save current state for backtracking."""
        # Simplified state saving
        state_snapshot = {
            'cells': [[cell.possible.copy() for cell in row] for row in cells],  # Use possible instead of get_possible_states
            'state': state.status
        }
        self._backtrack_stack.append(state_snapshot)
    
    def _backtrack(self, cells: List[List[Cell]], state: GeneratorState) -> bool:
        """Perform backtracking to previous state."""
        if not self._backtrack_stack:
            return False
        
        # Restore previous state
        previous_state = self._backtrack_stack.pop()
        for i in range(len(cells)):
            for j in range(len(cells[0])):
                cells[i][j].constrain(previous_state['cells'][i][j])
        
        state.update_status(previous_state['state'])
        self._logger.info("Backtracked to previous state")
        return True


class AlgorithmStrategy(BaseStrategy):
    """Base class for WFC algorithm strategies."""
    
    def __init__(self, selection_strategy, propagation_strategy):
        super().__init__()
        self._selection_strategy = selection_strategy
        self._propagation_strategy = propagation_strategy
    
    @abstractmethod
    def run_algorithm(self, cells: List[List[Cell]], state: GeneratorState) -> bool:
        """Run the specific WFC algorithm."""
        pass
    
    def execute(self, cells: List[List[Cell]], state: GeneratorState) -> bool:
        """Execute the algorithm strategy."""
        return self.run_algorithm(cells, state)


class BasicWFCStrategy(AlgorithmStrategy):
    """Basic Wave Function Collapse algorithm strategy."""
    
    def run_algorithm(self, cells: List[List[Cell]], state: GeneratorState) -> bool:
        """Run basic WFC algorithm."""
        max_iterations = 10000
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Select cell to collapse
            cell_pos = self._selection_strategy.select_cell(cells, state)
            if cell_pos is None:
                # All cells collapsed
                break
            
            x, y = cell_pos
            cell = cells[x][y]
            
            # Collapse cell
            if not cell.collapse():
                if hasattr(self._selection_strategy, 'should_backtrack') and \
                   self._selection_strategy.should_backtrack(state):
                    # Try backtracking
                    continue
                else:
                    return False
            
            # Propagate constraints
            if not self._propagation_strategy.propagate_constraints(
                cells, x, y, cell.collapsed_value, state
            ):
                return False
            
            # Check for contradictions
            if self._has_contradictions(cells):
                return False
        
        return not self._has_contradictions(cells)
    
    def _has_contradictions(self, cells: List[List[Cell]]) -> bool:
        """Check if there are any contradictions in the grid."""
        for row in cells:
            for cell in row:
                if cell.collapsed_value is None and not cell.possible:  # Use correct attributes
                    return True
        return False


class StrategyFactory:
    """Factory for creating strategy instances."""
    
    def __init__(self):
        self._selection_strategies = {
            SelectionStrategy.RANDOM: RandomSelectionStrategy,
            SelectionStrategy.MIN_ENTROPY: MinEntropySelectionStrategy,
            SelectionStrategy.PRIORITY_QUEUE: PriorityQueueSelectionStrategy,
        }
        
        self._propagation_strategies = {
            'basic': BasicPropagationStrategy,
            'backtracking': BacktrackingPropagationStrategy,
        }
        
        self._algorithm_strategies = {
            AlgorithmType.BASIC_WFC: BasicWFCStrategy,
        }
    
    def create_selection_strategy(self, strategy_type: SelectionStrategy):
        """Create a selection strategy."""
        if strategy_type not in self._selection_strategies:
            raise StrategyError(f"Unknown selection strategy: {strategy_type}")
        
        return self._selection_strategies[strategy_type]()
    
    def create_propagation_strategy(self, strategy_name: str):
        """Create a propagation strategy."""
        if strategy_name not in self._propagation_strategies:
            raise StrategyError(f"Unknown propagation strategy: {strategy_name}")
        
        return self._propagation_strategies[strategy_name]()
    
    def create_algorithm_strategy(self, config: AlgorithmConfig):
        """Create a complete algorithm strategy."""
        selection = self.create_selection_strategy(config.selection_strategy)
        prop_key = 'basic'
        if config.custom_params and isinstance(config.custom_params, dict):
            candidate = config.custom_params.get('propagation')
            if candidate in self._propagation_strategies:
                prop_key = candidate
        propagation = self.create_propagation_strategy(prop_key)
        
        if config.algorithm_type not in self._algorithm_strategies:
            raise StrategyError(f"Unknown algorithm type: {config.algorithm_type}")
        
        algorithm_class = self._algorithm_strategies[config.algorithm_type]
        return algorithm_class(selection, propagation)
