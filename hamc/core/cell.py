from typing import Dict, Optional, Set, List, Tuple, Any
from ..core.entropy import shannon_entropy, weighted_random_choice, normalized_weights
import logging
import numpy as np
import copy
import time

class Cell:
    """Base class for all cell types in the HAMC system with enhanced entropy calculation.
    
    Attributes:
        possible (Dict[str, float]): Dictionary of possible values and their weights
        collapsed_value (Optional[str]): The final value after collapse, or None if not collapsed
    """
    
    def __init__(self, possible_values: Dict[str, float]):
        """Initialize a cell with possible values and weights.
        
        Args:
            possible_values (Dict[str, float]): Dictionary of possible values and their weights
        
        Raises:
            ValueError: If possible_values is empty or contains invalid weights
        """
        if not possible_values:
            raise ValueError("Cannot initialize cell with empty possibilities")
        
        if any(w < 0 for w in possible_values.values()):
            raise ValueError("All weights must be non-negative")
            
        # Filter out zero weights
        self.possible = {k: v for k, v in possible_values.items() if v > 0}
        
        if not self.possible:
            raise ValueError("After filtering zero weights, no valid options remain")
            
        self.collapsed_value: Optional[str] = None
        self.history: List[Tuple[Dict[str, float], Optional[str]]] = []
        self.logger = logging.getLogger(__name__)
        
        # Track metadata for advanced pattern recognition
        self.position: Optional[Tuple[int, int]] = None
        self.generation_level: Optional[str] = None
        self.collapse_timestamp: Optional[float] = None
        self.entropy_history: List[float] = []
        self.constraints_applied: int = 0
        
    def set_metadata(self, position: Optional[Tuple[int, int]] = None, 
                   level: Optional[str] = None) -> None:
        """Set metadata for this cell.
        
        Args:
            position: The cell's position as (row, col)
            level: Generation level ('global', 'intermediate', 'local')
        """
        self.position = position
        self.generation_level = level
    
    def entropy(self, temperature: float = 1.0) -> float:
        """Calculate the entropy of the cell's possible values.
        
        Args:
            temperature: Temperature parameter for entropy calculation
            
        Returns:
            float: Shannon entropy of possible values, -1 if collapsed, or inf if no valid options
        """
        if self.collapsed_value is not None:
            return -1
        
        entropy_value = shannon_entropy(self.possible)
        self.entropy_history.append(entropy_value)
        
        # Keep history size reasonable
        if len(self.entropy_history) > 10:
            self.entropy_history = self.entropy_history[-10:]
            
        return entropy_value
    
    def collapse(self, 
                avoid_values: Optional[Set[str]] = None, 
                temperature: float = 1.0,
                context: Optional[Dict[str, Any]] = None) -> str:
        """Collapse the cell to a single value based on weighted probabilities.
        
        Args:
            avoid_values: Set of values to avoid when collapsing
            temperature: Temperature parameter for exploration control
            context: Additional context information for smarter collapse
            
        Returns:
            str: The selected value or "ERROR" if collapse fails
        """
        try:
            # Save current state to history
            self.history.append((copy.deepcopy(self.possible), self.collapsed_value))
            
            # Filter out values to avoid
            if avoid_values:
                options = {k: v for k, v in self.possible.items() if k not in avoid_values}
                if not options:
                    self.logger.warning("All options are in avoid_values set")
                    options = self.possible  # Fall back to all options if necessary
            else:
                options = self.possible
            
            # Apply context-based adjustment if available
            if context and 'neighbor_values' in context:
                options = self._adjust_for_neighbors(options, context['neighbor_values'])
            
            # Select value using temperature-controlled random choice
            self.collapsed_value = weighted_random_choice(options, temperature, avoid_values)
            
            if self.collapsed_value == "ERROR":
                self.logger.error("Weighted random choice failed")
                return "ERROR"
                
            self.possible = {self.collapsed_value: 1.0}
            self.collapse_timestamp = time.time()
            return self.collapsed_value
            
        except Exception as e:
            self.logger.error(f"Error during cell collapse: {str(e)}")
            return "ERROR"
    
    def _adjust_for_neighbors(self, 
                            options: Dict[str, float], 
                            neighbors: Dict[str, str]) -> Dict[str, float]:
        """Adjust weights based on neighboring cells' values.
        
        Args:
            options: Current option weights
            neighbors: Dictionary mapping direction to neighbor value
            
        Returns:
            Dict[str, float]: Adjusted option weights
        """
        adjusted = dict(options)
        
        # Count neighbor value frequencies
        neighbor_counts = {}
        for direction, value in neighbors.items():
            if value:
                neighbor_counts[value] = neighbor_counts.get(value, 0) + 1
        
        # Boost weights of values that appear in neighbors
        for option in list(adjusted.keys()):
            # Similarity boost for values that match neighbors
            if option in neighbor_counts:
                adjusted[option] *= 1.0 + (neighbor_counts[option] * 0.2)
        
        return adjusted
    
    def reset(self) -> None:
        """Reset the cell to its initial state."""
        self.collapsed_value = None
        # Restore initial state if history exists
        if self.history:
            self.possible = copy.deepcopy(self.history[0][0])
            self.history.clear()
        self.entropy_history.clear()
        self.constraints_applied = 0
            
    def undo(self) -> bool:
        """Undo the last operation on this cell.
        
        Returns:
            bool: True if undo was successful, False otherwise
        """
        if not self.history:
            return False
            
        previous_state = self.history.pop()
        self.possible = previous_state[0]
        self.collapsed_value = previous_state[1]
        if self.constraints_applied > 0:
            self.constraints_applied -= 1
        return True
        
    def constrain(self, 
                allowed_values: Set[str],
                reason: Optional[str] = None) -> bool:
        """Constrain possible values to intersection with allowed values.
        
        Args:
            allowed_values: Set of values to allow
            reason: Optional reason for the constraint (for logging)
            
        Returns:
            bool: True if constraints were applied successfully and options remain
        """
        # Save current state to history
        self.history.append((copy.deepcopy(self.possible), self.collapsed_value))
        
        # Find intersection and keep weights
        new_possible = {k: v for k, v in self.possible.items() if k in allowed_values}
        
        if not new_possible:
            self.logger.warning(f"Constraint removed all options: {reason or 'unknown reason'}")
            return False
            
        changed = new_possible != self.possible
        self.possible = new_possible
        
        # If already collapsed to a value not in allowed_values, uncollapse
        if self.collapsed_value is not None and self.collapsed_value not in allowed_values:
            self.collapsed_value = None
            
        if changed:
            self.constraints_applied += 1
            
        return changed
    
    def boost_values(self, 
                   values_to_boost: Dict[str, float],
                   boost_factor: float = 1.5) -> None:
        """Boost weights of specific values by a factor.
        
        Args:
            values_to_boost: Dictionary mapping values to boost factors
            boost_factor: Default factor to apply
        """
        if self.collapsed_value is not None:
            return
            
        # Save current state to history
        self.history.append((copy.deepcopy(self.possible), self.collapsed_value))
        
        # Apply boosts
        for value, factor in values_to_boost.items():
            if value in self.possible:
                self.possible[value] *= factor or boost_factor
                
        # Normalize again
        total_weight = sum(self.possible.values())
        if total_weight > 0:
            self.possible = {k: v / total_weight for k, v in self.possible.items()}
        
    def get_top_options(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get the top N options by weight.
        
        Args:
            n: Number of top options to return
            
        Returns:
            List[Tuple[str, float]]: Top options with their weights
        """
        return sorted(self.possible.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def get_context_dict(self) -> Dict[str, Any]:
        """Get a dictionary of context information for this cell.
        
        Returns:
            Dict[str, Any]: Context information
        """
        return {
            'position': self.position,
            'level': self.generation_level,
            'collapsed': self.collapsed_value is not None,
            'value': self.collapsed_value,
            'entropy': self.entropy() if self.collapsed_value is None else -1,
            'top_options': self.get_top_options(3),
            'constraints_applied': self.constraints_applied,
            'history_depth': len(self.history)
        }
    
    def copy_from(self, other: 'Cell') -> None:
        """Copy state from another cell.
        
        Args:
            other: Cell to copy from
        """
        self.possible = copy.deepcopy(other.possible)
        self.collapsed_value = other.collapsed_value
        # Don't copy history - this is a fresh state
        self.history.clear()
