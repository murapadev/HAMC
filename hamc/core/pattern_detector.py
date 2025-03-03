from typing import Dict, List, Set, Tuple, Optional, Any, Generator
import numpy as np
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass

@dataclass
class Pattern:
    """Represents a detected pattern in generated content."""
    elements: List[Tuple[str, Tuple[int, int]]]  # (value, (row, col))
    frequency: int = 1
    significance: float = 0.0
    
    @property
    def size(self) -> int:
        """Get pattern size."""
        return len(self.elements)
    
    def matches(self, other: 'Pattern', fuzzy: bool = False) -> bool:
        """Check if this pattern matches another pattern."""
        if not fuzzy and len(self.elements) != len(other.elements):
            return False
            
        # Check for exact match
        self_values = sorted((val, pos) for val, pos in self.elements)
        other_values = sorted((val, pos) for val, pos in other.elements)
        
        if not fuzzy:
            return self_values == other_values
        
        # Fuzzy match - allow partial overlap
        overlap = 0
        for val, pos in self_values:
            if any(val == oval and pos == opos for oval, opos in other_values):
                overlap += 1
        
        return overlap / max(len(self_values), len(other_values)) >= 0.7  # 70% overlap

class PatternDetector:
    """Detects and analyzes patterns in generated content."""
    
    def __init__(self, min_pattern_size: int = 2, max_pattern_size: int = 5):
        """Initialize pattern detector.
        
        Args:
            min_pattern_size: Minimum size of patterns to detect
            max_pattern_size: Maximum size of patterns to detect
        """
        self.min_pattern_size = min_pattern_size
        self.max_pattern_size = max_pattern_size
        self.known_patterns: List[Pattern] = []
        self.logger = logging.getLogger(__name__)
        self.value_positions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    
    def analyze_grid(self, 
                    grid: List[List[str]], 
                    store_positions: bool = True) -> List[Pattern]:
        """Analyze a grid to detect patterns.
        
        Args:
            grid: 2D grid of values
            store_positions: Whether to store value positions for future reference
            
        Returns:
            List[Pattern]: Detected patterns
        """
        height, width = len(grid), len(grid[0])
        self.logger.info(f"Analyzing grid of size {height}x{width} for patterns")
        
        # Reset or update value positions
        if store_positions:
            self.value_positions = defaultdict(list)
            for r in range(height):
                for c in range(width):
                    value = grid[r][c]
                    if value:
                        self.value_positions[value].append((r, c))
        
        # Detect patterns of different sizes
        detected_patterns = []
        for size in range(self.min_pattern_size, min(self.max_pattern_size + 1, 
                                                   min(height, width) + 1)):
            window_patterns = self._detect_window_patterns(grid, size)
            detected_patterns.extend(window_patterns)
            
            if window_patterns:
                self.logger.info(f"Detected {len(window_patterns)} patterns of size {size}")
        
        # Merge similar patterns and update known patterns
        self._update_known_patterns(detected_patterns)
        return detected_patterns
    
    def _detect_window_patterns(self, 
                              grid: List[List[str]], 
                              size: int) -> List[Pattern]:
        """Detect patterns using a sliding window approach.
        
        Args:
            grid: 2D grid of values
            size: Size of window
            
        Returns:
            List[Pattern]: Detected patterns in windows
        """
        height, width = len(grid), len(grid[0])
        patterns = []
        
        # Slide window over the grid
        for r in range(height - size + 1):
            for c in range(width - size + 1):
                window_elements = []
                for i in range(size):
                    for j in range(size):
                        value = grid[r+i][c+j]
                        if value:
                            # Store relative positions for pattern matching
                            window_elements.append((value, (i, j)))
                
                if len(window_elements) >= self.min_pattern_size:
                    pattern = Pattern(elements=window_elements)
                    
                    # Check if this pattern is already known
                    merged = False
                    for existing in patterns:
                        if existing.matches(pattern):
                            existing.frequency += 1
                            merged = True
                            break
                    
                    if not merged:
                        patterns.append(pattern)
        
        return patterns
    
    def _update_known_patterns(self, new_patterns: List[Pattern]) -> None:
        """Update known patterns with newly detected ones.
        
        Args:
            new_patterns: List of newly detected patterns
        """
        for new_pattern in new_patterns:
            merged = False
            for known in self.known_patterns:
                if known.matches(new_pattern):
                    known.frequency += new_pattern.frequency
                    known.significance = self._calculate_significance(known)
                    merged = True
                    break
            
            if not merged:
                new_pattern.significance = self._calculate_significance(new_pattern)
                self.known_patterns.append(new_pattern)
    
    def _calculate_significance(self, pattern: Pattern) -> float:
        """Calculate pattern significance based on frequency and size.
        
        Args:
            pattern: Pattern to evaluate
            
        Returns:
            float: Significance score
        """
        # Patterns that are both frequent and large are more significant
        return pattern.frequency * (pattern.size / self.max_pattern_size)
    
    def get_constraints(self, 
                       position: Tuple[int, int], 
                       context: Dict[str, Any]) -> Dict[str, float]:
        """Generate constraints for a position based on detected patterns.
        
        Args:
            position: Position to generate constraints for
            context: Additional context information
            
        Returns:
            Dict[str, float]: Value weights based on detected patterns
        """
        weights = defaultdict(float)
        r, c = position
        
        # Check position against known pattern positions
        for pattern in self.known_patterns:
            for value, (pr, pc) in pattern.elements:
                # If the relative position matches, increase weight for that value
                if (r - pr, c - pc) in self.value_positions.get(value, []):
                    weights[value] += pattern.significance
        
        # Normalize weights
        total = sum(weights.values()) or 1.0
        return {k: v/total for k, v in weights.items()}
    
    def suggest_value(self, 
                    position: Tuple[int, int], 
                    context: Dict[str, Any],
                    existing_weights: Dict[str, float]) -> Dict[str, float]:
        """Suggest value weights based on patterns.
        
        Args:
            position: Position to generate suggestion for
            context: Additional context information
            existing_weights: Existing value weights
            
        Returns:
            Dict[str, float]: Adjusted value weights
        """
        pattern_weights = self.get_constraints(position, context)
        
        # Only use values that are in existing weights
        valid_pattern_weights = {k: v for k, v in pattern_weights.items() 
                               if k in existing_weights}
        
        if not valid_pattern_weights:
            return existing_weights
            
        # Combine existing weights with pattern weights (70% existing, 30% pattern)
        result = {}
        for value, weight in existing_weights.items():
            pattern_weight = valid_pattern_weights.get(value, 0.0)
            result[value] = 0.7 * weight + 0.3 * pattern_weight
        
        return result
