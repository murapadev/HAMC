import math
from typing import Dict, Union, Tuple, List, Optional, Any, Set, Callable
import numpy as np
from numpy.typing import NDArray
from functools import lru_cache
import time
from ..config.advanced_config import get_entropy_config


class AdaptiveEntropyCalculator:
    """Advanced entropy calculation with spatial context awareness and learning capabilities."""
    
    def __init__(self, temperature: Optional[float] = None, learning_rate: Optional[float] = None):
        """Initialize the adaptive entropy calculator.
        
        Args:
            temperature: Temperature parameter for softmax normalization
            learning_rate: Rate at which the system adapts to successful patterns
        """
        config = get_entropy_config()
        self.temperature = temperature if temperature is not None else config.default_temperature
        self.learning_rate = learning_rate if learning_rate is not None else config.default_learning_rate
        self.pattern_weights: Dict[str, float] = {}
        self.context_cache = {}
        self.successful_patterns: List[Dict[str, Any]] = []
        self.temporal_decay_factor = config.temporal_decay_factor  # Decay factor for older patterns
        self.last_update_time = time.time()
        
    def calculate_entropy(self, 
                         weighted_options: Dict[str, float],
                         context: Optional[Dict[str, Any]] = None,
                         spatial_coordinates: Optional[Tuple[int, int]] = None,
                         level: Optional[str] = None,
                         epsilon: float = 1e-10) -> float:
        """Calculate context-aware entropy with pattern recognition.
        
        Args:
            weighted_options: Dictionary mapping options to their weights
            context: Additional context information to guide entropy calculation
            spatial_coordinates: Position information for spatial bias
            level: Generation level ('global', 'intermediate', 'local')
            epsilon: Small constant to prevent log(0)
            
        Returns:
            float: Adjusted entropy value
        """
        if not weighted_options:
            return float('inf')
        
        # Apply pattern-based adjustments if we have spatial coordinates
        if spatial_coordinates and self.pattern_weights:
            adjusted_weights = {}
            for option, weight in weighted_options.items():
                # Consider both position-specific and level-specific patterns
                pattern_keys = [
                    f"{option}_{spatial_coordinates[0]}_{spatial_coordinates[1]}",
                    f"{option}_{level}" if level else None,
                    f"{level}_{spatial_coordinates[0]}_{spatial_coordinates[1]}" if level else None
                ]
                
                pattern_boost = 1.0
                for key in pattern_keys:
                    if key and key in self.pattern_weights:
                        pattern_boost *= self.pattern_weights[key]
                
                adjusted_weights[option] = weight * pattern_boost
            weighted_options = adjusted_weights
        
        # Apply context-specific adjustments
        if context:
            context_hash = self._hash_context(context)
            if context_hash in self.context_cache:
                weighted_options = self._apply_context_adjustment(weighted_options, 
                                                               self.context_cache[context_hash])
        
        # Use softmax-normalized entropy for better gradient
        return self._softmax_entropy(weighted_options, self.temperature)
    
    def _softmax_entropy(self, 
                        weighted_options: Dict[str, float], 
                        temperature: float = 1.0,
                        epsilon: float = 1e-10) -> float:
        """Calculate entropy using softmax normalization for improved stability.
        
        Args:
            weighted_options: Dictionary mapping options to their weights
            temperature: Temperature parameter (higher = more uniform distribution)
            epsilon: Small constant to prevent numerical issues
            
        Returns:
            float: Softmax entropy value
        """
        weights = np.array(list(weighted_options.values()), dtype=np.float64)
        
        if np.any(weights < 0) or np.all(weights < epsilon):
            return float('inf')
            
        # Apply softmax with temperature
        weights = np.exp(weights / temperature)
        softmax_weights = weights / np.sum(weights)
        
        # Calculate entropy with masked array for numerical stability
        masked_weights = np.ma.masked_less_equal(softmax_weights, 0)
        log_weights = np.ma.log2(masked_weights)
        
        return float(-np.sum(masked_weights * log_weights))
    
    def learn_from_success(self, 
                          pattern: Dict[str, Any], 
                          success_score: float = 1.0) -> None:
        """Learn from successful generation patterns.
        
        Args:
            pattern: Dictionary describing the successful pattern
            success_score: A score indicating how successful the pattern was
        """
        config = get_entropy_config()
        self.successful_patterns.append(pattern)
        
        # Apply temporal decay to existing patterns
        current_time = time.time()
        time_since_update = current_time - self.last_update_time
        if time_since_update > 1.0:  # Only decay after significant time
            decay_factor = self.temporal_decay_factor ** (time_since_update / config.temporal_decay_interval)
            self.pattern_weights = {k: v * decay_factor for k, v in self.pattern_weights.items()}
            self.last_update_time = current_time
        
        # Update pattern weights
        if 'value' in pattern and 'position' in pattern:
            value = pattern['value']
            row, col = pattern['position']
            level = pattern.get('level', '')
            
            # Generate different pattern keys for multi-scale learning
            pattern_keys = [
                f"{value}_{row}_{col}",                  # Specific position-value
                f"{value}_{level}" if level else None,   # Value-level association
                f"{level}_{row}_{col}" if level else None   # Level-position association
            ]
            
            # Update weights for all valid pattern keys
            for key in pattern_keys:
                if key:
                    current_weight = self.pattern_weights.get(key, 1.0)
                    self.pattern_weights[key] = current_weight + (self.learning_rate * success_score)
            
            # If we have neighboring information, learn those patterns too
            if 'neighbors' in pattern:
                for direction, neighbor in pattern['neighbors'].items():
                    if neighbor:
                        neighbor_key = f"{neighbor}_{direction}_to_{value}"
                        current_weight = self.pattern_weights.get(neighbor_key, 1.0)
                        self.pattern_weights[neighbor_key] = current_weight + (self.learning_rate * success_score * config.global_learning_weight)
        
        # Learn from context patterns
        if 'context' in pattern:
            context_hash = self._hash_context(pattern['context'])
            if context_hash not in self.context_cache:
                self.context_cache[context_hash] = {}
            
            if 'value' in pattern:
                value = pattern['value']
                current_weight = self.context_cache[context_hash].get(value, 1.0)
                self.context_cache[context_hash][value] = current_weight + (self.learning_rate * success_score)
    
    def adjust_temperature(self, success_rate: float) -> None:
        """Dynamically adjust temperature based on success rate.
        
        Args:
            success_rate: Current success rate of the generation process
        """
        config = get_entropy_config()
        if success_rate < config.min_success_rate:
            # Lower temperature when success rate is low to be more conservative
            self.temperature = max(config.min_temperature, self.temperature * config.temperature_decay_factor)
        elif success_rate > config.max_success_rate:
            # Higher temperature when success rate is high to explore more
            self.temperature = min(config.max_temperature, self.temperature * (2.0 - config.temperature_decay_factor))
    
    def reset_learning(self) -> None:
        """Reset learned patterns."""
        self.pattern_weights = {}
        self.successful_patterns = []
        self.context_cache = {}
        self.temperature = 1.0
        self.last_update_time = time.time()
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create a hashable representation of context."""
        return str(sorted((k, str(v)) for k, v in context.items()))
    
    def _apply_context_adjustment(self, 
                                weighted_options: Dict[str, float], 
                                context_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply context-specific adjustments to option weights."""
        result = {}
        for option, weight in weighted_options.items():
            context_boost = context_weights.get(option, 1.0)
            result[option] = weight * context_boost
        return result

    def get_self_adapting_temperature(self, 
                                    entropy_history: List[float],
                                    min_temp: Optional[float] = None,
                                    max_temp: Optional[float] = None) -> float:
        """Calculate self-adapting temperature based on entropy history.
        
        Args:
            entropy_history: List of recent entropy values
            min_temp: Minimum temperature
            max_temp: Maximum temperature
            
        Returns:
            float: Adjusted temperature
        """
        config = get_entropy_config()
        min_temp = min_temp if min_temp is not None else config.min_temperature
        max_temp = max_temp if max_temp is not None else config.max_temperature
        
        if not entropy_history or len(entropy_history) < 3:
            return self.temperature
            
        # Check entropy trend
        recent = np.array(entropy_history[-config.trend_window_size:])
        if len(recent) < config.trend_window_size:
            return self.temperature
            
        # Calculate trend
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > config.trend_threshold:
            # Entropy increasing - reduce temperature to focus search
            return max(min_temp, self.temperature * config.temperature_decay_factor)
        elif trend < -config.trend_threshold:
            # Entropy decreasing - increase temperature to explore more
            return min(max_temp, self.temperature * (2.0 - config.temperature_decay_factor))
            
        return self.temperature


class MultiScaleEntropyAnalyzer:
    """Analyzes entropy across different scales and generation levels."""
    
    def __init__(self):
        """Initialize the multi-scale entropy analyzer."""
        config = get_entropy_config()
        self.global_entropy_calculator = AdaptiveEntropyCalculator(temperature=config.default_temperature * 1.2)
        self.intermediate_entropy_calculator = AdaptiveEntropyCalculator(temperature=config.default_temperature)
        self.local_entropy_calculator = AdaptiveEntropyCalculator(temperature=config.default_temperature * 0.8)
        self.entropy_history = {
            'global': [],
            'intermediate': [],
            'local': []
        }
        self.success_rates = {
            'global': 1.0,
            'intermediate': 1.0,
            'local': 1.0
        }
    
    def calculate_entropy(self,
                        weighted_options: Dict[str, float],
                        level: str,
                        context: Optional[Dict[str, Any]] = None,
                        spatial_coordinates: Optional[Tuple[int, int]] = None) -> float:
        """Calculate entropy for a specific level with appropriate context.
        
        Args:
            weighted_options: Dictionary mapping options to their weights
            level: Generation level ('global', 'intermediate', 'local')
            context: Additional context information
            spatial_coordinates: Position information for spatial bias
            
        Returns:
            float: Calculated entropy value
        """
        if level == 'global':
            calculator = self.global_entropy_calculator
        elif level == 'intermediate':
            calculator = self.intermediate_entropy_calculator
        else:  # local or any other level
            calculator = self.local_entropy_calculator
        
        entropy = calculator.calculate_entropy(
            weighted_options=weighted_options,
            context=context,
            spatial_coordinates=spatial_coordinates,
            level=level
        )
        
        # Store entropy in history
        history = self.entropy_history.get(level, [])
        history.append(entropy)
        if len(history) > 100:  # Limit history size
            history = history[-100:]
        self.entropy_history[level] = history
        
        return entropy
    
    def learn_from_success(self, 
                         pattern: Dict[str, Any],
                         level: str,
                         success_score: float = 1.0) -> None:
        """Learn from successful pattern at appropriate level.
        
        Args:
            pattern: Dictionary describing the successful pattern
            level: Generation level
            success_score: A score indicating how successful the pattern was
        """
        config = get_entropy_config()
        pattern['level'] = level  # Ensure level is in pattern
        
        if level == 'global':
            self.global_entropy_calculator.learn_from_success(pattern, success_score)
        elif level == 'intermediate':
            self.intermediate_entropy_calculator.learn_from_success(pattern, success_score)
            # Also learn at global level with reduced weight
            self.global_entropy_calculator.learn_from_success(pattern, success_score * config.global_learning_weight)
        else:  # local or any other level
            self.local_entropy_calculator.learn_from_success(pattern, success_score)
            # Also learn at intermediate level with reduced weight
            self.intermediate_entropy_calculator.learn_from_success(pattern, success_score * config.intermediate_learning_weight)
    
    def update_success_rate(self, level: str, success: bool) -> None:
        """Update success rate for a level.
        
        Args:
            level: Generation level
            success: Whether generation was successful
        """
        config = get_entropy_config()
        rate = self.success_rates.get(level, 1.0)
        # Exponential moving average
        rate = rate * config.success_decay_factor + (1.0 if success else 0.0) * config.success_smoothing_factor
        self.success_rates[level] = rate
        
        # Adjust temperature based on success rate
        if level == 'global':
            self.global_entropy_calculator.adjust_temperature(rate)
        elif level == 'intermediate':
            self.intermediate_entropy_calculator.adjust_temperature(rate)
        else:  # local or any other level
            self.local_entropy_calculator.adjust_temperature(rate)
    
    def reset(self, level: Optional[str] = None) -> None:
        """Reset learning for specified level or all levels.
        
        Args:
            level: Level to reset, or None for all levels
        """
        if level is None or level == 'global':
            self.global_entropy_calculator.reset_learning()
            self.entropy_history['global'] = []
            self.success_rates['global'] = 1.0
            
        if level is None or level == 'intermediate':
            self.intermediate_entropy_calculator.reset_learning()
            self.entropy_history['intermediate'] = []
            self.success_rates['intermediate'] = 1.0
            
        if level is None or level == 'local':
            self.local_entropy_calculator.reset_learning()
            self.entropy_history['local'] = []
            self.success_rates['local'] = 1.0


@lru_cache(maxsize=1024)
def optimized_shannon_entropy(options_tuple: Tuple[Tuple[str, float], ...], 
                             epsilon: float = 1e-10) -> float:
    """Cached version of Shannon entropy for improved performance.
    
    Args:
        options_tuple: Tuple of (option, weight) pairs
        epsilon: Small constant to prevent log(0)
        
    Returns:
        float: Shannon entropy value
    """
    if not options_tuple:
        return float('inf')
    
    weights = np.array([w for _, w in options_tuple], dtype=np.float64)
    
    if np.any(weights < 0) or np.all(weights < epsilon):
        return float('inf')
        
    total_weight = np.sum(weights)
    if total_weight <= epsilon:
        return float('inf')
        
    probs = weights / total_weight
    masked_probs = np.ma.masked_less_equal(probs, 0)
    log_probs = np.ma.log2(masked_probs)
    
    return float(-np.sum(masked_probs * log_probs))


def shannon_entropy(weighted_options: Dict[str, float], epsilon: float = 1e-10) -> float:
    """Calculate Shannon entropy for a dictionary of weighted options with improved numerical stability.
    
    Args:
        weighted_options: Dictionary mapping options to their weights
        epsilon: Small constant to prevent log(0)
        
    Returns:
        float: Shannon entropy value, or infinity if no valid options
    """
    # Convert dict to tuple for caching
    options_tuple = tuple(sorted(weighted_options.items()))
    return optimized_shannon_entropy(options_tuple, epsilon)


def weighted_random_choice(weighted_options: Dict[str, float], 
                         temperature: float = 1.0,
                         avoid_values: Optional[Set[str]] = None,
                         epsilon: float = 1e-10) -> str:
    """Choose an option based on weighted probabilities with improved numerical stability.
    
    Args:
        weighted_options: Dictionary mapping options to their weights
        temperature: Temperature parameter for exploration (higher = more random)
        avoid_values: Set of values to avoid if possible
        epsilon: Small constant for numerical stability
        
    Returns:
        str: Selected option or "ERROR" if selection fails
    """
    if not weighted_options:
        return "ERROR"
    
    try:
        # Filter out values to avoid if possible
        if avoid_values:
            available_options = {k: v for k, v in weighted_options.items() if k not in avoid_values}
            # Fall back to all options if filtering removed everything
            if not available_options:
                available_options = weighted_options
        else:
            available_options = weighted_options
        
        options = list(available_options.keys())
        weights = np.array([available_options[opt] for opt in options], dtype=np.float64)
        
        # Handle numerical instability
        total = np.sum(weights)
        if total <= epsilon:
            return "ERROR"
            
        # Apply temperature scaling to introduce randomness/determinism
        if temperature != 1.0:
            weights = np.power(weights, 1.0 / temperature)
            
        # Normalize probabilities
        probs = weights / np.sum(weights)
        
        # Check for NaN values after normalization
        if np.any(np.isnan(probs)):
            return "ERROR"
            
        # Use numpy's random choice for better performance
        return np.random.choice(options, p=probs)
        
    except Exception as e:
        # Log the specific error for debugging
        return "ERROR"


def normalized_weights(weighted_options: Dict[str, float], 
                     epsilon: float = 1e-10) -> Tuple[Dict[str, float], bool]:
    """Normalize option weights and check for validity.
    
    Args:
        weighted_options: Dictionary mapping options to their weights
        epsilon: Small constant for numerical stability
        
    Returns:
        Tuple[Dict[str, float], bool]: Normalized weights and success flag
    """
    if not weighted_options:
        return {}, False
        
    total = sum(weighted_options.values())
    if total <= epsilon:
        return {}, False
        
    normalized = {k: v/total for k, v in weighted_options.items()}
    return normalized, True


def filter_options(weighted_options: Dict[str, float],
                  predicate: Callable[[str], bool],
                  min_options: int = 1) -> Dict[str, float]:
    """Filter options based on a predicate function while ensuring minimum options remain.
    
    Args:
        weighted_options: Dictionary mapping options to their weights
        predicate: Function that returns True for options to keep
        min_options: Minimum number of options to maintain
        
    Returns:
        Dict[str, float]: Filtered options
    """
    filtered = {opt: weight for opt, weight in weighted_options.items() if predicate(opt)}
    
    # If filtering removed too many options, keep the highest weighted ones
    if filtered and len(filtered) < min_options and len(weighted_options) >= min_options:
        sorted_options = sorted(weighted_options.items(), key=lambda x: x[1], reverse=True)
        top_options = dict(sorted_options[:min_options])
        return top_options
    
    return filtered or weighted_options  # Return original if filtering removed everything
