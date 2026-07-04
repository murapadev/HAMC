import logging
from typing import Dict, Tuple, Optional, Set, Callable
import numpy as np
from functools import lru_cache


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
        logging.getLogger(__name__).warning(
            "weighted_random_choice failed (options=%d): %s", len(weighted_options), e
        )
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
