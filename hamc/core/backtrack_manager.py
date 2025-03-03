from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
import logging
import heapq
import time
from .generator_state import GeneratorState, GeneratorStatus

@dataclass
class BacktrackPoint:
    """Represents a point in generation where we can backtrack to."""
    level: str  # 'global', 'intermediate', or 'local'
    position: Tuple[int, int]  # (row, col)
    state_snapshot: Any
    entropy: float
    failed_attempts: Set[str]  # Values that were tried and failed
    metadata: Dict[str, Any]  # Additional context-specific data
    timestamp: float = field(default_factory=time.time)
    attempts: int = 0  # Number of times this point was tried
    
    def __lt__(self, other: 'BacktrackPoint') -> bool:
        """Custom comparison for priority queue ordering.
        
        We prioritize points with higher entropy (more options) and 
        more recent timestamps as tie-breakers.
        """
        if abs(self.entropy - other.entropy) < 1e-6:
            return self.timestamp > other.timestamp
        return self.entropy > other.entropy

class BacktrackManager:
    """Manages sophisticated backtracking across multiple generation levels."""
    
    def __init__(self, max_attempts: int = 100, 
                 priority_strategy: str = 'entropy',
                 max_level_attempts: Dict[str, int] = None,
                 adaptive_backtracking: bool = True):
        """Initialize backtrack manager.
        
        Args:
            max_attempts: Maximum number of backtrack attempts
            priority_strategy: Strategy for selecting backtrack points
                               ('entropy', 'timestamp', 'level')
            max_level_attempts: Maximum attempts per level
            adaptive_backtracking: Whether to use adaptive backtracking
        """
        self.max_attempts = max_attempts
        self.priority_strategy = priority_strategy
        self.adaptive_backtracking = adaptive_backtracking
        self.heap: List[BacktrackPoint] = []
        self.attempt_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.level_attempts: Dict[str, int] = {'global': 0, 'intermediate': 0, 'local': 0}
        self.max_level_attempts = max_level_attempts or {
            'global': max_attempts // 5,
            'intermediate': max_attempts // 3,
            'local': max_attempts // 2
        }
        self.logger = logging.getLogger(__name__)
        self.level_success_rate: Dict[str, float] = {'global': 1.0, 'intermediate': 1.0, 'local': 1.0}
        
    def push(self, level: str, position: Tuple[int, int], 
            state_snapshot: Any, entropy: float,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new backtrack point.
        
        Args:
            level: Generation level ('global', 'intermediate', 'local')
            position: Cell position (row, col)
            state_snapshot: State to restore when backtracking
            entropy: Entropy value for this point
            metadata: Additional information about this point
        """
        point = BacktrackPoint(
            level=level,
            position=position,
            state_snapshot=state_snapshot,
            entropy=entropy,
            failed_attempts=set(),
            metadata=metadata or {},
            timestamp=time.time()
        )
        heapq.heappush(self.heap, point)
        self.logger.debug(f"Added backtrack point at {level}:{position} with entropy {entropy:.4f}")
        
    def pop(self) -> Optional[BacktrackPoint]:
        """Get the most promising backtrack point based on priority strategy.
        
        Returns:
            Optional[BacktrackPoint]: Best backtrack point or None if none available
        """
        if not self.heap:
            return None
            
        self.attempt_count += 1
        if self.attempt_count > self.max_attempts:
            self.logger.warning(f"Exceeded maximum backtrack attempts ({self.max_attempts})")
            return None
        
        # Apply adaptive backtracking strategies
        if self.adaptive_backtracking:
            return self._pop_adaptive()
        
        # Standard backtracking
        point = heapq.heappop(self.heap)
        self.level_attempts[point.level] += 1
        point.attempts += 1
        self.logger.info(
            f"Backtracking to {point.level}:{point.position} "
            f"with entropy {point.entropy:.4f}, "
            f"attempt {self.attempt_count}/{self.max_attempts}"
        )
        return point

    def _pop_adaptive(self) -> Optional[BacktrackPoint]:
        """Adaptively select backtrack point based on level success rates and constraints."""
        # Find a good candidate based on current success metrics
        candidates = []
        
        # Copy heap to avoid modifying the original
        temp_heap = list(self.heap)
        
        # First, check if any level has exceeded its maximum attempts
        exceed_levels = {lvl for lvl, att in self.level_attempts.items() 
                        if att >= self.max_level_attempts[lvl]}
        
        if exceed_levels:
            # Avoid levels that have exceeded their attempt limit
            temp_heap = [p for p in temp_heap if p.level not in exceed_levels]
        
        # If we have levels with low success rates, prioritize other levels
        low_success_levels = {lvl for lvl, rate in self.level_success_rate.items() 
                             if rate < 0.3 and self.level_attempts[lvl] > 5}
        
        if low_success_levels and len(low_success_levels) < len(self.level_success_rate):
            # If we have some problematic levels but not all, avoid them
            temp_heap = [p for p in temp_heap if p.level not in low_success_levels]
        
        if not temp_heap and self.heap:
            # Fall back to original heap if filtering removed all options
            temp_heap = list(self.heap)
            
        if not temp_heap:
            return None
            
        # Select top candidates based on entropy and recency
        candidates = heapq.nsmallest(min(3, len(temp_heap)), temp_heap)
        
        # Choose the best candidate with weighted random selection
        weights = {}
        for i, point in enumerate(candidates):
            # Higher weight for:
            # - Higher entropy (more options)
            # - Fewer previous attempts
            # - Higher success rate for this level
            # - More recent timestamp
            w = (point.entropy * 0.4 + 
                (1.0 / (point.attempts + 1)) * 0.3 +
                self.level_success_rate[point.level] * 0.2 +
                (1.0 / (time.time() - point.timestamp + 1)) * 0.1)
            weights[i] = max(w, 0.1)  # Ensure positive weight
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Random weighted selection
        import random
        r = random.random()
        cumulative = 0
        for i, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                point = candidates[i]
                self.heap.remove(point)
                heapq.heapify(self.heap)
                break
        else:
            # Fallback - take first candidate
            point = candidates[0]
            self.heap.remove(point)
            heapq.heapify(self.heap)
        
        self.level_attempts[point.level] += 1
        point.attempts += 1
        self.logger.info(
            f"[ADAPTIVE] Backtracking to {point.level}:{point.position} "
            f"with entropy {point.entropy:.4f}, "
            f"attempt {self.attempt_count}/{self.max_attempts}, "
            f"level success rate: {self.level_success_rate[point.level]:.2f}"
        )
        return point
        
    def record_failure(self, level: str, position: Tuple[int, int], 
                      failed_value: str) -> None:
        """Record a failed attempt at a specific position.
        
        Args:
            level: Generation level
            position: Cell position
            failed_value: Value that failed
        """
        self.failure_count += 1
        
        # Update level success rate
        if self.adaptive_backtracking:
            total = self.success_count + self.failure_count
            if total > 0:
                for lvl in self.level_attempts.keys():
                    lvl_failures = sum(1 for p in self.heap 
                                      if p.level == lvl and p.failed_attempts)
                    lvl_attempts = max(self.level_attempts[lvl], 1)
                    self.level_success_rate[lvl] = max(0.1, 1 - (lvl_failures / lvl_attempts))
        
        for point in self.heap:
            if point.level == level and point.position == position:
                point.failed_attempts.add(failed_value)
                self.logger.debug(f"Recorded failure at {level}:{position} for value {failed_value}")
                # Reheapify if we modified a heap element
                heapq.heapify(self.heap)
                return
    
    def record_success(self, level: str) -> None:
        """Record a successful generation at a specific level.
        
        Args:
            level: Generation level that succeeded
        """
        self.success_count += 1
        
        # Update level success rate
        if self.adaptive_backtracking:
            for lvl in self.level_success_rate.keys():
                if lvl == level:
                    # Improve success rate for this level
                    self.level_success_rate[lvl] = min(1.0, self.level_success_rate[lvl] + 0.1)
        
        self.logger.info(f"Recorded successful generation at level {level}")
                
    def get_failed_values(self, level: str, 
                         position: Tuple[int, int]) -> Set[str]:
        """Get all values that failed at a specific position.
        
        Args:
            level: Generation level
            position: Cell position
            
        Returns:
            Set[str]: Set of failed values
        """
        for point in self.heap:
            if point.level == level and point.position == position:
                return point.failed_attempts
        return set()
        
    def clear(self) -> None:
        """Reset the backtrack manager."""
        self.heap.clear()
        self.attempt_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.level_attempts = {'global': 0, 'intermediate': 0, 'local': 0}
        self.level_success_rate = {'global': 1.0, 'intermediate': 1.0, 'local': 1.0}
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about backtracking."""
        return {
            "total_attempts": self.attempt_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "remaining_points": len(self.heap),
            "level_attempts": self.level_attempts,
            "level_success_rates": self.level_success_rate,
            "stack_entropy_stats": {
                "min": min((p.entropy for p in self.heap), default=0),
                "max": max((p.entropy for p in self.heap), default=0),
                "avg": sum((p.entropy for p in self.heap), default=0) / max(len(self.heap), 1),
                "levels": list(set(p.level for p in self.heap))
            } if self.heap else {}
        }
    
    def get_problematic_positions(self, level: str, threshold: int = 3) -> List[Tuple[int, int]]:
        """Get positions that have had multiple failures.
        
        Args:
            level: Generation level to check
            threshold: Minimum number of failures to be considered problematic
            
        Returns:
            List[Tuple[int, int]]: List of problematic positions
        """
        position_failures = {}
        for point in self.heap:
            if point.level == level:
                position_failures[point.position] = len(point.failed_attempts)
        
        return [pos for pos, failures in position_failures.items() 
               if failures >= threshold]