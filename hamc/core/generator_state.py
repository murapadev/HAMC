from enum import Enum
from typing import Optional, Dict, Any, List, Set
from time import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
import json
import sys

@dataclass
class GeneratorMetrics:
    """Metrics tracked during generation."""
    collapses: int = 0
    backtracks: int = 0
    propagations: int = 0
    constraint_violations: int = 0
    start_time: float = field(default_factory=time)
    events: List[str] = field(default_factory=list)
    max_entropy: float = float('-inf')
    min_entropy: float = float('inf')
    avg_entropy: float = 0.0
    entropy_samples: int = 0
    entropy_history: List[float] = field(default_factory=list)
    cells_collapsed: int = 0
    cells_total: int = 0
    
    def add_event(self, event: str) -> None:
        """Add timestamped event."""
        timestamp = datetime.now().isoformat()
        self.events.append(f"{timestamp}: {event}")
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time() - self.start_time
        
    def update_entropy_stats(self, entropy: float) -> None:
        """Update entropy statistics.
        
        Args:
            entropy: Current entropy value
        """
        if entropy < 0:  # Skip negative entropy values (collapsed cells)
            return
            
        self.entropy_samples += 1
        self.max_entropy = max(self.max_entropy, entropy)
        self.min_entropy = min(self.min_entropy, entropy)
        
        # Update running average
        self.avg_entropy = ((self.avg_entropy * (self.entropy_samples - 1)) + entropy) / self.entropy_samples
        
        # Store history (with limit to prevent memory issues)
        if len(self.entropy_history) < 1000:
            self.entropy_history.append(entropy)

class GeneratorStatus(Enum):
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BACKTRACKING = "backtracking"
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"

class GeneratorState:
    """Tracks state and metrics of a generator during execution."""
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        """Initialize generator state.
        
        Args:
            name: Name of the generator
            log_level: Logging level
        """
        self.name = name
        self.status = GeneratorStatus.INITIALIZED
        self.metrics = GeneratorMetrics()
        self.logger = logging.getLogger(name)
        self.step_counter = 0
        self.failures: Dict[str, int] = {}
        self.warnings: Dict[str, int] = {}
        self.timing: Dict[str, float] = {}
        
        # Configure logger
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

    def update_status(self, status: GeneratorStatus, message: Optional[str] = None) -> None:
        """Update generator status with optional message."""
        old_status = self.status
        self.status = status
        event = f"Status changed from {old_status.value} to {status.value}"
        if message:
            event += f": {message}"
        
        self.metrics.add_event(event)
        self.logger.info(event)
        self.step_counter += 1

    def increment_stat(self, stat_name: str) -> None:
        """Increment a metric counter."""
        if hasattr(self.metrics, stat_name):
            setattr(self.metrics, stat_name, getattr(self.metrics, stat_name) + 1)
            
            # Log significant milestones
            value = getattr(self.metrics, stat_name)
            if value > 0 and value % 100 == 0:
                self.logger.info(f"{stat_name} milestone: {value}")

    def update_entropy(self, entropy: float) -> None:
        """Update entropy statistics."""
        self.metrics.update_entropy_stats(entropy)

    def record_timing(self, operation: str, elapsed: float) -> None:
        """Record timing information for an operation.
        
        Args:
            operation: Name of the operation
            elapsed: Time elapsed in seconds
        """
        if operation not in self.timing:
            self.timing[operation] = 0
        self.timing[operation] += elapsed

    def record_failure(self, failure_type: str) -> None:
        """Record a failure.
        
        Args:
            failure_type: Type of failure
        """
        self.failures[failure_type] = self.failures.get(failure_type, 0) + 1
        self.logger.warning(f"Failure: {failure_type}")

    def record_warning(self, warning_type: str) -> None:
        """Record a warning.
        
        Args:
            warning_type: Type of warning
        """
        self.warnings[warning_type] = self.warnings.get(warning_type, 0) + 1

    def update_cell_count(self, total: int, collapsed: int) -> None:
        """Update cell count statistics.
        
        Args:
            total: Total number of cells
            collapsed: Number of collapsed cells
        """
        self.metrics.cells_total = total
        self.metrics.cells_collapsed = collapsed
        
        # Log progress
        progress = (collapsed / total) * 100 if total > 0 else 0
        if collapsed % max(1, total // 10) == 0:  # Log every 10%
            self.logger.info(f"Progress: {progress:.1f}% ({collapsed}/{total})")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of generation metrics."""
        return {
            "name": self.name,
            "status": self.status.value,
            "elapsed_time": self.metrics.get_elapsed_time(),
            "collapses": self.metrics.collapses,
            "backtracks": self.metrics.backtracks,
            "propagations": self.metrics.propagations,
            "constraint_violations": self.metrics.constraint_violations,
            "max_entropy": self.metrics.max_entropy,
            "min_entropy": self.metrics.min_entropy,
            "avg_entropy": self.metrics.avg_entropy,
            "cells_progress": f"{self.metrics.cells_collapsed}/{self.metrics.cells_total}",
            "progress_percent": (self.metrics.cells_collapsed / self.metrics.cells_total * 100 
                               if self.metrics.cells_total > 0 else 0),
            "total_events": len(self.metrics.events),
            "failures": self.failures,
            "warnings": self.warnings,
            "performance": self.timing
        }
        
    def save_to_json(self, filepath: str) -> None:
        """Save metrics to a JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_summary(), f, indent=2)
                self.logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {str(e)}")
