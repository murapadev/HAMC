from typing import List, Dict, Tuple, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
import time
from ..core.generator_state import GeneratorState, GeneratorStatus
from ..core.pattern_detector import PatternDetector
from ..generators.local_generator import LocalGenerator

class ParallelLocalGenerator:
    """Manage parallel generation of multiple local blocks."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize parallel generator.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self.state = GeneratorState("ParallelLocalGenerator")
        self.pattern_detector = PatternDetector()
        self.successful_blocks: Dict[str, List[List[List[str]]]] = {}
        self.lock = threading.Lock()
        
    def generate_blocks(self, 
                      block_configs: List[Dict[str, Any]], 
                      timeout: float = 60.0) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """Generate multiple local blocks in parallel.
        
        Args:
            block_configs: Configuration for each block to generate
            timeout: Maximum time to wait for generation (in seconds)
            
        Returns:
            Dict[Tuple[int, int], Dict[str, Any]]: Generated blocks by position
        """
        self.state.update_status(GeneratorStatus.IN_PROGRESS)
        results = {}
        start_time = time.time()
        
        # Group blocks by type to apply pattern learning
        blocks_by_type = {}
        for config in block_configs:
            block_type = config.get('block_type', '')
            if block_type not in blocks_by_type:
                blocks_by_type[block_type] = []
            blocks_by_type[block_type].append(config)
        
        self.logger.info(f"Generating {len(block_configs)} blocks using {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            # Process blocks by type for better pattern learning
            for block_type, configs in blocks_by_type.items():
                self.logger.info(f"Processing {len(configs)} blocks of type '{block_type}'")
                
                # First submit blocks with stronger constraints
                configs.sort(key=lambda c: self._priority_score(c))
                
                for config in configs:
                    position = (config.get('row', 0), config.get('col', 0))
                    
                    future = executor.submit(
                        self._generate_block_with_retry,
                        config,
                        # Apply patterns from previous blocks of same type
                        self.successful_blocks.get(block_type, [])
                    )
                    futures[future] = position
            
            # Process results as they complete
            for future in as_completed(futures):
                position = futures[future]
                try:
                    block_result = future.result(timeout=max(1.0, timeout - (time.time() - start_time)))
                    
                    if block_result and block_result.get('success'):
                        results[position] = block_result
                        
                        # Update successful blocks for pattern learning
                        block_type = block_result.get('block_type', '')
                        tiles = block_result.get('tiles', [])
                        if tiles:
                            with self.lock:
                                if block_type not in self.successful_blocks:
                                    self.successful_blocks[block_type] = []
                                self.successful_blocks[block_type].append(tiles)
                                
                                # Update pattern detector with this successful block
                                self.pattern_detector.analyze_grid(tiles)
                    else:
                        self.logger.error(f"Failed to generate block at position {position}")
                
                except Exception as e:
                    self.logger.error(f"Error generating block at {position}: {str(e)}")
        
        self.state.update_status(
            GeneratorStatus.COMPLETED if len(results) == len(block_configs) 
            else GeneratorStatus.FAILED
        )
        
        generation_time = time.time() - start_time
        self.logger.info(f"Generated {len(results)}/{len(block_configs)} blocks in {generation_time:.2f}s")
        
        return results
    
    def _generate_block_with_retry(self, 
                                 config: Dict[str, Any],
                                 pattern_examples: List[List[List[str]]],
                                 max_attempts: int = 3) -> Dict[str, Any]:
        """Generate a single block with retry logic.
        
        Args:
            config: Block configuration
            pattern_examples: Examples of successful blocks of same type
            max_attempts: Maximum number of generation attempts
            
        Returns:
            Dict[str, Any]: Generation result
        """
        block_type = config.get('block_type', '')
        size = config.get('size', 4)
        neighbors = config.get('neighbors', {})
        position = (config.get('row', 0), config.get('col', 0))
        
        self.logger.debug(f"Generating {block_type} block at {position}")
        
        for attempt in range(max_attempts):
            generator = LocalGenerator(block_type, size, neighbors)
            
            # Apply pattern constraints if we have examples
            if pattern_examples and attempt > 0:
                self._apply_pattern_constraints(generator, pattern_examples)
                
            success = generator.collapse()
            
            if success and generator.validate():
                tiles = [[cell.collapsed_value for cell in row] for row in generator.cells]
                return {
                    'success': True,
                    'block_type': block_type,
                    'position': position,
                    'tiles': tiles,
                    'attempts': attempt + 1
                }
            
            self.logger.debug(f"Attempt {attempt+1} failed for {block_type} at {position}")
        
        return {
            'success': False,
            'block_type': block_type,
            'position': position,
            'attempts': max_attempts
        }
    
    def _apply_pattern_constraints(self, 
                                 generator: LocalGenerator,
                                 examples: List[List[List[str]]]) -> None:
        """Apply pattern-based constraints to a generator.
        
        Args:
            generator: Generator to apply constraints to
            examples: Example successful blocks
        """
        # Extract patterns from examples
        for example in examples:
            self.pattern_detector.analyze_grid(example)
            
        # Apply constraints to each cell based on detected patterns
        for r in range(generator.height):
            for c in range(generator.width):
                cell = generator.cells[r][c]
                if not cell.collapsed_value:
                    context = {
                        'block_type': generator.block_type,
                        'neighbors': generator.neighbors
                    }
                    # Adjust weights based on patterns
                    pattern_weights = self.pattern_detector.suggest_value(
                        (r, c), context, cell.possible
                    )
                    cell.possible = pattern_weights
    
    def _priority_score(self, config: Dict[str, Any]) -> float:
        """Calculate priority score for block generation ordering.
        
        Args:
            config: Block configuration
            
        Returns:
            float: Priority score (higher = more constrained)
        """
        score = 0.0
        neighbors = config.get('neighbors', {})
        
        # Blocks with more specific neighbors have higher priority
        for direction, neighbor_type in neighbors.items():
            if neighbor_type:
                score += 1.0
                
                # Special transitions are more constrained
                if direction in ['top', 'bottom'] and neighbor_type == 'rio':
                    score += 2.0
                if direction in ['left', 'right'] and neighbor_type == 'carretera':
                    score += 2.0
        
        return score
