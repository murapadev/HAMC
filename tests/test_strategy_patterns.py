# tests/test_strategy_patterns.py
"""
Tests for Strategy Patterns
Tests the strategy pattern implementations for different algorithms and behaviors.
"""

import unittest
from unittest.mock import Mock
from hamc.core.strategy_patterns import (
    StrategyFactory, AlgorithmConfig, AlgorithmType, SelectionStrategy,
    BasicWFCStrategy, MinEntropySelectionStrategy, PriorityQueueSelectionStrategy,
    BasicPropagationStrategy, BacktrackingPropagationStrategy
)
from hamc.core.cell import Cell
from hamc.core.generator_state import GeneratorState, GeneratorStatus


class TestStrategyFactory(unittest.TestCase):
    """Test cases for StrategyFactory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = StrategyFactory()
    
    def test_create_selection_strategy_random(self):
        """Test creating random selection strategy."""
        strategy = self.factory.create_selection_strategy(SelectionStrategy.RANDOM)
        
        self.assertIsInstance(strategy, type(self.factory._selection_strategies[SelectionStrategy.RANDOM]()))
    
    def test_create_selection_strategy_min_entropy(self):
        """Test creating minimum entropy selection strategy."""
        strategy = self.factory.create_selection_strategy(SelectionStrategy.MIN_ENTROPY)
        
        self.assertIsInstance(strategy, MinEntropySelectionStrategy)
    
    def test_create_selection_strategy_priority_queue(self):
        """Test creating priority queue selection strategy."""
        strategy = self.factory.create_selection_strategy(SelectionStrategy.PRIORITY_QUEUE)
        
        self.assertIsInstance(strategy, PriorityQueueSelectionStrategy)
    
    def test_create_propagation_strategy_basic(self):
        """Test creating basic propagation strategy."""
        strategy = self.factory.create_propagation_strategy('basic')
        
        self.assertIsInstance(strategy, BasicPropagationStrategy)
    
    def test_create_propagation_strategy_backtracking(self):
        """Test creating backtracking propagation strategy."""
        strategy = self.factory.create_propagation_strategy('backtracking')
        
        self.assertIsInstance(strategy, BacktrackingPropagationStrategy)
    
    def test_create_algorithm_strategy(self):
        """Test creating complete algorithm strategy."""
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BASIC_WFC,
            selection_strategy=SelectionStrategy.MIN_ENTROPY
        )
        
        strategy = self.factory.create_algorithm_strategy(config)
        
        self.assertIsInstance(strategy, BasicWFCStrategy)


class TestSelectionStrategies(unittest.TestCase):
    """Test cases for selection strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cells = [
            [Cell({"A": 0.5, "B": 0.5}), Cell({"A": 0.3, "B": 0.7})],
            [Cell({"A": 0.8, "B": 0.2}), Cell({"A": 0.1, "B": 0.9})]
        ]
        self.state = GeneratorState("test")
    
    def test_min_entropy_selection(self):
        """Test minimum entropy selection strategy."""
        strategy = MinEntropySelectionStrategy()
        
        # Cell (1,1) has lowest entropy (A:0.1, B:0.9)
        cell_pos = strategy.select_cell(self.cells, self.state)
        
        self.assertEqual(cell_pos, (1, 1))
    
    def test_priority_queue_selection(self):
        """Test priority queue selection strategy."""
        strategy = PriorityQueueSelectionStrategy()
        
        # Should select cell with minimum entropy
        cell_pos = strategy.select_cell(self.cells, self.state)
        
        self.assertEqual(cell_pos, (1, 1))
    
    def test_backtrack_when_failed(self):
        """Test backtracking when generator is in failed state."""
        strategy = MinEntropySelectionStrategy()
        self.state.update_status(GeneratorStatus.FAILED)
        
        should_backtrack = strategy.should_backtrack(self.state)
        self.assertTrue(should_backtrack)


class TestPropagationStrategies(unittest.TestCase):
    """Test cases for propagation strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cells = [
            [Cell({"A": 1.0}), Cell({"A": 0.5, "B": 0.5})],
            [Cell({"B": 1.0}), Cell({"A": 0.3, "B": 0.7})]
        ]
        self.state = GeneratorState("test")
    
    def test_basic_propagation(self):
        """Test basic constraint propagation."""
        strategy = BasicPropagationStrategy()
        
        # Collapse first cell
        self.cells[0][0].collapse()
        
        # Propagate constraints
        success = strategy.propagate_constraints(self.cells, 0, 0, "A", self.state)
        
        self.assertTrue(success)
    
    def test_backtracking_propagation(self):
        """Test backtracking propagation strategy."""
        strategy = BacktrackingPropagationStrategy()
        
        # Collapse first cell
        self.cells[0][0].collapse()
        
        # Propagate with backtracking
        success = strategy.propagate_constraints(self.cells, 0, 0, "A", self.state)
        
        self.assertTrue(success)


class TestAlgorithmStrategies(unittest.TestCase):
    """Test cases for algorithm strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cells = [
            [Cell({"A": 0.5, "B": 0.5}), Cell({"A": 0.5, "B": 0.5})],
            [Cell({"A": 0.5, "B": 0.5}), Cell({"A": 0.5, "B": 0.5})]
        ]
        self.state = GeneratorState("test")
    
    def test_basic_wfc_strategy(self):
        """Test basic WFC algorithm strategy."""
        selection = MinEntropySelectionStrategy()
        propagation = BasicPropagationStrategy()
        strategy = BasicWFCStrategy(selection, propagation)
        
        # Run algorithm
        success = strategy.run_algorithm(self.cells, self.state)
        
        # Should succeed with simple 2x2 grid
        self.assertTrue(success)


class TestAlgorithmConfig(unittest.TestCase):
    """Test cases for AlgorithmConfig."""
    
    def test_algorithm_config_creation(self):
        """Test creating algorithm configuration."""
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BASIC_WFC,
            selection_strategy=SelectionStrategy.MIN_ENTROPY,
            max_iterations=1000,
            backtrack_limit=50
        )
        
        self.assertEqual(config.algorithm_type, AlgorithmType.BASIC_WFC)
        self.assertEqual(config.selection_strategy, SelectionStrategy.MIN_ENTROPY)
        self.assertEqual(config.max_iterations, 1000)
        self.assertEqual(config.backtrack_limit, 50)
    
    def test_algorithm_config_defaults(self):
        """Test algorithm configuration defaults."""
        config = AlgorithmConfig(
            algorithm_type=AlgorithmType.BASIC_WFC,
            selection_strategy=SelectionStrategy.RANDOM
        )
        
        self.assertEqual(config.max_iterations, 10000)
        self.assertEqual(config.backtrack_limit, 1000)
        self.assertIsNone(config.custom_params)


if __name__ == '__main__':
    unittest.main()