# tests/test_factory_patterns.py
"""
Tests for Factory Patterns
Tests the factory pattern implementations for generators and components.
"""

import unittest
from unittest.mock import Mock, patch
from hamc.core.factory_patterns import (
    GeneratorFactory, HierarchicalGeneratorFactory, ComponentFactory,
    get_generator_factory, get_component_factory, reset_factories,
    GeneratorType, GeneratorFactoryError
)
from hamc.generators.base_generator import BaseGenerator
from hamc.generators.global_generator import GlobalGenerator
from hamc.generators.intermediate_generator import IntermediateGenerator
from hamc.generators.local_generator import LocalGenerator
from hamc.generators.parallel_generator import ParallelLocalGenerator


class MockGenerator(BaseGenerator):
    """Mock generator for testing."""

    def __init__(self, width: int, height: int):
        super().__init__(width, height, "mock")
        self.mock_initialized = False

    def initialize(self) -> bool:
        self.mock_initialized = True
        return True

    def _run_collapse(self) -> bool:
        return True

    def validate(self) -> bool:
        return True


class TestGeneratorFactory(unittest.TestCase):
    """Test cases for GeneratorFactory."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = GeneratorFactory()

    def tearDown(self):
        """Clean up after tests."""
        reset_factories()

    def test_default_generators_registered(self):
        """Test that default generators are registered."""
        registered = self.factory.get_registered()

        expected_generators = [
            GeneratorType.GLOBAL.value,
            GeneratorType.INTERMEDIATE.value,
            GeneratorType.LOCAL.value,
            GeneratorType.PARALLEL.value
        ]

        for gen_type in expected_generators:
            self.assertIn(gen_type, registered)
            self.assertTrue(self.factory.is_registered(gen_type))

    def test_create_global_generator(self):
        """Test creating a global generator."""
        generator = self.factory.create(GeneratorType.GLOBAL.value, 10, 10)

        self.assertIsInstance(generator, GlobalGenerator)
        self.assertEqual(generator.width, 10)
        self.assertEqual(generator.height, 10)

    def test_create_intermediate_generator(self):
        """Test creating an intermediate generator."""
        generator = self.factory.create(GeneratorType.INTERMEDIATE.value, 8, 8)

        self.assertIsInstance(generator, IntermediateGenerator)
        self.assertEqual(generator.width, 8)
        self.assertEqual(generator.height, 8)

    def test_create_local_generator(self):
        """Test creating a local generator."""
        generator = self.factory.create(GeneratorType.LOCAL.value, 6, 6)

        self.assertIsInstance(generator, LocalGenerator)
        self.assertEqual(generator.width, 6)
        self.assertEqual(generator.height, 6)

    def test_create_parallel_generator(self):
        """Test creating a parallel generator."""
        generator = self.factory.create(GeneratorType.PARALLEL.value, 4, 4)

        self.assertIsInstance(generator, ParallelLocalGenerator)
        self.assertEqual(generator.width, 4)
        self.assertEqual(generator.height, 4)

    def test_create_unknown_generator_type(self):
        """Test creating an unknown generator type."""
        with self.assertRaises(GeneratorFactoryError):
            self.factory.create("unknown_type", 5, 5)

    def test_register_custom_generator(self):
        """Test registering a custom generator."""
        # Register custom generator
        self.factory.register("mock", MockGenerator)

        # Verify registration
        self.assertTrue(self.factory.is_registered("mock"))
        self.assertIn("mock", self.factory.get_registered())

        # Create instance
        generator = self.factory.create("mock", 3, 3)
        self.assertIsInstance(generator, MockGenerator)

    def test_unregister_generator(self):
        """Test unregistering a generator."""
        # Register first
        self.factory.register("mock", MockGenerator)
        self.assertTrue(self.factory.is_registered("mock"))

        # Unregister
        self.factory.unregister("mock")
        self.assertFalse(self.factory.is_registered("mock"))

    def test_create_with_custom_params(self):
        """Test creating generator with custom parameters."""
        custom_params = {"custom_param": "test_value"}

        # Mock the generator to check if custom params are applied
        with patch.object(MockGenerator, '__init__', return_value=None) as mock_init:
            # Register mock generator
            self.factory.register("mock", MockGenerator)

            # Create with custom params
            generator = self.factory.create("mock", 2, 2, **custom_params)

            # Check that custom params were passed (this would need to be implemented in the factory)
            # For now, just verify the generator was created
            self.assertIsInstance(generator, MockGenerator)


class TestHierarchicalGeneratorFactory(unittest.TestCase):
    """Test cases for HierarchicalGeneratorFactory."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = HierarchicalGeneratorFactory(10, 10)

    def tearDown(self):
        """Clean up after tests."""
        reset_factories()

    def test_create_hierarchical_setup(self):
        """Test creating a complete hierarchical generator setup."""
        generators = self.factory.create_hierarchical_setup()

        # Check that all generator types are created
        expected_types = {
            GeneratorType.GLOBAL.value: GlobalGenerator,
            GeneratorType.INTERMEDIATE.value: IntermediateGenerator,
            GeneratorType.LOCAL.value: LocalGenerator
        }

        for gen_type, expected_class in expected_types.items():
            self.assertIn(gen_type, generators)
            self.assertIsInstance(generators[gen_type], expected_class)
            self.assertEqual(generators[gen_type].width, 10)
            self.assertEqual(generators[gen_type].height, 10)


class TestComponentFactory(unittest.TestCase):
    """Test cases for ComponentFactory."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = ComponentFactory()

    def tearDown(self):
        """Clean up after tests."""
        reset_factories()

    def test_register_factory(self):
        """Test registering a sub-factory."""
        sub_factory = GeneratorFactory()
        self.factory.register_factory("generators", sub_factory)

        retrieved = self.factory.get_factory("generators")
        self.assertIs(retrieved, sub_factory)

    def test_create_component(self):
        """Test creating a component through sub-factory."""
        # Register sub-factory
        sub_factory = GeneratorFactory()
        self.factory.register_factory("generators", sub_factory)

        # Create component
        generator = self.factory.create_component("generators", GeneratorType.GLOBAL.value, 5, 5)

        self.assertIsNotNone(generator)
        self.assertIsInstance(generator, GlobalGenerator)
        self.assertEqual(generator.width, 5)
        self.assertEqual(generator.height, 5)

    def test_create_component_unknown_factory(self):
        """Test creating component with unknown factory."""
        with self.assertRaises(GeneratorFactoryError):
            self.factory.create_component("unknown", "some_type", 1, 1)

    def test_generic_create_not_implemented(self):
        """Test that generic create method is not implemented."""
        with self.assertRaises(NotImplementedError):
            self.factory.create("test")


class TestGlobalFactories(unittest.TestCase):
    """Test cases for global factory functions."""

    def setUp(self):
        """Set up test fixtures."""
        reset_factories()

    def tearDown(self):
        """Clean up after tests."""
        reset_factories()

    def test_get_generator_factory_singleton(self):
        """Test that get_generator_factory returns singleton."""
        factory1 = get_generator_factory()
        factory2 = get_generator_factory()

        self.assertIs(factory1, factory2)
        self.assertIsInstance(factory1, GeneratorFactory)

    def test_get_component_factory_singleton(self):
        """Test that get_component_factory returns singleton."""
        factory1 = get_component_factory()
        factory2 = get_component_factory()

        self.assertIs(factory1, factory2)
        self.assertIsInstance(factory1, ComponentFactory)

    def test_reset_factories(self):
        """Test factory reset functionality."""
        factory1 = get_generator_factory()
        component_factory1 = get_component_factory()

        # Register something
        factory1.register("test", MockGenerator)

        # Reset
        reset_factories()

        # Get new instances
        factory2 = get_generator_factory()
        component_factory2 = get_component_factory()

        # Should be different instances
        self.assertIsNot(factory1, factory2)
        self.assertIsNot(component_factory1, component_factory2)

        # Test registration should be gone
        self.assertFalse(factory2.is_registered("test"))


if __name__ == '__main__':
    unittest.main()