# tests/test_plugin_system.py
"""
Tests for Plugin System
Tests the advanced plugin system with dynamic loading and hooks.
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path
from hamc.plugins.plugin_system import (
    PluginManager, BasePlugin, PluginMetadata, PluginContext,
    PluginState, HookPoint, get_plugin_manager, reset_plugin_manager
)


class TestPlugin(BasePlugin):
    """Test plugin for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.shutdown_called = False
        self.hook_called = False
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            hooks=["pre_generation", "post_generation"]
        )
    
    def _on_initialize(self) -> bool:
        self.initialized = True
        return True
    
    def _on_shutdown(self) -> None:
        self.shutdown_called = True
    
    def on_pre_generation(self, **kwargs):
        self.hook_called = True
        return "pre_generation_result"
    
    def on_post_generation(self, **kwargs):
        return "post_generation_result"


class FailingPlugin(BasePlugin):
    """Plugin that fails to initialize."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="failing_plugin",
            version="1.0.0",
            description="Failing test plugin",
            author="Test Author"
        )
    
    def _on_initialize(self) -> bool:
        return False
    
    def _on_shutdown(self) -> None:
        pass


class TestPluginManager(unittest.TestCase):
    """Test cases for PluginManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.plugin_manager = PluginManager([self.temp_dir])
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        reset_plugin_manager()
    
    def create_test_plugin_file(self, plugin_name: str, plugin_code: str):
        """Create a test plugin file."""
        plugin_path = Path(self.temp_dir) / f"{plugin_name}.py"
        plugin_path.write_text(plugin_code)
        return plugin_path
    
    def test_plugin_discovery(self):
        """Test plugin discovery in search paths."""
        # Create a test plugin file
        plugin_code = '''
from hamc.plugins.plugin_system import BasePlugin, PluginMetadata

class TestDiscoveredPlugin(BasePlugin):
    @property
    def metadata(self):
        return PluginMetadata(
            name="test_discovered",
            version="1.0.0",
            description="Test discovered plugin",
            author="Test Author"
        )
    
    def _on_initialize(self):
        return True
    
    def _on_shutdown(self):
        pass
'''
        self.create_test_plugin_file("test_discovered", plugin_code)
        
        # Discover plugins
        discovered = self.plugin_manager.discover_plugins()
        
        self.assertIn("test_discovered", discovered)
    
    def test_plugin_loading_and_initialization(self):
        """Test loading and initializing a plugin."""
        # Create a test plugin instance
        test_plugin = TestPlugin()
        self.plugin_manager._plugins["test_plugin"] = test_plugin
        self.plugin_manager._plugin_states["test_plugin"] = PluginState.UNLOADED
        
        # Load plugin
        success = self.plugin_manager.load_plugin("test_plugin")
        
        self.assertTrue(success)
        self.assertTrue(test_plugin.initialized)
        self.assertEqual(self.plugin_manager._plugin_states["test_plugin"], PluginState.ACTIVE)
    
    def test_plugin_unloading(self):
        """Test unloading a plugin."""
        # Create and load a test plugin
        test_plugin = TestPlugin()
        self.plugin_manager._plugins["test_plugin"] = test_plugin
        self.plugin_manager._plugin_states["test_plugin"] = PluginState.ACTIVE
        
        # Unload plugin
        success = self.plugin_manager.unload_plugin("test_plugin")
        
        self.assertTrue(success)
        self.assertTrue(test_plugin.shutdown_called)
        self.assertEqual(self.plugin_manager._plugin_states["test_plugin"], PluginState.UNLOADED)
    
    def test_hook_execution(self):
        """Test hook execution across plugins."""
        # Create a test plugin and add it to the manager
        test_plugin = TestPlugin()
        self.plugin_manager._plugins["test_plugin"] = test_plugin
        self.plugin_manager._plugin_states["test_plugin"] = PluginState.UNLOADED
        
        # Load plugin (this should register hooks)
        success = self.plugin_manager.load_plugin("test_plugin")
        self.assertTrue(success)
        
        # Execute hook
        results = self.plugin_manager.execute_hook("pre_generation", width=10, height=10)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "pre_generation_result")
        self.assertTrue(test_plugin.hook_called)
    
    def test_plugin_with_dependencies(self):
        """Test plugin with dependencies."""
        # Create plugin with dependencies
        plugin_metadata = PluginMetadata(
            name="dependent_plugin",
            version="1.0.0",
            description="Plugin with dependencies",
            author="Test Author",
            dependencies=["nonexistent_plugin"]
        )
        
        # This should fail due to missing dependency
        success = self.plugin_manager.load_plugin("dependent_plugin")
        self.assertFalse(success)
    
    def test_get_active_plugins(self):
        """Test getting list of active plugins."""
        # Create and activate a plugin
        test_plugin = TestPlugin()
        self.plugin_manager._plugins["test_plugin"] = test_plugin
        self.plugin_manager._plugin_states["test_plugin"] = PluginState.ACTIVE
        
        active_plugins = self.plugin_manager.get_active_plugins()
        
        self.assertIn("test_plugin", active_plugins)
    
    def test_get_plugin_states(self):
        """Test getting plugin states."""
        # Create a plugin
        test_plugin = TestPlugin()
        self.plugin_manager._plugins["test_plugin"] = test_plugin
        self.plugin_manager._plugin_states["test_plugin"] = PluginState.ACTIVE
        
        states = self.plugin_manager.get_plugin_states()
        
        self.assertEqual(states["test_plugin"], PluginState.ACTIVE)


class TestBasePlugin(unittest.TestCase):
    """Test cases for BasePlugin."""
    
    def test_plugin_initialization_success(self):
        """Test successful plugin initialization."""
        plugin = TestPlugin()
        context = PluginContext(
            plugin_manager=Mock(),
            config={},
            logger=Mock(),
            shared_data={}
        )
        
        success = plugin.initialize(context)
        
        self.assertTrue(success)
        self.assertTrue(plugin.initialized)
        self.assertEqual(plugin.state, PluginState.ACTIVE)
    
    def test_plugin_initialization_failure(self):
        """Test plugin initialization failure."""
        plugin = FailingPlugin()
        context = PluginContext(
            plugin_manager=Mock(),
            config={},
            logger=Mock(),
            shared_data={}
        )
        
        success = plugin.initialize(context)
        
        self.assertFalse(success)
        self.assertEqual(plugin.state, PluginState.ERROR)


class TestPluginMetadata(unittest.TestCase):
    """Test cases for PluginMetadata."""
    
    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin description",
            author="Test Author",
            dependencies=["dep1", "dep2"],
            hooks=["hook1", "hook2"]
        )
        
        self.assertEqual(metadata.name, "test_plugin")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.description, "Test plugin description")
        self.assertEqual(metadata.author, "Test Author")
        self.assertEqual(metadata.dependencies, ["dep1", "dep2"])
        self.assertEqual(metadata.hooks, ["hook1", "hook2"])


class TestGlobalPluginManager(unittest.TestCase):
    """Test cases for global plugin manager functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        reset_plugin_manager()
    
    def tearDown(self):
        """Clean up after tests."""
        reset_plugin_manager()
    
    def test_get_plugin_manager_singleton(self):
        """Test that get_plugin_manager returns singleton."""
        manager1 = get_plugin_manager()
        manager2 = get_plugin_manager()
        
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, PluginManager)
    
    def test_reset_plugin_manager(self):
        """Test plugin manager reset functionality."""
        manager1 = get_plugin_manager()
        manager1._plugins["test"] = Mock()
        
        # Reset manager
        reset_plugin_manager()
        manager2 = get_plugin_manager()
        
        # Should be different instances
        self.assertIsNot(manager1, manager2)
        self.assertNotIn("test", manager2._plugins)


if __name__ == '__main__':
    unittest.main()