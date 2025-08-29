# hamc/plugins/plugin_system.py
"""
Advanced Plugin System for HAMC
Implements dynamic plugin loading, hooks, and extension points.
"""

from typing import Dict, List, Any, Optional, Callable, Type, TypeVar, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import importlib.util
import inspect
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from weakref import WeakSet
import json

T = TypeVar('T')

class PluginState(Enum):
    """Plugin lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class HookPoint(Enum):
    """Standard hook points in the system."""
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"
    PRE_COLLAPSE = "pre_collapse"
    POST_COLLAPSE = "post_collapse"
    PRE_VALIDATION = "pre_validation"
    POST_VALIDATION = "post_validation"
    ON_ERROR = "on_error"
    ON_CONFIG_LOAD = "on_config_load"
    ON_SHUTDOWN = "on_shutdown"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None


@runtime_checkable
class PluginInterface(Protocol):
    """Protocol that all plugins must implement."""
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        ...
    
    def initialize(self, context: 'PluginContext') -> bool:
        """Initialize the plugin."""
        ...
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        ...


@dataclass
class PluginContext:
    """Context provided to plugins during initialization."""
    plugin_manager: 'PluginManager'
    config: Dict[str, Any]
    logger: logging.Logger
    shared_data: Dict[str, Any] = field(default_factory=dict)


class PluginException(Exception):
    """Exception raised by plugin operations."""
    pass


class PluginLoadException(PluginException):
    """Exception raised when plugin loading fails."""
    pass


class PluginDependencyException(PluginException):
    """Exception raised when plugin dependencies are not satisfied."""
    pass


class BasePlugin(ABC):
    """Base class for HAMC plugins."""
    
    def __init__(self):
        self._context: Optional[PluginContext] = None
        self._state = PluginState.UNLOADED
        self._logger: Optional[logging.Logger] = None
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    @property
    def context(self) -> Optional[PluginContext]:
        """Get the plugin context."""
        return self._context
    
    @property
    def logger(self) -> logging.Logger:
        """Get the plugin logger."""
        if self._logger is None:
            self._logger = logging.getLogger(f"plugin.{self.metadata.name}")
        return self._logger
    
    @property
    def state(self) -> PluginState:
        """Get the current plugin state."""
        return self._state
    
    def initialize(self, context: PluginContext) -> bool:
        """Initialize the plugin with context."""
        try:
            self._context = context
            self._state = PluginState.INITIALIZING
            self._logger = context.logger
            
            success = self._on_initialize()
            
            if success:
                self._state = PluginState.ACTIVE
                self.logger.info(f"Plugin {self.metadata.name} initialized successfully")
            else:
                self._state = PluginState.ERROR
                self.logger.error(f"Plugin {self.metadata.name} initialization failed")
            
            return success
            
        except Exception as e:
            self._state = PluginState.ERROR
            if self._logger:
                self._logger.error(f"Plugin {self.metadata.name} initialization error: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        try:
            self._on_shutdown()
            self._state = PluginState.UNLOADED
            self.logger.info(f"Plugin {self.metadata.name} shut down")
        except Exception as e:
            self.logger.error(f"Plugin {self.metadata.name} shutdown error: {e}")
    
    @abstractmethod
    def _on_initialize(self) -> bool:
        """Plugin-specific initialization logic."""
        pass
    
    @abstractmethod
    def _on_shutdown(self) -> None:
        """Plugin-specific shutdown logic."""
        pass


class HookManager:
    """Manages plugin hooks and their execution."""
    
    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {}
        self._logger = logging.getLogger(__name__)
    
    def register_hook(self, hook_point: str, callback: Callable) -> None:
        """Register a callback for a hook point."""
        if hook_point not in self._hooks:
            self._hooks[hook_point] = []
        
        self._hooks[hook_point].append(callback)
        self._logger.debug(f"Registered hook for {hook_point}")
    
    def unregister_hook(self, hook_point: str, callback: Callable) -> None:
        """Unregister a callback from a hook point."""
        if hook_point in self._hooks:
            try:
                self._hooks[hook_point].remove(callback)
                self._logger.debug(f"Unregistered hook for {hook_point}")
            except ValueError:
                self._logger.warning(f"Hook callback not found for {hook_point}")
    
    def execute_hook(self, hook_point: str, *args, **kwargs) -> List[Any]:
        """Execute all callbacks for a hook point."""
        if hook_point not in self._hooks:
            return []
        
        results = []
        for callback in self._hooks[hook_point]:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                self._logger.error(f"Hook execution failed for {hook_point}: {e}")
        
        return results
    
    def has_hooks(self, hook_point: str) -> bool:
        """Check if any hooks are registered for a point."""
        return hook_point in self._hooks and len(self._hooks[hook_point]) > 0
    
    def clear_hooks(self, hook_point: Optional[str] = None) -> None:
        """Clear hooks for a specific point or all points."""
        if hook_point:
            self._hooks.pop(hook_point, None)
        else:
            self._hooks.clear()


class PluginManager:
    """Advanced plugin manager with dependency resolution and lifecycle management."""
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_states: Dict[str, PluginState] = {}
        self._plugin_dirs = plugin_dirs or [str(Path(__file__).parent)]
        self._hook_manager = HookManager()
        self._shared_data: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)
        
        # Plugin search paths
        self._search_paths = [
            Path(d) for d in self._plugin_dirs if Path(d).exists()
        ]
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in search paths."""
        discovered = []
        
        for search_path in self._search_paths:
            if not search_path.exists():
                continue
            
            # Look for plugin files
            for plugin_file in search_path.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue
                
                try:
                    # Try to load plugin metadata
                    if self._load_plugin_metadata(plugin_file):
                        discovered.append(plugin_file.stem)
                        
                except Exception as e:
                    self._logger.warning(f"Failed to load plugin {plugin_file.stem}: {e}")
        
        return discovered
    
    def _load_plugin_metadata(self, plugin_path: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from plugin file."""
        plugin_name = plugin_path.stem

        try:
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec is None or spec.loader is None:
                logging.warning(f"Could not load plugin spec for {plugin_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Extract metadata from module
            return self._extract_metadata_from_module(module, plugin_name)

        except Exception as e:
            logging.error(f"Error loading plugin {plugin_name}: {e}")
            return None
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load and initialize a plugin."""
        if plugin_name not in self._plugins:
            self._logger.error(f"Plugin {plugin_name} not found")
            return False
        
        plugin = self._plugins[plugin_name]
        
        if self._plugin_states[plugin_name] == PluginState.ACTIVE:
            self._logger.warning(f"Plugin {plugin_name} is already active")
            return True
        
        try:
            self._plugin_states[plugin_name] = PluginState.LOADING
            
            # Check dependencies
            if not self._check_dependencies(plugin.metadata):
                raise PluginDependencyException(f"Dependencies not satisfied for {plugin_name}")
            
            # Create plugin context
            plugin_config = config or {}
            context = PluginContext(
                plugin_manager=self,
                config=plugin_config,
                logger=self._logger,
                shared_data=self._shared_data
            )
            
            # Initialize plugin
            self._plugin_states[plugin_name] = PluginState.INITIALIZING
            
            if plugin.initialize(context):
                self._plugin_states[plugin_name] = PluginState.ACTIVE
                
                # Register plugin hooks
                self._register_plugin_hooks(plugin)
                
                self._logger.info(f"Plugin {plugin_name} loaded successfully")
                return True
            else:
                self._plugin_states[plugin_name] = PluginState.ERROR
                return False
                
        except Exception as e:
            self._plugin_states[plugin_name] = PluginState.ERROR
            self._logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    def _check_dependencies(self, metadata: PluginMetadata) -> bool:
        """Check if plugin dependencies are satisfied."""
        for dep in metadata.dependencies:
            if dep not in self._plugins:
                self._logger.error(f"Plugin dependency {dep} not found for {metadata.name}")
                return False
            
            if self._plugin_states.get(dep) != PluginState.ACTIVE:
                self._logger.error(f"Plugin dependency {dep} not active for {metadata.name}")
                return False
        
        return True
    
    def _register_plugin_hooks(self, plugin: BasePlugin) -> None:
        """Register hooks defined by the plugin."""
        metadata = plugin.metadata
        
        for hook_name in metadata.hooks:
            if hasattr(plugin, f'on_{hook_name}'):
                hook_method = getattr(plugin, f'on_{hook_name}')
                self._hook_manager.register_hook(hook_name, hook_method)
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name not in self._plugins:
            return False
        
        plugin = self._plugins[plugin_name]
        
        try:
            # Unregister hooks
            metadata = plugin.metadata
            for hook_name in metadata.hooks:
                if hasattr(plugin, f'on_{hook_name}'):
                    hook_method = getattr(plugin, f'on_{hook_name}')
                    self._hook_manager.unregister_hook(hook_name, hook_method)
            
            # Shutdown plugin
            plugin.shutdown()
            self._plugin_states[plugin_name] = PluginState.UNLOADED
            
            self._logger.info(f"Plugin {plugin_name} unloaded successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def execute_hook(self, hook_point: str, *args, **kwargs) -> List[Any]:
        """Execute a hook point across all active plugins."""
        return self._hook_manager.execute_hook(hook_point, *args, **kwargs)
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(plugin_name)
    
    def get_active_plugins(self) -> List[str]:
        """Get list of active plugin names."""
        return [name for name, state in self._plugin_states.items() 
                if state == PluginState.ACTIVE]
    
    def get_plugin_states(self) -> Dict[str, PluginState]:
        """Get states of all plugins."""
        return self._plugin_states.copy()
    
    def shutdown(self) -> None:
        """Shutdown all plugins."""
        for plugin_name in list(self._plugins.keys()):
            self.unload_plugin(plugin_name)
        
        self._hook_manager.clear_hooks()
        self._shared_data.clear()
    
    def _extract_metadata_from_module(self, module, plugin_name: str) -> Optional[PluginMetadata]:
        """Extract plugin metadata from a loaded module."""
        try:
            # Look for plugin class
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    
                    # Create temporary instance to get metadata
                    temp_plugin = obj()
                    metadata = temp_plugin.metadata
                    
                    # Store plugin for later use
                    self._plugins[plugin_name] = temp_plugin
                    self._plugin_states[plugin_name] = PluginState.UNLOADED
                    
                    self._logger.info(f"Discovered plugin: {metadata.name} v{metadata.version}")
                    return metadata
            
            return None
            
        except Exception as e:
            self._logger.error(f"Error extracting metadata from {plugin_name}: {e}")
            return None


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def reset_plugin_manager():
    """Reset the global plugin manager (mainly for testing)."""
    global _plugin_manager
    if _plugin_manager:
        _plugin_manager.shutdown()
    _plugin_manager = PluginManager()