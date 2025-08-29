# hamc/core/architectural_integration.py
"""
Architectural Integration for HAMC Phase 2
Combines dependency injection, plugin system, factory patterns, and strategy patterns.
"""

from typing import Dict, List, Any, Optional, Type
import logging
from dataclasses import dataclass
from enum import Enum

from .di_container import DependencyContainer, get_container
from ..plugins.plugin_system import PluginManager, get_plugin_manager
from .factory_patterns import GeneratorFactory, get_generator_factory, GeneratorType
from .strategy_patterns import StrategyFactory, AlgorithmConfig, AlgorithmType, SelectionStrategy
from ..config.advanced_config import AdvancedConfig
from .cell import Cell
from .generator_state import GeneratorState

@dataclass
class HAMCArchitecture:
    """Complete HAMC architectural setup."""
    container: DependencyContainer
    plugin_manager: PluginManager
    generator_factory: GeneratorFactory
    strategy_factory: StrategyFactory
    config: AdvancedConfig
    
    def initialize(self) -> bool:
        """Initialize the complete architecture."""
        try:
            # Register core services in DI container
            self._register_core_services()
            
            # Load and initialize plugins
            self._initialize_plugins()
            
            # Set up factory relationships
            self._setup_factory_relationships()
            
            self._logger.info("HAMC Architecture initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize architecture: {e}")
            return False
    
    def _register_core_services(self) -> None:
        """Register core services in dependency injection container."""
        # Register configuration
        self.container.register_instance(AdvancedConfig, self.config)
        
        # Register plugin manager
        self.container.register_instance(PluginManager, self.plugin_manager)
        
        # Register factories
        self.container.register_instance(GeneratorFactory, self.generator_factory)
        self.container.register_instance(StrategyFactory, self.strategy_factory)
        
        # Register logger
        self.container.register_instance(logging.Logger, logging.getLogger('hamc.core'))
    
    def _initialize_plugins(self) -> None:
        """Initialize plugin system."""
        # Discover available plugins
        available_plugins = self.plugin_manager.discover_plugins()
        self._logger.info(f"Discovered {len(available_plugins)} plugins")
        
        # Load core plugins (if any)
        core_plugins = ['hamc.core.plugins.generator_plugin', 'hamc.core.plugins.validation_plugin']
        for plugin_name in core_plugins:
            if plugin_name in available_plugins:
                self.plugin_manager.load_plugin(plugin_name)
    
    def _setup_factory_relationships(self) -> None:
        """Set up relationships between factories."""
        # The factories are already set up, but we could add cross-references here
        pass
    
    @property
    def _logger(self) -> logging.Logger:
        """Get logger for this architecture."""
        return logging.getLogger(__name__)


class HAMCCore:
    """Core HAMC system with integrated architecture."""
    
    def __init__(self, config: AdvancedConfig):
        self._config = config
        self._architecture: Optional[HAMCArchitecture] = None
        self._logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize the HAMC core system."""
        try:
            # Create architectural components
            container = get_container()
            plugin_manager = get_plugin_manager()
            generator_factory = get_generator_factory()
            strategy_factory = StrategyFactory()  # Create directly instead of using getter
            
            # Create complete architecture
            self._architecture = HAMCArchitecture(
                container=container,
                plugin_manager=plugin_manager,
                generator_factory=generator_factory,
                strategy_factory=strategy_factory,
                config=self._config
            )
            
            # Initialize architecture
            if not self._architecture.initialize():
                return False
            
            self._logger.info("HAMC Core initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize HAMC Core: {e}")
            return False
    
    def create_generator(self, generator_type: GeneratorType, width: int, height: int) -> Any:
        """Create a generator using the factory pattern."""
        if not self._architecture:
            raise RuntimeError("HAMC Core not initialized")
        
        return self._architecture.generator_factory.create(
            generator_type.value, width, height
        )
    
    def create_strategy(self, algorithm_config: AlgorithmConfig):
        """Create an algorithm strategy."""
        if not self._architecture:
            raise RuntimeError("HAMC Core not initialized")
        
        return self._architecture.strategy_factory.create_algorithm_strategy(algorithm_config)
    
    def execute_hook(self, hook_point: str, *args, **kwargs) -> List[Any]:
        """Execute a plugin hook."""
        if not self._architecture:
            raise RuntimeError("HAMC Core not initialized")
        
        return self._architecture.plugin_manager.execute_hook(hook_point, *args, **kwargs)
    
    def get_service(self, service_type: Type) -> Any:
        """Get a service from the DI container."""
        if not self._architecture:
            raise RuntimeError("HAMC Core not initialized")
        
        return self._architecture.container.resolve(service_type)
    
    def shutdown(self) -> None:
        """Shutdown the HAMC core system."""
        if self._architecture:
            self._architecture.plugin_manager.shutdown()
            self._logger.info("HAMC Core shut down")


class HAMCFacade:
    """Facade for simplified HAMC usage with enterprise architecture."""
    
    def __init__(self, config: Optional[AdvancedConfig] = None):
        self._core: Optional[HAMCCore] = None
        self._config = config
        self._logger = logging.getLogger(__name__)
    
    def initialize(self, config: Optional[AdvancedConfig] = None) -> bool:
        """Initialize HAMC with enterprise architecture."""
        if config:
            self._config = config
        
        if not self._config:
            raise ValueError("Configuration required for initialization")
        
        self._core = HAMCCore(self._config)
        return self._core.initialize()
    
    def generate_world(self, width: int, height: int, 
                      generator_type: GeneratorType = GeneratorType.GLOBAL,
                      algorithm_config: Optional[AlgorithmConfig] = None) -> Dict[str, Any]:
        """Generate a world using the enterprise architecture."""
        if not self._core:
            raise RuntimeError("HAMC not initialized")
        
        try:
            # Execute pre-generation hooks
            self._core.execute_hook('pre_generation', width=width, height=height)
            
            # Create generator
            generator = self._core.create_generator(generator_type, width, height)
            
            # Create strategy if provided
            if algorithm_config:
                strategy = self._core.create_strategy(algorithm_config)
                # Apply strategy to generator (implementation would depend on generator interface)
            
            # Generate world
            success = generator.collapse()
            
            # Execute post-generation hooks
            self._core.execute_hook('post_generation', success=success, generator=generator)
            
            return {
                'success': success,
                'generator': generator,
                'cells': generator.cells if success else None
            }
            
        except Exception as e:
            # Execute error hook
            self._core.execute_hook('on_error', error=e)
            raise
    
    def get_plugin_manager(self) -> PluginManager:
        """Get the plugin manager."""
        if not self._core or not self._core._architecture:
            raise RuntimeError("HAMC not initialized")
        return self._core._architecture.plugin_manager
    
    def get_container(self) -> DependencyContainer:
        """Get the dependency injection container."""
        if not self._core or not self._core._architecture:
            raise RuntimeError("HAMC not initialized")
        return self._core._architecture.container
    
    def shutdown(self) -> None:
        """Shutdown HAMC."""
        if self._core:
            self._core.shutdown()


# Global facade instance
_hamc_facade: Optional[HAMCFacade] = None


def get_hamc_facade() -> HAMCFacade:
    """Get the global HAMC facade instance."""
    global _hamc_facade
    if _hamc_facade is None:
        _hamc_facade = HAMCFacade()
    return _hamc_facade


def reset_hamc_facade():
    """Reset the global HAMC facade (mainly for testing)."""
    global _hamc_facade
    if _hamc_facade:
        _hamc_facade.shutdown()
    _hamc_facade = None