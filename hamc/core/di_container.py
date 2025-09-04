# hamc/core/di_container.py
"""
Dependency Injection Container for HAMC
Implements advanced dependency injection with lifecycle management and automatic resolution.
"""

from typing import Dict, Type, Any, Optional, Callable, TypeVar, Generic, Union, List
from weakref import WeakKeyDictionary
from threading import Lock
import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum

T = TypeVar('T')

class Scope(Enum):
    """Dependency injection scopes."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class Lifecycle(Enum):
    """Component lifecycle states."""
    REGISTERED = "registered"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    DISPOSED = "disposed"


@dataclass
class ComponentRegistration(Generic[T]):
    """Registration information for a component."""
    service_type: Type[T]
    implementation_type: Type[T]
    scope: Scope = Scope.SINGLETON
    factory: Optional[Callable[[], T]] = None
    instance: Optional[T] = None
    dependencies: List[str] = field(default_factory=list)
    lifecycle: Lifecycle = Lifecycle.REGISTERED
    dispose_method: Optional[str] = None


class DependencyInjectionException(Exception):
    """Exception raised when dependency injection fails."""
    pass


class CircularDependencyException(DependencyInjectionException):
    """Exception raised when circular dependencies are detected."""
    pass


class ServiceNotFoundException(DependencyInjectionException):
    """Exception raised when a service cannot be found."""
    pass


class DependencyContainer:
    """Advanced dependency injection container with lifecycle management."""
    
    def __init__(self):
        self._registrations: Dict[Type, ComponentRegistration] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._lock = Lock()
        self._current_scope: Optional[str] = None
        self._resolving_stack: List[Type] = []
        self.logger = logging.getLogger(__name__)
    
    def register(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None, 
                scope: Scope = Scope.SINGLETON, factory: Optional[Callable[[], T]] = None) -> 'DependencyContainer':
        """Register a service with the container.
        
        Args:
            service_type: The service interface/type
            implementation_type: The concrete implementation (defaults to service_type)
            scope: The lifecycle scope
            factory: Optional factory function
            
        Returns:
            Self for method chaining
        """
        if implementation_type is None:
            implementation_type = service_type
            
        with self._lock:
            registration = ComponentRegistration(
                service_type=service_type,
                implementation_type=implementation_type,
                scope=scope,
                factory=factory
            )
            
            # Analyze dependencies
            registration.dependencies = self._analyze_dependencies(implementation_type)
            
            self._registrations[service_type] = registration
            self.logger.debug(f"Registered {service_type.__name__} -> {implementation_type.__name__} ({scope.value})")
            
        return self
    
    def register_singleton(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> 'DependencyContainer':
        """Register a singleton service."""
        return self.register(service_type, implementation_type, Scope.SINGLETON)
    
    def register_transient(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> 'DependencyContainer':
        """Register a transient service."""
        return self.register(service_type, implementation_type, Scope.TRANSIENT)
    
    def register_factory(self, service_type: Type[T], factory: Callable[[], T], 
                        scope: Scope = Scope.SINGLETON) -> 'DependencyContainer':
        """Register a service with a factory function."""
        return self.register(service_type, None, scope, factory)
    
    def register_instance(self, service_type: Type[T], instance: T) -> 'DependencyContainer':
        """Register a pre-created instance as a singleton."""
        with self._lock:
            registration = ComponentRegistration(
                service_type=service_type,
                implementation_type=type(instance),
                scope=Scope.SINGLETON,
                instance=instance
            )
            self._registrations[service_type] = registration
            self.logger.debug(f"Registered instance {service_type.__name__} -> {type(instance).__name__}")
            
        return self
    
    def resolve(self, service_type: Type[T], scope_name: Optional[str] = None) -> T:
        """Resolve a service from the container.
        
        Args:
            service_type: The service type to resolve
            scope_name: Optional scope name for scoped services
            
        Returns:
            The resolved service instance
            
        Raises:
            ServiceNotFoundException: If service is not registered
            CircularDependencyException: If circular dependency is detected
        """
        with self._lock:
            # Check for circular dependency
            if service_type in self._resolving_stack:
                raise CircularDependencyException(
                    f"Circular dependency detected: {' -> '.join([t.__name__ for t in self._resolving_stack])} -> {service_type.__name__}"
                )
            
            if service_type not in self._registrations:
                raise ServiceNotFoundException(f"Service {service_type.__name__} is not registered")
            
            registration = self._registrations[service_type]
            
            # Check if already resolving
            if registration.lifecycle == Lifecycle.RESOLVING:
                raise CircularDependencyException(f"Service {service_type.__name__} is already being resolved")
            
            # Return existing instance for singletons
            if registration.scope == Scope.SINGLETON and registration.instance is not None:
                return registration.instance
            
            # Return scoped instance
            if registration.scope == Scope.SCOPED and scope_name:
                scoped_instances = self._scoped_instances.get(scope_name, {})
                if service_type in scoped_instances:
                    return scoped_instances[service_type]
            
            # Mark as resolving
            registration.lifecycle = Lifecycle.RESOLVING
            self._resolving_stack.append(service_type)
            
            try:
                # Create instance
                instance = self._create_instance(registration, scope_name)
                
                # Mark as resolved
                registration.lifecycle = Lifecycle.RESOLVED
                
                # Store singleton instance
                if registration.scope == Scope.SINGLETON:
                    registration.instance = instance
                
                # Store scoped instance
                elif registration.scope == Scope.SCOPED and scope_name:
                    if scope_name not in self._scoped_instances:
                        self._scoped_instances[scope_name] = {}
                    self._scoped_instances[scope_name][service_type] = instance
                
                return instance
                
            finally:
                self._resolving_stack.pop()
    
    def _create_instance(self, registration: ComponentRegistration[T], scope_name: Optional[str]) -> T:
        """Create an instance of the service."""
        if registration.instance is not None:
            return registration.instance
        
        if registration.factory is not None:
            return registration.factory()
        
        # Get constructor parameters
        init_signature = inspect.signature(registration.implementation_type.__init__)
        init_params = {}
        
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
                
            # Try to resolve dependency
            if param.annotation != inspect.Parameter.empty:
                try:
                    target = self._normalize_annotation(param.annotation)
                    if isinstance(target, str):
                        # Find by registered service name
                        match = next((st for st in self._registrations.keys() if st.__name__ == target), None)
                        if match is None:
                            raise ServiceNotFoundException(target)
                        init_params[param_name] = self.resolve(match, scope_name)
                    else:
                        init_params[param_name] = self.resolve(target, scope_name)
                except ServiceNotFoundException:
                    # If dependency not found, use default value or None
                    if param.default != inspect.Parameter.empty:
                        init_params[param_name] = param.default
                    else:
                        init_params[param_name] = None
            else:
                # No type annotation, use default or None
                if param.default != inspect.Parameter.empty:
                    init_params[param_name] = param.default
                else:
                    init_params[param_name] = None
        
        return registration.implementation_type(**init_params)

    def _normalize_annotation(self, annotation: Any) -> Any:
        """Normalize typing annotations: handle Optional[T] and forward refs."""
        try:
            from typing import get_origin, get_args
            origin = get_origin(annotation)
            if origin is None:
                # Forward ref as string
                if isinstance(annotation, str):
                    return annotation
                return annotation
            # Optional[T] or Union[T, None]
            args = [a for a in get_args(annotation) if a is not type(None)]  # noqa: E721
            return args[0] if args else annotation
        except Exception:
            return annotation
    
    def _analyze_dependencies(self, implementation_type: Type) -> List[str]:
        """Analyze the dependencies of an implementation type."""
        if not hasattr(implementation_type, '__init__'):
            return []
        
        init_signature = inspect.signature(implementation_type.__init__)
        dependencies = []
        
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
                
            if param.annotation != inspect.Parameter.empty:
                ann = self._normalize_annotation(param.annotation)
                dependencies.append(ann if isinstance(ann, str) else getattr(ann, '__name__', str(ann)))
        
        return dependencies
    
    def begin_scope(self, scope_name: str) -> 'ScopedContainer':
        """Begin a new scope for scoped services."""
        return ScopedContainer(self, scope_name)
    
    def dispose(self, service_type: Optional[Type] = None):
        """Dispose of services and clean up resources."""
        with self._lock:
            if service_type:
                if service_type in self._registrations:
                    registration = self._registrations[service_type]
                    if registration.instance and hasattr(registration.instance, 'dispose'):
                        registration.instance.dispose()
                    registration.instance = None
                    registration.lifecycle = Lifecycle.DISPOSED
            else:
                # Dispose all services
                for registration in self._registrations.values():
                    if registration.instance and hasattr(registration.instance, 'dispose'):
                        registration.instance.dispose()
                    registration.instance = None
                    registration.lifecycle = Lifecycle.DISPOSED
                
                self._scoped_instances.clear()
    
    def get_registered_services(self) -> Dict[Type, ComponentRegistration]:
        """Get all registered services."""
        return self._registrations.copy()
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._registrations


class ScopedContainer:
    """Container for managing scoped service instances."""
    
    def __init__(self, container: DependencyContainer, scope_name: str):
        self.container = container
        self.scope_name = scope_name
        self.container._current_scope = scope_name
        
        if scope_name not in self.container._scoped_instances:
            self.container._scoped_instances[scope_name] = {}
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service within this scope."""
        return self.container.resolve(service_type, self.scope_name)
    
    def dispose(self):
        """Dispose of all scoped instances."""
        if self.scope_name in self.container._scoped_instances:
            scoped_instances = self.container._scoped_instances[self.scope_name]
            
            # Dispose instances that have dispose methods
            for instance in scoped_instances.values():
                if hasattr(instance, 'dispose'):
                    instance.dispose()
            
            del self.container._scoped_instances[self.scope_name]
        
        if self.container._current_scope == self.scope_name:
            self.container._current_scope = None


# Global container instance
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Get the global dependency injection container."""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container


def reset_container():
    """Reset the global container (mainly for testing)."""
    global _container
    if _container:
        _container.dispose()
    _container = DependencyContainer()


# Convenience functions
def register(service_type: Type[T], implementation_type: Optional[Type[T]] = None, 
            scope: Scope = Scope.SINGLETON) -> DependencyContainer:
    """Register a service with the global container."""
    return get_container().register(service_type, implementation_type, scope)


def register_singleton(service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> DependencyContainer:
    """Register a singleton service with the global container."""
    return get_container().register_singleton(service_type, implementation_type)


def register_transient(service_type: Type[T], implementation_type: Optional[Type[T]] = None) -> DependencyContainer:
    """Register a transient service with the global container."""
    return get_container().register_transient(service_type, implementation_type)


def resolve(service_type: Type[T]) -> T:
    """Resolve a service from the global container."""
    return get_container().resolve(service_type)


def begin_scope(scope_name: str) -> ScopedContainer:
    """Begin a new scope with the global container."""
    return get_container().begin_scope(scope_name)
