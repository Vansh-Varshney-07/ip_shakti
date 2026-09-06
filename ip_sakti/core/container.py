"""
Dependency Injection Container for IP-SAKTI RAG system.
Uses the new modular interfaces from ip_sakti.interfaces.
"""
from typing import Any, Dict, Type, Callable, Optional, TypeVar, Generic, List
from abc import ABC, abstractmethod
import inspect
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime scopes."""
    SINGLETON = "singleton"      # One instance per container
    TRANSIENT = "transient"      # New instance each resolution
    SCOPED = "scoped"            # One instance per scope (request)


@dataclass
class ServiceDescriptor:
    """Describes a registered service."""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    dependencies: Dict[str, Type] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class IServiceProvider(ABC):
    """Service provider interface for dependency resolution."""
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> T:
        """Get service instance."""
        pass
    
    @abstractmethod
    def get_required_service(self, service_type: Type[T]) -> T:
        """Get required service instance, raises if not found."""
        pass
    
    @abstractmethod
    def get_services(self, service_type: Type[T]) -> List[T]:
        """Get all services of a type (for multiple implementations)."""
        pass
    
    @abstractmethod
    def create_scope(self) -> 'IServiceScope':
        """Create a new service scope."""
        pass


class IServiceScope(ABC):
    """Service scope interface for scoped lifetime management."""
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> T:
        """Get service instance within scope."""
        pass
    
    @abstractmethod
    def get_required_service(self, service_type: Type[T]) -> T:
        """Get required service instance within scope."""
        pass
    
    @abstractmethod
    async def dispose_async(self) -> None:
        """Dispose scope asynchronously."""
        pass


class ServiceCollection:
    """Service collection for registering dependencies."""
    
    def __init__(self):
        self._descriptors: Dict[Type, ServiceDescriptor] = {}
        self._multi_descriptors: Dict[Type, List[ServiceDescriptor]] = {}
    
    def add_singleton(
        self, 
        service_type: Type[T], 
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        instance: Optional[T] = None,
        tags: Optional[List[str]] = None
    ) -> 'ServiceCollection':
        """Register a singleton service."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON,
            tags=tags or []
        )
        self._descriptors[service_type] = descriptor
        return self
    
    def add_transient(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        tags: Optional[List[str]] = None
    ) -> 'ServiceCollection':
        """Register a transient service."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=ServiceLifetime.TRANSIENT,
            tags=tags or []
        )
        self._descriptors[service_type] = descriptor
        return self
    
    def add_scoped(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        tags: Optional[List[str]] = None
    ) -> 'ServiceCollection':
        """Register a scoped service."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=ServiceLifetime.SCOPED,
            tags=tags or []
        )
        self._descriptors[service_type] = descriptor
        return self
    
    def add_singleton_multi(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        instance: Optional[T] = None,
        tags: Optional[List[str]] = None
    ) -> 'ServiceCollection':
        """Register a singleton service (multiple implementations allowed)."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON,
            tags=tags or []
        )
        if service_type not in self._multi_descriptors:
            self._multi_descriptors[service_type] = []
        self._multi_descriptors[service_type].append(descriptor)
        return self
    
    def add_transient_multi(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        tags: Optional[List[str]] = None
    ) -> 'ServiceCollection':
        """Register a transient service (multiple implementations allowed)."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=ServiceLifetime.TRANSIENT,
            tags=tags or []
        )
        if service_type not in self._multi_descriptors:
            self._multi_descriptors[service_type] = []
        self._multi_descriptors[service_type].append(descriptor)
        return self
    
    def try_add_singleton(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        instance: Optional[T] = None
    ) -> 'ServiceCollection':
        """Register singleton only if not already registered."""
        if service_type not in self._descriptors:
            return self.add_singleton(service_type, implementation_type, factory, instance)
        return self
    
    def build(self) -> 'ServiceProvider':
        """Build the service provider."""
        return ServiceProvider(
            self._descriptors.copy(),
            {k: v.copy() for k, v in self._multi_descriptors.items()}
        )
    
    def get_descriptor(self, service_type: Type) -> Optional[ServiceDescriptor]:
        """Get service descriptor."""
        return self._descriptors.get(service_type)
    
    def get_multi_descriptors(self, service_type: Type) -> List[ServiceDescriptor]:
        """Get all descriptors for a service type."""
        return self._multi_descriptors.get(service_type, [])


class ServiceScope(IServiceScope):
    """Service scope implementation."""
    
    def __init__(self, provider: 'ServiceProvider'):
        self._provider = provider
        self._scoped_instances: Dict[Type, Any] = {}
        self._scoped_multi_instances: Dict[Type, List[Any]] = {}
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get service instance within scope."""
        descriptor = self._provider._descriptors.get(service_type)
        if not descriptor:
            raise KeyError(f"Service {service_type.__name__} not registered")
        
        if descriptor.lifetime == ServiceLifetime.SCOPED:
            if service_type not in self._scoped_instances:
                self._scoped_instances[service_type] = self._provider._create_instance(descriptor, self)
            return self._scoped_instances[service_type]
        
        return self._provider.get_service(service_type)
    
    def get_required_service(self, service_type: Type[T]) -> T:
        """Get required service instance within scope."""
        try:
            return self.get_service(service_type)
        except KeyError:
            raise KeyError(f"Required service {service_type.__name__} not registered")
    
    async def dispose_async(self) -> None:
        """Dispose scoped instances."""
        # Dispose single instances
        for instance in self._scoped_instances.values():
            await self._dispose_instance(instance)
        
        # Dispose multi instances
        for instances in self._scoped_multi_instances.values():
            for instance in instances:
                await self._dispose_instance(instance)
        
        self._scoped_instances.clear()
        self._scoped_multi_instances.clear()
    
    async def _dispose_instance(self, instance: Any) -> None:
        """Dispose a single instance."""
        if hasattr(instance, 'dispose_async') and callable(instance.dispose_async):
            try:
                await instance.dispose_async()
            except Exception as e:
                logger.warning(f"Error disposing {type(instance).__name__}: {e}")
        elif hasattr(instance, 'close') and callable(instance.close):
            try:
                if inspect.iscoroutinefunction(instance.close):
                    await instance.close()
                else:
                    instance.close()
            except Exception as e:
                logger.warning(f"Error closing {type(instance).__name__}: {e}")


class ServiceProvider(IServiceProvider):
    """Service provider implementation with dependency injection."""
    
    def __init__(
        self, 
        descriptors: Dict[Type, ServiceDescriptor],
        multi_descriptors: Dict[Type, List[ServiceDescriptor]]
    ):
        self._descriptors = descriptors
        self._multi_descriptors = multi_descriptors
        self._singletons: Dict[Type, Any] = {}
        self._singleton_multi: Dict[Type, List[Any]] = {}
        self._building: set = set()  # For circular dependency detection
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get service instance."""
        descriptor = self._descriptors.get(service_type)
        if not descriptor:
            raise KeyError(f"Service {service_type.__name__} not registered")
        
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type not in self._singletons:
                self._singletons[service_type] = self._create_instance(descriptor, self)
            return self._singletons[service_type]
        
        # Transient - new instance each time
        return self._create_instance(descriptor, self)
    
    def get_required_service(self, service_type: Type[T]) -> T:
        """Get required service instance."""
        try:
            return self.get_service(service_type)
        except KeyError:
            raise KeyError(f"Required service {service_type.__name__} not registered")
    
    def get_services(self, service_type: Type[T]) -> List[T]:
        """Get all services of a type."""
        results = []
        
        # Single registration
        if service_type in self._descriptors:
            results.append(self.get_service(service_type))
        
        # Multi registrations
        if service_type in self._multi_descriptors:
            if service_type not in self._singleton_multi:
                self._singleton_multi[service_type] = [
                    self._create_instance(desc, self) 
                    for desc in self._multi_descriptors[service_type]
                ]
            results.extend(self._singleton_multi[service_type])
        
        return results
    
    def create_scope(self) -> IServiceScope:
        """Create a new service scope."""
        return ServiceScope(self)
    
    def _create_instance(self, descriptor: ServiceDescriptor, scope: IServiceScope) -> Any:
        """Create service instance with dependency injection."""
        # Check for circular dependency
        if descriptor.service_type in self._building:
            raise RuntimeError(f"Circular dependency detected for {descriptor.service_type.__name__}")
        
        self._building.add(descriptor.service_type)
        try:
            # Use existing instance if provided
            if descriptor.instance is not None:
                return descriptor.instance
            
            # Use factory if provided
            if descriptor.factory is not None:
                return self._invoke_factory(descriptor.factory, scope)
            
            # Use implementation type
            impl_type = descriptor.implementation_type or descriptor.service_type
            return self._create_from_type(impl_type, scope)
        finally:
            self._building.discard(descriptor.service_type)
    
    def _invoke_factory(self, factory: Callable, scope: IServiceScope) -> Any:
        """Invoke factory with dependency injection."""
        sig = inspect.signature(factory)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = scope.get_service(param.annotation)
                except KeyError:
                    if param.default == inspect.Parameter.empty:
                        raise RuntimeError(
                            f"Cannot resolve dependency {param.annotation.__name__} "
                            f"for factory parameter {param_name}"
                        )
        
        # Only call with arguments that the factory accepts
        if kwargs:
            return factory(**kwargs)
        else:
            return factory()
    
    def _create_from_type(self, impl_type: Type, scope: IServiceScope) -> Any:
        """Create instance from type with constructor injection."""
        # Get constructor
        init = getattr(impl_type, '__init__', None)
        if not init or init is object.__init__:
            return impl_type()
        
        sig = inspect.signature(init)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = scope.get_service(param.annotation)
                except KeyError:
                    if param.default == inspect.Parameter.empty:
                        raise RuntimeError(
                            f"Cannot resolve dependency {param.annotation.__name__} "
                            f"for {impl_type.__name__}.{param_name}"
                        )
        
        return impl_type(**kwargs)


# =============================================================================
# Application Builder
# =============================================================================

class ApplicationBuilder:
    """Application builder for configuring services and middleware."""
    
    def __init__(self):
        self._services = ServiceCollection()
        self._middleware: list = []
        self._startup_tasks: list = []
        self._shutdown_tasks: list = []
        self._configure_services: Optional[Callable] = None
    
    @property
    def services(self) -> ServiceCollection:
        """Get service collection."""
        return self._services
    
    def configure(self, configure_func: Callable[['ApplicationBuilder'], None]) -> 'ApplicationBuilder':
        """Configure services using a function."""
        self._configure_services = configure_func
        return self
    
    def add_middleware(self, middleware: Callable) -> 'ApplicationBuilder':
        """Add middleware."""
        self._middleware.append(middleware)
        return self
    
    def add_startup_task(self, task: Callable) -> 'ApplicationBuilder':
        """Add startup task."""
        self._startup_tasks.append(task)
        return self
    
    def add_shutdown_task(self, task: Callable) -> 'ApplicationBuilder':
        """Add shutdown task."""
        self._shutdown_tasks.append(task)
        return self
    
    def build(self) -> 'Application':
        """Build the application."""
        # Apply service configuration if provided
        if self._configure_services:
            self._configure_services(self)
        
        provider = self._services.build()
        return Application(provider, self._middleware, self._startup_tasks, self._shutdown_tasks)


class Application:
    """Application with lifecycle management."""
    
    def __init__(
        self,
        provider: ServiceProvider,
        middleware: list,
        startup_tasks: list,
        shutdown_tasks: list
    ):
        self._provider = provider
        self._middleware = middleware
        self._startup_tasks = startup_tasks
        self._shutdown_tasks = shutdown_tasks
        self._running = False
    
    @property
    def services(self) -> ServiceProvider:
        """Get service provider."""
        return self._provider
    
    async def start(self) -> None:
        """Start the application."""
        if self._running:
            return
        
        logger.info("Starting application...")
        
        # Run startup tasks
        for task in self._startup_tasks:
            if asyncio.iscoroutinefunction(task):
                await task(self._provider)
            else:
                task(self._provider)
        
        self._running = True
        logger.info("Application started")
    
    async def stop(self) -> None:
        """Stop the application."""
        if not self._running:
            return
        
        logger.info("Stopping application...")
        
        # Run shutdown tasks
        for task in reversed(self._shutdown_tasks):
            if asyncio.iscoroutinefunction(task):
                await task(self._provider)
            else:
                task(self._provider)
        
        self._running = False
        logger.info("Application stopped")
    
    @asynccontextmanager
    async def lifespan(self):
        """Application lifespan context manager."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_application() -> ApplicationBuilder:
    """Create a new application builder."""
    return ApplicationBuilder()


def register_core_services(services: ServiceCollection) -> ServiceCollection:
    """Register core framework services."""
    from ip_sakti.config.loader import Settings, get_settings
    
    # Register configuration
    services.add_singleton(Settings, instance=get_settings())
    
    return services


def register_core_services_builder(builder: ApplicationBuilder) -> ApplicationBuilder:
    """Register core framework services on an ApplicationBuilder."""
    from ip_sakti.config.loader import Settings, get_settings
    
    # Register configuration
    builder.services.add_singleton(Settings, instance=get_settings())
    
    return builder


# =============================================================================
# Service Locator (for backward compatibility)
# =============================================================================

class ServiceLocator:
    """Service locator pattern for accessing services globally."""
    
    _provider: Optional[ServiceProvider] = None
    
    @classmethod
    def set_provider(cls, provider: ServiceProvider) -> None:
        """Set the global service provider."""
        cls._provider = provider
    
    @classmethod
    def get_provider(cls) -> ServiceProvider:
        """Get the global service provider."""
        if cls._provider is None:
            raise RuntimeError("Service provider not initialized. Call ServiceLocator.set_provider() first.")
        return cls._provider
    
    @classmethod
    def get_service(cls, service_type: Type[T]) -> T:
        """Get service from global provider."""
        return cls.get_provider().get_service(service_type)
    
    @classmethod
    def get_required_service(cls, service_type: Type[T]) -> T:
        """Get required service from global provider."""
        return cls.get_provider().get_required_service(service_type)
    
    @classmethod
    def get_services(cls, service_type: Type[T]) -> List[T]:
        """Get all services of a type from global provider."""
        return cls.get_provider().get_services(service_type)


# =============================================================================
# Decorators for Easy Registration
# =============================================================================

def singleton(service_type: Type[T] = None, tags: List[str] = None):
    """Decorator to register a class as singleton."""
    def decorator(cls: Type[T]) -> Type[T]:
        cls._di_lifetime = ServiceLifetime.SINGLETON
        cls._di_service_type = service_type or cls
        cls._di_tags = tags or []
        return cls
    return decorator


def transient(service_type: Type[T] = None, tags: List[str] = None):
    """Decorator to register a class as transient."""
    def decorator(cls: Type[T]) -> Type[T]:
        cls._di_lifetime = ServiceLifetime.TRANSIENT
        cls._di_service_type = service_type or cls
        cls._di_tags = tags or []
        return cls
    return decorator


def scoped(service_type: Type[T] = None, tags: List[str] = None):
    """Decorator to register a class as scoped."""
    def decorator(cls: Type[T]) -> Type[T]:
        cls._di_lifetime = ServiceLifetime.SCOPED
        cls._di_service_type = service_type or cls
        cls._di_tags = tags or []
        return cls
    return decorator


def inject(func: Callable) -> Callable:
    """Decorator to inject dependencies into a function."""
    sig = inspect.signature(func)
    
    async def async_wrapper(*args, **kwargs):
        provider = ServiceLocator.get_provider()
        scope = provider.create_scope()
        try:
            # Resolve dependencies
            for param_name, param in sig.parameters.items():
                if param_name not in kwargs and param.annotation != inspect.Parameter.empty:
                    try:
                        kwargs[param_name] = scope.get_service(param.annotation)
                    except KeyError:
                        if param.default == inspect.Parameter.empty:
                            raise
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        finally:
            await scope.dispose_async()
    
    def sync_wrapper(*args, **kwargs):
        provider = ServiceLocator.get_provider()
        scope = provider.create_scope()
        try:
            # Resolve dependencies
            for param_name, param in sig.parameters.items():
                if param_name not in kwargs and param.annotation != inspect.Parameter.empty:
                    try:
                        kwargs[param_name] = scope.get_service(param.annotation)
                    except KeyError:
                        if param.default == inspect.Parameter.empty:
                            raise
            
            return func(*args, **kwargs)
        finally:
            # For sync, we can't await dispose_async, so we just clear
            scope._scoped_instances.clear()
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper