"""
Phase 10: Production Hardening - Monitoring & Health Checks
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
import asyncio
import time
import logging
import psutil
import threading
from collections import defaultdict, deque
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class HealthCheck:
    name: str
    check_func: Callable[[], Any]
    timeout: float = 5.0
    critical: bool = True
    interval: float = 30.0
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    last_error: Optional[str] = None


@dataclass
class HealthReport:
    status: HealthStatus
    checks: Dict[str, bool]
    errors: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._last_report: Optional[HealthReport] = None
        self._lock = threading.RLock()

    def register_check(self, check: HealthCheck) -> None:
        with self._lock:
            self._checks[check.name] = check

    def unregister_check(self, name: str) -> None:
        with self._lock:
            self._checks.pop(name, None)

    async def run_checks(self) -> HealthReport:
        checks = {}
        errors = {}
        critical_failed = False
        degraded = False

        with self._lock:
            check_list = list(self._checks.values())

        for check in check_list:
            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, check.check_func),
                    timeout=check.timeout
                )
                checks[check.name] = bool(result)
                check.last_result = bool(result)
                check.last_check = datetime.now()
                check.last_error = None

                if not result and check.critical:
                    critical_failed = True
                elif not result:
                    degraded = True

            except Exception as e:
                checks[check.name] = False
                errors[check.name] = str(e)
                check.last_result = False
                check.last_check = datetime.now()
                check.last_error = str(e)

                if check.critical:
                    critical_failed = True
                else:
                    degraded = True

        if critical_failed:
            status = HealthStatus.UNHEALTHY
        elif degraded:
            status = HealthStatus.DEGRADED
        elif checks:
            status = HealthStatus.HEALTHY
        else:
            status = HealthStatus.UNKNOWN

        report = HealthReport(
            status=status,
            checks=checks,
            errors=errors,
            timestamp=datetime.now()
        )

        with self._lock:
            self._last_report = report

        return report

    def get_last_report(self) -> Optional[HealthReport]:
        with self._lock:
            return self._last_report


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0,
        excluded_exceptions: tuple = ()
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.excluded_exceptions = excluded_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time and (time.time() - self._last_failure_time) >= self.timeout:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    async def call(self, func: Callable, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(None, func, *args, **kwargs)

            with self._lock:
                self._on_success()

            return result

        except self.excluded_exceptions:
            raise
        except Exception as e:
            with self._lock:
                self._on_failure()
            raise

    def _on_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        else:
            self._failure_count = 0

    def _on_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._success_count = 0
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN


class CircuitBreakerOpenError(Exception):
    pass


class CircuitBreakerRegistry:
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def register(self, breaker: CircuitBreaker) -> None:
        with self._lock:
            self._breakers[breaker.name] = breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        with self._lock:
            return self._breakers.get(name)

    def get_all_states(self) -> Dict[str, CircuitState]:
        with self._lock:
            return {name: b.state for name, b in self._breakers.items()}


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: deque = deque()
        self._lock = threading.RLock()

    def acquire(self) -> bool:
        with self._lock:
            now = time.time()
            while self._requests and self._requests[0] <= now - self.window_seconds:
                self._requests.popleft()

            if len(self._requests) < self.max_requests:
                self._requests.append(now)
                return True
            return False

    async def acquire_async(self) -> bool:
        return await asyncio.get_event_loop().run_in_executor(None, self.acquire)

    def get_current_usage(self) -> int:
        with self._lock:
            now = time.time()
            while self._requests and self._requests[0] <= now - self.window_seconds:
                self._requests.popleft()
            return len(self._requests)


class RateLimiterRegistry:
    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.RLock()

    def register(self, name: str, max_requests: int, window_seconds: float) -> RateLimiter:
        with self._lock:
            limiter = RateLimiter(max_requests, window_seconds)
            self._limiters[name] = limiter
            return limiter

    def get(self, name: str) -> Optional[RateLimiter]:
        with self._lock:
            return self._limiters.get(name)


@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    def __init__(self, max_points: int = 10000):
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self._lock = threading.RLock()

    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        point = MetricPoint(name=name, value=value, labels=labels or {})
        with self._lock:
            self._metrics[name].append(point)

    def get_metrics(
        self,
        name: str,
        since: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricPoint]:
        with self._lock:
            points = list(self._metrics.get(name, []))
            if since:
                points = [p for p in points if p.timestamp >= since]
            if labels:
                points = [p for p in points if all(p.labels.get(k) == v for k, v in labels.items())]
            return points

    def get_latest(self, name: str) -> Optional[MetricPoint]:
        with self._lock:
            points = self._metrics.get(name, [])
            return points[-1] if points else None

    def get_summary(self, name: str, since: Optional[datetime] = None) -> Dict[str, float]:
        points = self.get_metrics(name, since)
        if not points:
            return {}

        values = [p.value for p in points]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1] if values else 0,
        }


class SystemMonitor:
    def __init__(self, metrics: MetricsCollector, interval: float = 10.0):
        self.metrics = metrics
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        while self._running:
            try:
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            await asyncio.sleep(self.interval)

    def _collect_system_metrics(self):
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.metrics.record("system.cpu.percent", cpu_percent)

        # Memory
        memory = psutil.virtual_memory()
        self.metrics.record("system.memory.percent", memory.percent)
        self.metrics.record("system.memory.available_mb", memory.available / (1024 * 1024))
        self.metrics.record("system.memory.used_mb", memory.used / (1024 * 1024))

        # Disk
        disk = psutil.disk_usage('/')
        self.metrics.record("system.disk.percent", disk.percent)
        self.metrics.record("system.disk.free_gb", disk.free / (1024 ** 3))

        # Network
        net = psutil.net_io_counters()
        self.metrics.record("system.network.bytes_sent", net.bytes_sent)
        self.metrics.record("system.network.bytes_recv", net.bytes_recv)

        # Process
        process = psutil.Process()
        self.metrics.record("process.memory.mb", process.memory_info().rss / (1024 * 1024))
        self.metrics.record("process.cpu.percent", process.cpu_percent())
        self.metrics.record("process.threads", process.num_threads())


class RequestTracker:
    def __init__(self, metrics: MetricsCollector):
        self.metrics = metrics
        self._active_requests = 0
        self._lock = threading.RLock()

    @asynccontextmanager
    async def track(self, endpoint: str, method: str = "GET"):
        request_id = f"{method}:{endpoint}:{time.time()}"
        start_time = time.time()

        with self._lock:
            self._active_requests += 1
            self.metrics.record("http.requests.active", self._active_requests)

        try:
            yield request_id
        except Exception as e:
            self.metrics.record("http.requests.errors", 1, {"endpoint": endpoint, "method": method})
            raise
        finally:
            duration = time.time() - start_time
            self.metrics.record("http.requests.duration", duration, {"endpoint": endpoint, "method": method})
            self.metrics.record("http.requests.total", 1, {"endpoint": endpoint, "method": method})

            with self._lock:
                self._active_requests -= 1
                self.metrics.record("http.requests.active", self._active_requests)


# Global instances
health_checker = HealthChecker()
circuit_breaker_registry = CircuitBreakerRegistry()
rate_limiter_registry = RateLimiterRegistry()
metrics_collector = MetricsCollector()
system_monitor = SystemMonitor(metrics_collector)
request_tracker = RequestTracker(metrics_collector)


# Default health checks
def _check_disk_space() -> bool:
    disk = psutil.disk_usage('/')
    return disk.percent < 90


def _check_memory() -> bool:
    memory = psutil.virtual_memory()
    return memory.percent < 90


def _check_cpu() -> bool:
    return psutil.cpu_percent(interval=0.5) < 95


health_checker.register_check(HealthCheck(
    name="disk_space",
    check_func=_check_disk_space,
    critical=True,
    interval=60.0
))

health_checker.register_check(HealthCheck(
    name="memory",
    check_func=_check_memory,
    critical=True,
    interval=30.0
))

health_checker.register_check(HealthCheck(
    name="cpu",
    check_func=_check_cpu,
    critical=False,
    interval=30.0
))


async def initialize_production_services():
    """Initialize all production services."""
    await system_monitor.start()
    
    # Register default circuit breakers
    circuit_breaker_registry.register(CircuitBreaker("vector_store", failure_threshold=5, timeout=30))
    circuit_breaker_registry.register(CircuitBreaker("embedding_service", failure_threshold=3, timeout=60))
    circuit_breaker_registry.register(CircuitBreaker("llm_service", failure_threshold=5, timeout=60))
    
    # Register default rate limiters
    rate_limiter_registry.register("api_default", 100, 60.0)  # 100 req/min
    rate_limiter_registry.register("api_strict", 20, 60.0)   # 20 req/min
    rate_limiter_registry.register("ingestion", 10, 60.0)     # 10 req/min


async def shutdown_production_services():
    """Shutdown all production services."""
    await system_monitor.stop()


__all__ = [
    "HealthStatus",
    "CircuitState",
    "HealthCheck",
    "HealthReport",
    "HealthChecker",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitBreakerRegistry",
    "RateLimiter",
    "RateLimiterRegistry",
    "MetricPoint",
    "MetricsCollector",
    "SystemMonitor",
    "RequestTracker",
    "health_checker",
    "circuit_breaker_registry",
    "rate_limiter_registry",
    "metrics_collector",
    "system_monitor",
    "request_tracker",
    "initialize_production_services",
    "shutdown_production_services",
]