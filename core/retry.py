import logging
import random
import time
from enum import Enum
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
    RetryError,
    AttemptManager
)

# Import new exception hierarchy
from exceptions import (
    LLMProviderError,
    ErrorContext,
    create_error_context,
    log_error_with_context,
    handle_llm_provider_failure,
    register_llm_providers
)


logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    # Supported LLM providers with their retry configurations
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GROQ = "groq"


@dataclass
class RetryConfig:
    # Configuration for retry behavior per provider
    max_attempts: int = 3
    base_wait_seconds: float = 1.0
    max_wait_seconds: float = 60.0
    jitter_multiplier: float = 0.1
    exponential_base: int = 2
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60


@dataclass
class CircuitBreakerState:
    # State tracking for circuit breaker pattern
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    is_open: bool = False
    failure_history: deque = field(default_factory=lambda: deque(maxlen=100))


class RetryableException(Exception):
    # Base class for exceptions that should trigger retries (kept for backward compatibility)
    pass


class LLMAPIException(LLMProviderError):
    # Legacy exception for backward compatibility - now inherits from LLMProviderError
    def __init__(self, provider: str, message: str, status_code: Optional[int] = None):
        super().__init__(
            message=message,
            provider=provider,
            status_code=status_code,
            error_context=create_error_context(
                component="LLM API",
                operation="api_call",
                provider=provider
            )
        )


class CircuitBreakerOpenException(LLMProviderError):
    # Exception raised when circuit breaker is open - now uses framework exception
    def __init__(self, provider: str):
        super().__init__(
            message=f"Circuit breaker is open for provider: {provider}",
            provider=provider,
            error_context=create_error_context(
                component="Circuit Breaker",
                operation="circuit_breaker_check",
                provider=provider
            )
        )


class RetryManager:
    # Manages retry policies and circuit breaker state for LLM providers

    # Default retry configurations per provider
    DEFAULT_CONFIGS: Dict[LLMProvider, RetryConfig] = {
        LLMProvider.GEMINI: RetryConfig(max_attempts=3, base_wait_seconds=1.0),
        LLMProvider.OPENAI: RetryConfig(max_attempts=3, base_wait_seconds=2.0),
        LLMProvider.ANTHROPIC: RetryConfig(max_attempts=5, base_wait_seconds=1.5),
        LLMProvider.AZURE: RetryConfig(max_attempts=4, base_wait_seconds=2.0),
        LLMProvider.GROQ: RetryConfig(max_attempts=3, base_wait_seconds=1.0, max_wait_seconds=30.0),
    }

    def __init__(self):
        self.configs = self.DEFAULT_CONFIGS.copy()
        self.circuit_breakers: Dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)
        
        # Register providers with graceful degradation manager
        provider_names = [provider.value for provider in self.DEFAULT_CONFIGS.keys()]
        register_llm_providers(provider_names)

    def get_config(self, provider: LLMProvider) -> RetryConfig:
        # Get retry configuration for a provider
        return self.configs.get(provider, RetryConfig())

    def update_config(self, provider: LLMProvider, config: RetryConfig):
        # Update retry configuration for a provider
        self.configs[provider] = config

    def check_circuit_breaker(self, provider_name: str) -> bool:
        # Check if circuit breaker allows requests for a provider
        breaker = self.circuit_breakers[provider_name]
        current_time = time.time()

        if breaker.is_open:
            # Check if recovery timeout has passed
            if (breaker.last_failure_time and
                current_time - breaker.last_failure_time > self.get_config(LLMProvider(provider_name)).circuit_breaker_recovery_timeout):
                logger.info(f"Circuit breaker recovery timeout passed for {provider_name}, attempting reset")
                breaker.is_open = False
                breaker.failure_count = 0
                return True
            return False

        return True

    def record_success(self, provider_name: str):
        # Record successful operation, reset circuit breaker if needed
        breaker = self.circuit_breakers[provider_name]
        if breaker.failure_count > 0:
            logger.info(f"Resetting failure count for {provider_name} after successful operation")
            
            # Log circuit breaker recovery
            if breaker.is_open:
                from exceptions.logging import log_circuit_breaker_event
                log_circuit_breaker_event(
                    provider=provider_name,
                    event_type="closed",
                    failure_count=0
                )
            
            breaker.failure_count = 0
            breaker.is_open = False
            
            # Reset provider status in degradation manager
            from exceptions.graceful_degradation import reset_provider_status
            reset_provider_status(provider_name)

    def record_failure(self, provider_name: str, exception: Exception):
        # Record failure and potentially open circuit breaker
        breaker = self.circuit_breakers[provider_name]
        current_time = time.time()

        breaker.failure_count += 1
        breaker.last_failure_time = current_time
        breaker.failure_history.append({
            'timestamp': current_time,
            'exception': str(exception),
            'type': type(exception).__name__
        })

        # Log failure with structured error logging
        context = create_error_context(
            component="Circuit Breaker",
            operation="record_failure",
            provider=provider_name,
            retry_count=breaker.failure_count
        )
        log_error_with_context(exception, context, level="warning")

        # Handle graceful degradation
        degradation_result = handle_llm_provider_failure(provider_name, exception, context)
        
        config = self.get_config(LLMProvider(provider_name))
        if breaker.failure_count >= config.circuit_breaker_threshold:
            breaker.is_open = True
            logger.error(f"Circuit breaker opened for {provider_name} after {breaker.failure_count} failures")
            
            # Log circuit breaker event
            from exceptions.logging import log_circuit_breaker_event
            log_circuit_breaker_event(
                provider=provider_name,
                event_type="opened",
                failure_count=breaker.failure_count
            )

    def create_retry_decorator(self, provider: LLMProvider, correlation_id: Optional[str] = None):
        # Create a retry decorator with provider-specific configuration
        config = self.get_config(provider)
        provider_name = provider.value

        def jitter_wait(retry_state):
            # Add jitter to exponential backoff
            base_wait = config.base_wait_seconds * (config.exponential_base ** (retry_state.attempt_number - 1))
            jitter = base_wait * config.jitter_multiplier * random.random()
            total_wait = min(base_wait + jitter, config.max_wait_seconds)
            return total_wait

        def before_sleep(retry_state):
            correlation_msg = f" [correlation_id: {correlation_id}]" if correlation_id else ""
            logger.warning(
                f"Retrying {provider_name} LLM call{correlation_msg} - "
                f"attempt {retry_state.attempt_number}/{config.max_attempts} "
                f"after {retry_state.seconds_since_start:.2f}s"
            )

        def after_attempt(retry_state):
            if retry_state.outcome and retry_state.outcome.failed:
                exception = retry_state.outcome.exception()
                self.record_failure(provider_name, exception)
                
                # Log retry attempt with correlation ID
                from exceptions.logging import log_recovery_attempt
                log_recovery_attempt(
                    correlation_id=correlation_id or "unknown",
                    strategy="retry_with_backoff",
                    attempt_number=retry_state.attempt_number,
                    success=False,
                    provider=provider_name,
                    exception=str(exception)
                )
            else:
                self.record_success(provider_name)
                
                # Log successful retry recovery
                if retry_state.attempt_number > 1:
                    from exceptions.logging import log_recovery_attempt
                    log_recovery_attempt(
                        correlation_id=correlation_id or "unknown",
                        strategy="retry_with_backoff",
                        attempt_number=retry_state.attempt_number,
                        success=True,
                        provider=provider_name
                    )

        return retry(
            stop=stop_after_attempt(config.max_attempts),
            wait=jitter_wait,
            retry=retry_if_exception_type((LLMAPIException, LLMProviderError, RetryableException)),
            before_sleep=before_sleep,
            after=after_attempt,
            reraise=True
        )


# Global retry manager instance
retry_manager = RetryManager()


def with_llm_retry(provider: LLMProvider, correlation_id: Optional[str] = None):
    # Decorator factory for adding retry logic to LLM operations
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            provider_name = provider.value

            # Check circuit breaker before attempting
            if not retry_manager.check_circuit_breaker(provider_name):
                raise CircuitBreakerOpenException(provider_name)

            # Apply retry decorator
            retry_decorator = retry_manager.create_retry_decorator(provider, correlation_id)
            retried_func = retry_decorator(func)

            try:
                return retried_func(*args, **kwargs)
            except RetryError as e:
                # Log final failure with correlation ID
                correlation_msg = f" [correlation_id: {correlation_id}]" if correlation_id else ""
                logger.error(f"All retry attempts exhausted for {provider_name}{correlation_msg}: {e}")
                
                # Create comprehensive error context
                context = create_error_context(
                    correlation_id=correlation_id,
                    component="Retry Manager",
                    operation="exhausted_retries",
                    provider=provider_name,
                    retry_count=retry_manager.get_config(provider).max_attempts
                )
                
                # Create framework exception with proper context
                framework_exception = LLMAPIException(
                    provider_name,
                    f"Failed after {retry_manager.get_config(provider).max_attempts} attempts"
                )
                framework_exception.error_context = context
                
                # Log with structured logging
                log_error_with_context(framework_exception, context, level="error")
                
                raise framework_exception from e

        return wrapper
    return decorator


def get_correlation_id() -> str:
    # Generate a correlation ID for request tracing
    import uuid
    return str(uuid.uuid4())[:8]


def configure_retry_policy(provider: LLMProvider, **kwargs):
    # Configure retry policy for a specific provider
    current_config = retry_manager.get_config(provider)

    # Update configuration with provided values
    config_dict = {
        'max_attempts': kwargs.get('max_attempts', current_config.max_attempts),
        'base_wait_seconds': kwargs.get('base_wait_seconds', current_config.base_wait_seconds),
        'max_wait_seconds': kwargs.get('max_wait_seconds', current_config.max_wait_seconds),
        'jitter_multiplier': kwargs.get('jitter_multiplier', current_config.jitter_multiplier),
        'exponential_base': kwargs.get('exponential_base', current_config.exponential_base),
        'circuit_breaker_threshold': kwargs.get('circuit_breaker_threshold', current_config.circuit_breaker_threshold),
        'circuit_breaker_recovery_timeout': kwargs.get('circuit_breaker_recovery_timeout', current_config.circuit_breaker_recovery_timeout),
    }

    new_config = RetryConfig(**config_dict)
    retry_manager.update_config(provider, new_config)

    logger.info(f"Updated retry configuration for {provider.value}: {new_config}")


def get_circuit_breaker_status(provider: LLMProvider) -> Dict[str, Any]:
    # Get current circuit breaker status for a provider
    provider_name = provider.value
    breaker = retry_manager.circuit_breakers[provider_name]

    return {
        'provider': provider_name,
        'is_open': breaker.is_open,
        'failure_count': breaker.failure_count,
        'last_failure_time': breaker.last_failure_time,
        'recent_failures': list(breaker.failure_history)[-5:] if breaker.failure_history else []
    }


def reset_circuit_breaker(provider: LLMProvider):
    # Manually reset circuit breaker for a provider
    provider_name = provider.value
    breaker = retry_manager.circuit_breakers[provider_name]
    breaker.is_open = False
    breaker.failure_count = 0
    breaker.last_failure_time = None
    logger.info(f"Manually reset circuit breaker for {provider_name}")
