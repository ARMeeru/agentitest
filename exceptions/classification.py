import traceback
import uuid
from typing import Dict, Any, Optional, List, Union, Type
from enum import Enum

from .base import (
    AgentiTestError,
    LLMProviderError,
    BrowserSessionError,
    ConfigurationError,
    ValidationError,
    ErrorClassification,
    ErrorContext,
)


class RecoveryStrategy(Enum):
    # Available recovery strategies for different error types
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK_PROVIDER = "fallback_provider"
    SESSION_RESTART = "session_restart"
    FAIL_FAST = "fail_fast"
    GRACEFUL_DEGRADATION = "graceful_degradation"


def create_error_context(
    correlation_id: Optional[str] = None,
    component: str = "",
    operation: str = "",
    provider: Optional[str] = None,
    retry_count: int = 0,
    **metadata
) -> ErrorContext:
    # Factory function to create error context with correlation ID
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())[:8]

    return ErrorContext(
        correlation_id=correlation_id,
        component=component,
        operation=operation,
        provider=provider,
        retry_count=retry_count,
        metadata=metadata,
        stack_trace=traceback.format_exc() if traceback.format_exc().strip() != "NoneType: None" else None
    )


def classify_error(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None
) -> ErrorClassification:
    # Intelligent error classification based on exception type and context
    context = context or {}

    # Already classified custom exceptions
    if isinstance(exception, AgentiTestError):
        return exception.classification

    # LLM provider related errors
    if _is_llm_provider_error(exception):
        return _classify_llm_error(exception, context)

    # Browser session related errors
    if _is_browser_session_error(exception):
        return _classify_browser_error(exception, context)

    # Configuration related errors
    if _is_configuration_error(exception):
        return ErrorClassification.CONFIGURATION

    # Validation related errors
    if _is_validation_error(exception):
        return ErrorClassification.VALIDATION

    # Network/connectivity related errors
    if _is_network_error(exception):
        return ErrorClassification.TRANSIENT

    # Default classification for unknown errors
    return ErrorClassification.TERMINAL


def is_retryable_error(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None
) -> bool:
    # Determine if an error should trigger automatic retry
    classification = classify_error(exception, context)

    return classification in (
        ErrorClassification.RETRYABLE,
        ErrorClassification.TRANSIENT
    )


def get_recovery_strategy(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None
) -> RecoveryStrategy:
    # Determine appropriate recovery strategy for an error
    classification = classify_error(exception, context)
    context = context or {}

    # Custom exceptions with built-in recovery strategies
    if isinstance(exception, LLMProviderError):
        if exception.status_code == 429:  # Rate limit
            return RecoveryStrategy.RETRY_WITH_BACKOFF
        elif exception.status_code in (500, 502, 503, 504):  # Server errors
            return RecoveryStrategy.CIRCUIT_BREAKER
        elif exception.status_code in (401, 403):  # Auth errors
            return RecoveryStrategy.FALLBACK_PROVIDER

    if isinstance(exception, BrowserSessionError):
        if "timeout" in str(exception).lower():
            return RecoveryStrategy.RETRY_WITH_BACKOFF
        else:
            return RecoveryStrategy.SESSION_RESTART

    # Classification-based strategies
    if classification == ErrorClassification.TRANSIENT:
        return RecoveryStrategy.RETRY_WITH_BACKOFF
    elif classification == ErrorClassification.RETRYABLE:
        retry_count = context.get("retry_count", 0)
        if retry_count > 2:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        return RecoveryStrategy.RETRY_WITH_BACKOFF
    elif classification in (ErrorClassification.CONFIGURATION, ErrorClassification.VALIDATION):
        return RecoveryStrategy.FAIL_FAST

    return RecoveryStrategy.FAIL_FAST


def convert_to_framework_exception(
    exception: Exception,
    context: Optional[ErrorContext] = None,
    component: str = "Unknown",
    operation: str = "Unknown"
) -> AgentiTestError:
    # Convert standard exceptions to framework exceptions with proper classification

    if isinstance(exception, AgentiTestError):
        return exception

    # Create context if not provided
    if context is None:
        context = create_error_context(
            component=component,
            operation=operation
        )

    error_message = str(exception)
    classification = classify_error(exception)

    # Convert to appropriate framework exception type
    if _is_llm_provider_error(exception):
        provider = context.provider or "unknown"
        status_code = getattr(exception, "status_code", None)
        return LLMProviderError(
            message=error_message,
            provider=provider,
            status_code=status_code,
            error_context=context,
            cause=exception
        )

    elif _is_browser_session_error(exception):
        return BrowserSessionError(
            message=error_message,
            error_context=context,
            cause=exception
        )

    elif _is_configuration_error(exception):
        return ConfigurationError(
            message=error_message,
            error_context=context,
            cause=exception
        )

    elif _is_validation_error(exception):
        return ValidationError(
            message=error_message,
            error_context=context,
            cause=exception
        )

    # Default to base framework exception
    return AgentiTestError(
        message=error_message,
        error_context=context,
        classification=classification,
        cause=exception
    )


# Private helper functions for error classification

def _is_llm_provider_error(exception: Exception) -> bool:
    # Check if exception is related to LLM provider
    error_str = str(exception).lower()
    exception_type = type(exception).__name__.lower()

    llm_indicators = [
        "api_key", "openai", "anthropic", "gemini", "groq", "azure",
        "rate limit", "quota", "model", "token", "llm", "chat"
    ]

    return any(indicator in error_str or indicator in exception_type
               for indicator in llm_indicators)


def _is_browser_session_error(exception: Exception) -> bool:
    # Check if exception is related to browser session
    error_str = str(exception).lower()
    exception_type = type(exception).__name__.lower()

    browser_indicators = [
        "browser", "playwright", "chromium", "firefox", "safari",
        "session", "page", "element", "timeout", "navigation",
        "screenshot", "click", "type", "scroll"
    ]

    return any(indicator in error_str or indicator in exception_type
               for indicator in browser_indicators)


def _is_configuration_error(exception: Exception) -> bool:
    # Check if exception is related to configuration
    error_str = str(exception).lower()
    exception_types = ("configerror", "environmenterror", "keyerror", "valueerror")

    config_indicators = [
        "config", "environment", "env", "setting", "variable",
        "missing", "required", "invalid", "format", ".env"
    ]

    return (any(exc_type in type(exception).__name__.lower() for exc_type in exception_types) or
            any(indicator in error_str for indicator in config_indicators))


def _is_validation_error(exception: Exception) -> bool:
    # Check if exception is related to validation
    error_str = str(exception).lower()
    exception_types = ("assertionerror", "validationerror", "valueerror")

    validation_indicators = [
        "assert", "expect", "validation", "match", "found", "not found",
        "invalid", "missing", "required", "format"
    ]

    return (any(exc_type in type(exception).__name__.lower() for exc_type in exception_types) or
            any(indicator in error_str for indicator in validation_indicators))


def _is_network_error(exception: Exception) -> bool:
    # Check if exception is related to network/connectivity
    error_str = str(exception).lower()
    exception_types = ("connectionerror", "timeouterror", "httperror", "urlerror")

    network_indicators = [
        "network", "connection", "timeout", "http", "https", "url",
        "dns", "socket", "ssl", "tls", "certificate", "proxy"
    ]

    return (any(exc_type in type(exception).__name__.lower() for exc_type in exception_types) or
            any(indicator in error_str for indicator in network_indicators))


def _classify_llm_error(exception: Exception, context: Dict[str, Any]) -> ErrorClassification:
    # Detailed classification for LLM provider errors
    error_str = str(exception).lower()

    # Rate limiting and quota errors - transient
    if any(indicator in error_str for indicator in ["rate limit", "quota", "429"]):
        return ErrorClassification.TRANSIENT

    # Authentication errors - configuration
    if any(indicator in error_str for indicator in ["401", "403", "unauthorized", "forbidden", "api_key"]):
        return ErrorClassification.CONFIGURATION

    # Server errors - retryable
    if any(indicator in error_str for indicator in ["500", "502", "503", "504", "server error"]):
        return ErrorClassification.RETRYABLE

    # Bad request errors - terminal
    if any(indicator in error_str for indicator in ["400", "422", "bad request", "invalid"]):
        return ErrorClassification.TERMINAL

    # Default to retryable for unknown LLM errors
    return ErrorClassification.RETRYABLE


def _classify_browser_error(exception: Exception, context: Dict[str, Any]) -> ErrorClassification:
    # Detailed classification for browser session errors
    error_str = str(exception).lower()

    # Timeout errors - retryable
    if "timeout" in error_str:
        return ErrorClassification.RETRYABLE

    # Navigation errors - retryable
    if any(indicator in error_str for indicator in ["navigation", "load", "network"]):
        return ErrorClassification.RETRYABLE

    # Element not found - could be retryable if page is loading
    if any(indicator in error_str for indicator in ["not found", "missing", "element"]):
        return ErrorClassification.RETRYABLE

    # Browser crash or session errors - retryable with session restart
    if any(indicator in error_str for indicator in ["crash", "session", "browser"]):
        return ErrorClassification.RETRYABLE

    # Default to retryable for browser errors
    return ErrorClassification.RETRYABLE
