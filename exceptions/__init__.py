# AgentiTest Custom Exception Hierarchy
# Provides structured error handling with classification and recovery strategies

from .base import (
    AgentiTestError,
    LLMProviderError,
    BrowserSessionError,
    ConfigurationError,
    ValidationError,
    ErrorClassification,
    ErrorContext,
)

from .classification import (
    classify_error,
    is_retryable_error,
    get_recovery_strategy,
    create_error_context,
    RecoveryStrategy,
)

from .logging import (
    StructuredErrorLogger,
    log_error_with_context,
    get_error_correlation_id,
    configure_error_logging,
)

from .graceful_degradation import (
    DegradationLevel,
    DegradationConfig,
    GracefulDegradationManager,
    handle_llm_provider_failure,
    register_llm_providers,
    get_degradation_status,
    reset_provider_status,
)

__all__ = [
    # Base exceptions
    "AgentiTestError",
    "LLMProviderError",
    "BrowserSessionError",
    "ConfigurationError",
    "ValidationError",

    # Error classification
    "ErrorClassification",
    "ErrorContext",
    "classify_error",
    "is_retryable_error",
    "get_recovery_strategy",
    "create_error_context",
    "RecoveryStrategy",

    # Structured logging
    "StructuredErrorLogger",
    "log_error_with_context",
    "get_error_correlation_id",
    "configure_error_logging",

    # Graceful degradation
    "DegradationLevel",
    "DegradationConfig",
    "GracefulDegradationManager",
    "handle_llm_provider_failure",
    "register_llm_providers",
    "get_degradation_status",
    "reset_provider_status",
]
