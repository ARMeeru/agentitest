import json
import time
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


class ErrorClassification(Enum):
    # Classification system for error types and recovery strategies
    RETRYABLE = "retryable"          # Can be automatically retried
    TERMINAL = "terminal"            # Should fail fast, no retry
    CONFIGURATION = "configuration" # Environment/config related
    TRANSIENT = "transient"         # Temporary network/service issues
    VALIDATION = "validation"       # Data validation failures


@dataclass
class ErrorContext:
    # Preserves comprehensive error context for debugging and recovery
    correlation_id: str
    timestamp: float = field(default_factory=time.time)
    component: str = ""
    operation: str = ""
    provider: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        # Convert to dictionary for JSON logging
        return {
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "component": self.component,
            "operation": self.operation,
            "provider": self.provider,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
            "stack_trace": self.stack_trace,
            "recovery_suggestions": self.recovery_suggestions,
        }

    def to_json(self) -> str:
        # Serialize to JSON string for logging
        return json.dumps(self.to_dict(), default=str, indent=2)


class AgentiTestError(Exception):
    # Base exception for all framework errors
    # Provides structured error information and recovery guidance

    def __init__(
        self,
        message: str,
        error_context: Optional[ErrorContext] = None,
        classification: ErrorClassification = ErrorClassification.TERMINAL,
        cause: Optional[Exception] = None,
        recovery_suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_context = error_context or ErrorContext(correlation_id="unknown")
        self.classification = classification
        self.cause = cause
        self.recovery_suggestions = recovery_suggestions or []

        # Add recovery suggestions to context
        if recovery_suggestions:
            self.error_context.recovery_suggestions.extend(recovery_suggestions)

    def is_retryable(self) -> bool:
        # Check if this error type supports automatic retry
        return self.classification in (
            ErrorClassification.RETRYABLE,
            ErrorClassification.TRANSIENT
        )

    def get_actionable_message(self) -> str:
        # Get error message with recovery suggestions
        base_message = f"{self.message}"

        if self.recovery_suggestions:
            suggestions = "\n".join(f"  - {suggestion}" for suggestion in self.recovery_suggestions)
            base_message += f"\n\nRecovery suggestions:\n{suggestions}"

        if self.error_context.provider:
            base_message += f"\n\nProvider: {self.error_context.provider}"

        if self.error_context.correlation_id != "unknown":
            base_message += f"\nCorrelation ID: {self.error_context.correlation_id}"

        return base_message

    def __str__(self) -> str:
        return self.get_actionable_message()


class LLMProviderError(AgentiTestError):
    # Exception for LLM provider failures - typically retryable

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        error_context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        self.provider = provider
        self.status_code = status_code

        # Default recovery suggestions for LLM provider errors
        recovery_suggestions = [
            f"Check {provider} API key configuration and validity",
            f"Verify {provider} service status and rate limits",
            "Consider switching to a different LLM provider temporarily",
            "Check network connectivity and firewall settings"
        ]

        # Classify based on status code
        classification = ErrorClassification.RETRYABLE
        if status_code:
            if status_code in (401, 403):  # Auth errors
                classification = ErrorClassification.CONFIGURATION
                recovery_suggestions.insert(0, f"API key authentication failed for {provider}")
            elif status_code in (400, 422):  # Bad request
                classification = ErrorClassification.TERMINAL
                recovery_suggestions.insert(0, "Request validation failed - check input parameters")
            elif status_code == 429:  # Rate limit
                classification = ErrorClassification.TRANSIENT
                recovery_suggestions.insert(0, "Rate limit exceeded - implementing backoff strategy")

        # Set provider in context
        if error_context:
            error_context.provider = provider
            error_context.component = "LLM Provider"

        super().__init__(
            message=f"LLM Provider Error ({provider}): {message}",
            error_context=error_context,
            classification=classification,
            cause=cause,
            recovery_suggestions=recovery_suggestions
        )


class BrowserSessionError(AgentiTestError):
    # Exception for browser session failures - typically retryable

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        browser_type: Optional[str] = None,
        error_context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        self.session_id = session_id
        self.browser_type = browser_type

        # Default recovery suggestions for browser session errors
        recovery_suggestions = [
            "Restart the browser session",
            "Check browser process and memory usage",
            "Verify browser binary and driver compatibility",
            "Check for conflicting browser extensions or policies"
        ]

        # Browser errors are usually retryable
        classification = ErrorClassification.RETRYABLE

        # Add timeout-specific suggestions
        if "timeout" in message.lower():
            recovery_suggestions.insert(0, "Increase timeout values for browser operations")
            recovery_suggestions.insert(1, "Check page load performance and network latency")

        # Set browser info in context
        if error_context:
            error_context.component = "Browser Session"
            if browser_type:
                error_context.metadata["browser_type"] = browser_type
            if session_id:
                error_context.metadata["session_id"] = session_id

        super().__init__(
            message=f"Browser Session Error: {message}",
            error_context=error_context,
            classification=classification,
            cause=cause,
            recovery_suggestions=recovery_suggestions
        )


class ConfigurationError(AgentiTestError):
    # Exception for configuration issues - terminal, should fail fast

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        expected_format: Optional[str] = None,
        error_context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        self.config_key = config_key
        self.config_file = config_file
        self.expected_format = expected_format

        # Configuration errors are terminal - need manual intervention
        recovery_suggestions = [
            "Review environment variables and configuration files",
            "Check .env file exists and contains required values",
            "Verify configuration syntax and formatting",
            "Consult framework documentation for configuration examples"
        ]

        if config_key:
            recovery_suggestions.insert(0, f"Set required configuration: {config_key}")

        if config_file:
            recovery_suggestions.insert(0, f"Check configuration file: {config_file}")

        if expected_format:
            recovery_suggestions.insert(0, f"Expected format: {expected_format}")

        # Set configuration info in context
        if error_context:
            error_context.component = "Configuration"
            if config_key:
                error_context.metadata["config_key"] = config_key
            if config_file:
                error_context.metadata["config_file"] = config_file

        super().__init__(
            message=f"Configuration Error: {message}",
            error_context=error_context,
            classification=ErrorClassification.CONFIGURATION,
            cause=cause,
            recovery_suggestions=recovery_suggestions
        )


class ValidationError(AgentiTestError):
    # Exception for test validation failures - terminal, indicates test logic issues

    def __init__(
        self,
        message: str,
        expected_value: Optional[str] = None,
        actual_value: Optional[str] = None,
        validation_type: str = "content",
        confidence_score: Optional[float] = None,
        error_context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        self.expected_value = expected_value
        self.actual_value = actual_value
        self.validation_type = validation_type
        self.confidence_score = confidence_score

        # Validation errors are terminal - indicate test or expectation issues
        recovery_suggestions = [
            "Review test expectations and validation criteria",
            "Check if page content or behavior has changed",
            "Consider using semantic validation instead of exact matching",
            "Verify test data and environment setup"
        ]

        if confidence_score is not None:
            recovery_suggestions.insert(0, f"Validation confidence: {confidence_score:.2%}")

        if expected_value and actual_value:
            recovery_suggestions.insert(0,
                f"Expected: '{expected_value}' but got: '{actual_value}'")

        # Set validation info in context
        if error_context:
            error_context.component = "Validation"
            error_context.operation = validation_type
            error_context.metadata.update({
                "expected_value": expected_value,
                "actual_value": actual_value,
                "validation_type": validation_type,
                "confidence_score": confidence_score
            })

        super().__init__(
            message=f"Validation Error ({validation_type}): {message}",
            error_context=error_context,
            classification=ErrorClassification.VALIDATION,
            cause=cause,
            recovery_suggestions=recovery_suggestions
        )
