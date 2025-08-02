import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

from .base import AgentiTestError, ErrorContext
from .classification import classify_error, get_recovery_strategy

# Import security utilities for credential masking
try:
    from core.security import mask_sensitive_data, SecureLogHandler, setup_secure_logging
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False


class StructuredErrorLogger:
    # JSON-structured error logger with correlation ID tracking

    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)

        # Configure JSON formatter if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = JSONFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_error(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        level: str = "error",
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> str:
        # Log error with structured JSON format and return correlation ID
        correlation_id = get_error_correlation_id()

        # Create context if not provided
        if context is None:
            from .classification import create_error_context
            context = create_error_context(correlation_id=correlation_id)

        # Ensure correlation ID is set
        if not context.correlation_id or context.correlation_id == "unknown":
            context.correlation_id = correlation_id

        # Convert to framework exception if needed
        if not isinstance(exception, AgentiTestError):
            from .classification import convert_to_framework_exception
            framework_exception = convert_to_framework_exception(exception, context)
        else:
            framework_exception = exception

        # Build structured log entry
        log_entry = {
            "timestamp": time.time(),
            "level": level.upper(),
            "correlation_id": context.correlation_id,
            "error": {
                "type": type(exception).__name__,
                "message": str(exception),
                "classification": framework_exception.classification.value,
                "is_retryable": framework_exception.is_retryable(),
                "recovery_strategy": get_recovery_strategy(exception).value,
                "recovery_suggestions": framework_exception.recovery_suggestions
            },
            "context": context.to_dict(),
            "framework_exception": {
                "type": type(framework_exception).__name__,
                "actionable_message": framework_exception.get_actionable_message()
            }
        }

        # Add additional fields if provided
        if additional_fields:
            # Mask sensitive data in additional fields
            if SECURITY_AVAILABLE:
                additional_fields = mask_sensitive_data(additional_fields)
            log_entry.update(additional_fields)

        # Add cause chain if available
        if hasattr(exception, '__cause__') and exception.__cause__:
            cause_message = str(exception.__cause__)
            if SECURITY_AVAILABLE:
                cause_message = mask_sensitive_data(cause_message)
            log_entry["error"]["cause"] = {
                "type": type(exception.__cause__).__name__,
                "message": cause_message
            }

        # Mask sensitive data in the entire log entry
        if SECURITY_AVAILABLE:
            log_entry = mask_sensitive_data(log_entry)

        # Log at appropriate level
        log_method = getattr(self.logger, level.lower(), self.logger.error)
        log_method(json.dumps(log_entry, default=str, indent=2))

        return context.correlation_id

    def log_recovery_attempt(
        self,
        correlation_id: str,
        strategy: str,
        attempt_number: int,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        # Log recovery attempt with correlation ID
        log_entry = {
            "timestamp": time.time(),
            "level": "INFO",
            "correlation_id": correlation_id,
            "event_type": "recovery_attempt",
            "recovery": {
                "strategy": strategy,
                "attempt_number": attempt_number,
                "success": success,
                "details": details or {}
            }
        }

        self.logger.info(json.dumps(log_entry, default=str, indent=2))

    def log_circuit_breaker_event(
        self,
        provider: str,
        event_type: str,  # "opened", "closed", "half_open"
        failure_count: int,
        details: Optional[Dict[str, Any]] = None
    ):
        # Log circuit breaker state changes
        correlation_id = get_error_correlation_id()

        log_entry = {
            "timestamp": time.time(),
            "level": "WARNING",
            "correlation_id": correlation_id,
            "event_type": "circuit_breaker",
            "circuit_breaker": {
                "provider": provider,
                "event": event_type,
                "failure_count": failure_count,
                "details": details or {}
            }
        }

        self.logger.warning(json.dumps(log_entry, default=str, indent=2))


class JSONFormatter(logging.Formatter):
    # Custom JSON formatter for structured logging

    def format(self, record: logging.LogRecord) -> str:
        # Convert log record to JSON format

        # Try to parse message as JSON first
        try:
            message = json.loads(record.getMessage())
            if isinstance(message, dict):
                return json.dumps(message, default=str)
        except (json.JSONDecodeError, TypeError):
            pass

        # Fall back to standard structured format
        log_entry = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


# Global structured error logger instance
_structured_logger = StructuredErrorLogger("agentitest.errors")


def log_error_with_context(
    exception: Exception,
    context: Optional[ErrorContext] = None,
    level: str = "error",
    **additional_fields
) -> str:
    # Convenience function to log error with context
    return _structured_logger.log_error(
        exception=exception,
        context=context,
        level=level,
        additional_fields=additional_fields
    )


def get_error_correlation_id() -> str:
    # Generate unique correlation ID for error tracking
    return str(uuid.uuid4())[:8]


def log_recovery_attempt(
    correlation_id: str,
    strategy: str,
    attempt_number: int,
    success: bool,
    **details
):
    # Convenience function to log recovery attempts
    _structured_logger.log_recovery_attempt(
        correlation_id=correlation_id,
        strategy=strategy,
        attempt_number=attempt_number,
        success=success,
        details=details
    )


def log_circuit_breaker_event(
    provider: str,
    event_type: str,
    failure_count: int,
    **details
):
    # Convenience function to log circuit breaker events
    _structured_logger.log_circuit_breaker_event(
        provider=provider,
        event_type=event_type,
        failure_count=failure_count,
        details=details
    )


def configure_error_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None,
    enable_security: bool = True
):
    # Configure error logging settings with optional security features
    logger = logging.getLogger("agentitest.errors")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler()
    if format_type.lower() == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    # Wrap with secure handler if security is available and enabled
    if enable_security and SECURITY_AVAILABLE:
        console_handler = SecureLogHandler(console_handler)
    
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        if format_type.lower() == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        
        # Wrap with secure handler if security is available and enabled
        if enable_security and SECURITY_AVAILABLE:
            file_handler = SecureLogHandler(file_handler)
        
        logger.addHandler(file_handler)
    
    # Setup global secure logging if available and enabled
    if enable_security and SECURITY_AVAILABLE:
        setup_secure_logging()
        logger.info("Secure logging configured - sensitive data will be masked")
