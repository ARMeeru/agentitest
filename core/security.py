"""
Security utilities for AgentiTest Framework.

This module provides security hardening features including:
- Credential masking in logs and outputs
- Secure data handling for test reports
- Security validation utilities
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from functools import wraps


# Common patterns for sensitive data
SENSITIVE_PATTERNS = {
    'api_key': [
        r'sk-[a-zA-Z0-9]{40,}',  # OpenAI-style keys
        r'sk-ant-[a-zA-Z0-9-]{90,}',  # Anthropic keys
        r'gsk_[a-zA-Z0-9]{40,}',  # Groq keys
        r'AIza[a-zA-Z0-9]{35}',  # Google API keys
        r'[a-zA-Z0-9]{32}',  # Azure keys (generic 32-char)
    ],
    'password': [
        r'password["\']?\s*[:=]\s*["\']?([^"\';\s]+)',
        r'passwd["\']?\s*[:=]\s*["\']?([^"\';\s]+)',
        r'pwd["\']?\s*[:=]\s*["\']?([^"\';\s]+)',
    ],
    'token': [
        r'token["\']?\s*[:=]\s*["\']?([^"\';\s]+)',
        r'bearer\s+([a-zA-Z0-9\-._~+/]+=*)',
    ],
    'auth': [
        r'authorization["\']?\s*[:=]\s*["\']?([^"\';\s]+)',
        r'auth["\']?\s*[:=]\s*["\']?([^"\';\s]+)',
    ]
}

# Environment variable patterns that contain sensitive data
SENSITIVE_ENV_VARS = {
    'GOOGLE_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 
    'AZURE_API_KEY', 'GROQ_API_KEY', 'PASSWORD', 'PASSWD', 
    'TOKEN', 'SECRET', 'KEY', 'AUTH'
}


def mask_credential(value: str, mask_char: str = "*", reveal_chars: int = 4) -> str:
    """
    Mask a credential string, revealing only the first and last few characters.
    
    Args:
        value: The credential string to mask
        mask_char: Character to use for masking (default: "*")
        reveal_chars: Number of characters to reveal at start and end (default: 4)
    
    Returns:
        Masked credential string
    """
    if not value or len(value) <= reveal_chars * 2:
        return mask_char * 8  # Standard masked length for short values
    
    start = value[:reveal_chars]
    end = value[-reveal_chars:]
    mask_length = max(8, len(value) - (reveal_chars * 2))
    
    return f"{start}{mask_char * mask_length}{end}"


def mask_sensitive_data(data: Any, deep_copy: bool = True) -> Any:
    """
    Recursively mask sensitive data in various data structures.
    
    Args:
        data: Data structure to mask (dict, list, string, etc.)
        deep_copy: Whether to create a deep copy (default: True)
    
    Returns:
        Data structure with sensitive values masked
    """
    if isinstance(data, dict):
        result = {} if deep_copy else data
        for key, value in data.items():
            key_lower = key.lower() if isinstance(key, str) else str(key).lower()
            
            # Check if key indicates sensitive data
            is_sensitive = any(sensitive in key_lower for sensitive in 
                             ['key', 'token', 'password', 'passwd', 'secret', 'auth', 'credential'])
            
            if is_sensitive and isinstance(value, str):
                result[key] = mask_credential(value)
            else:
                result[key] = mask_sensitive_data(value, deep_copy)
        return result
    
    elif isinstance(data, list):
        return [mask_sensitive_data(item, deep_copy) for item in data]
    
    elif isinstance(data, str):
        # Apply pattern-based masking for strings
        masked_data = data
        for category, patterns in SENSITIVE_PATTERNS.items():
            for pattern in patterns:
                masked_data = re.sub(pattern, lambda m: mask_credential(m.group(0)), 
                                   masked_data, flags=re.IGNORECASE)
        return masked_data
    
    else:
        return data


def secure_log_formatter(record: logging.LogRecord) -> logging.LogRecord:
    """
    Custom log formatter that masks sensitive data in log messages.
    
    Args:
        record: Log record to format
    
    Returns:
        Log record with sensitive data masked
    """
    # Mask the message
    if hasattr(record, 'msg') and isinstance(record.msg, str):
        record.msg = mask_sensitive_data(record.msg, deep_copy=False)
    
    # Mask arguments if present
    if hasattr(record, 'args') and record.args:
        if isinstance(record.args, (list, tuple)):
            record.args = tuple(mask_sensitive_data(list(record.args)))
        elif isinstance(record.args, dict):
            record.args = mask_sensitive_data(record.args)
    
    return record


class SecureLogHandler(logging.Handler):
    """
    Custom log handler that ensures all sensitive data is masked before logging.
    """
    
    def __init__(self, base_handler: logging.Handler):
        super().__init__()
        self.base_handler = base_handler
        self.setLevel(base_handler.level)
        self.setFormatter(base_handler.formatter)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record after masking sensitive data."""
        try:
            # Create a copy to avoid modifying the original record
            secure_record = logging.LogRecord(
                name=record.name,
                level=record.levelno,
                pathname=record.pathname,
                lineno=record.lineno,
                msg=record.msg,
                args=record.args,
                exc_info=record.exc_info,
                func=record.funcName,
                sinfo=record.stack_info
            )
            
            # Apply security formatting
            secure_record = secure_log_formatter(secure_record)
            
            # Emit through base handler
            self.base_handler.emit(secure_record)
        except Exception:
            self.handleError(record)


def secure_logging_wrapper(func):
    """
    Decorator that ensures function arguments and return values are logged securely.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log function entry with masked arguments
        safe_args = [mask_sensitive_data(arg) for arg in args]
        safe_kwargs = mask_sensitive_data(kwargs)
        
        logging.debug(f"Entering {func.__name__} with args={safe_args}, kwargs={safe_kwargs}")
        
        try:
            result = func(*args, **kwargs)
            # Log successful exit (don't log return value as it might be sensitive)
            logging.debug(f"Successfully completed {func.__name__}")
            return result
        except Exception as e:
            # Log error without exposing sensitive data
            logging.error(f"Error in {func.__name__}: {type(e).__name__}")
            raise
    
    return wrapper


def sanitize_for_allure(data: Any) -> str:
    """
    Sanitize data for safe inclusion in Allure reports.
    
    Args:
        data: Data to sanitize
    
    Returns:
        String representation safe for reporting
    """
    if isinstance(data, (dict, list)):
        sanitized = mask_sensitive_data(data)
        return str(sanitized)
    elif isinstance(data, str):
        return mask_sensitive_data(data)
    else:
        return str(data)


def validate_environment_security() -> List[str]:
    """
    Validate environment for security issues.
    
    Returns:
        List of security warnings/issues found
    """
    import os
    warnings = []
    
    # Check for sensitive data in environment variables
    for var_name, var_value in os.environ.items():
        if any(sensitive in var_name.upper() for sensitive in SENSITIVE_ENV_VARS):
            if var_value in ('', 'YOUR_API_KEY', 'changeme', 'password', '123456'):
                warnings.append(f"Environment variable {var_name} has insecure default value")
    
    # Check for .env file in version control indicators
    if os.path.exists('.env'):
        if os.path.exists('.git'):
            try:
                with open('.gitignore', 'r') as f:
                    gitignore_content = f.read()
                    if '.env' not in gitignore_content:
                        warnings.append(".env file exists but is not in .gitignore")
            except FileNotFoundError:
                warnings.append(".env file exists but no .gitignore found")
    
    return warnings


def setup_secure_logging():
    """
    Setup secure logging configuration for the entire application.
    """
    # Get root logger
    root_logger = logging.getLogger()
    
    # Wrap existing handlers with secure handlers
    existing_handlers = root_logger.handlers.copy()
    root_logger.handlers.clear()
    
    for handler in existing_handlers:
        secure_handler = SecureLogHandler(handler)
        root_logger.addHandler(secure_handler)
    
    # If no handlers exist, add a secure console handler
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        secure_handler = SecureLogHandler(console_handler)
        root_logger.addHandler(secure_handler)
    
    logging.info("Secure logging initialized - sensitive data will be masked")


# Global security configuration
SECURITY_CONFIG = {
    'mask_credentials': True,
    'mask_in_logs': True,
    'mask_in_reports': True,
    'validate_environment': True,
    'secure_error_messages': True
}


def get_security_status() -> Dict[str, Any]:
    """
    Get current security configuration status.
    
    Returns:
        Dictionary containing security status information
    """
    warnings = validate_environment_security()
    
    return {
        'config': SECURITY_CONFIG,
        'warnings': warnings,
        'secure_logging_enabled': len([h for h in logging.getLogger().handlers 
                                     if isinstance(h, SecureLogHandler)]) > 0,
        'environment_validated': len(warnings) == 0
    }