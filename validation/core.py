# validation/core.py
"""
Core validation framework components.

This module defines the fundamental data structures and interfaces used throughout
the validation framework. All validation strategies implement the ValidationStrategy
protocol and return ValidationResult objects with confidence scoring.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timezone


class ValidationStatus(Enum):
    """Validation result status enumeration."""
    PASSED = "passed"
    FAILED = "failed" 
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"


class ValidationType(Enum):
    """Types of validation strategies available."""
    SEMANTIC = "semantic"
    TEXT_CONTENT = "text_content"
    VISUAL_ELEMENT = "visual_element"
    DOM_STRUCTURE = "dom_structure"
    ACCESSIBILITY = "accessibility"
    FUZZY_TEXT = "fuzzy_text"
    CUSTOM = "custom"


@dataclass
class ConfidenceScore:
    """
    Represents a confidence score with detailed breakdown.
    
    Attributes:
        value: Primary confidence score (0.0 to 1.0)
        components: Breakdown of confidence components
        method: Method used to calculate confidence
        reliability: How reliable this confidence score is (0.0 to 1.0)
    """
    value: float
    components: Dict[str, float] = field(default_factory=dict)
    method: str = "unknown"
    reliability: float = 1.0
    
    def __post_init__(self):
        """Validate confidence score values."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence value must be between 0.0 and 1.0, got {self.value}")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(f"Reliability must be between 0.0 and 1.0, got {self.reliability}")
    
    def is_above_threshold(self, threshold: float) -> bool:
        """Check if confidence score meets or exceeds threshold."""
        return self.value >= threshold
    
    def adjusted_score(self) -> float:
        """Get reliability-adjusted confidence score."""
        return self.value * self.reliability


@dataclass
class ValidationContext:
    """
    Context information for validation operations.
    
    Contains metadata, configuration, and state information needed
    for validation strategies to operate effectively.
    """
    validation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    validation_type: ValidationType = ValidationType.SEMANTIC
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Browser/page context
    page_url: Optional[str] = None
    viewport_size: Optional[Dict[str, int]] = None
    browser_info: Optional[Dict[str, str]] = None
    
    # Test context
    test_name: Optional[str] = None
    test_method: Optional[str] = None
    test_class: Optional[str] = None
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the validation context."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with optional default."""
        return self.metadata.get(key, default)
    
    def set_configuration(self, key: str, value: Any) -> None:
        """Set configuration parameter."""
        self.configuration[key] = value
    
    def get_configuration(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.configuration.get(key, default)


@dataclass 
class ValidationResult:
    """
    Comprehensive validation result with confidence scoring and metadata.
    
    This is the primary return type for all validation operations, providing
    detailed information about validation success/failure with confidence metrics.
    """
    status: ValidationStatus
    confidence_score: ConfidenceScore
    validation_type: ValidationType
    context: ValidationContext
    
    # Core validation data
    expected_value: Any = None
    actual_value: Any = None
    message: str = ""
    
    # Detailed results
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Performance metrics
    execution_time_ms: Optional[float] = None
    retry_count: int = 0
    
    # Cache information
    from_cache: bool = False
    cache_key: Optional[str] = None
    
    def is_successful(self) -> bool:
        """Check if validation was successful."""
        return self.status == ValidationStatus.PASSED
    
    def is_confident(self, threshold: float = 0.8) -> bool:
        """Check if validation result meets confidence threshold."""
        return self.confidence_score.is_above_threshold(threshold)
    
    def meets_criteria(self, confidence_threshold: float = 0.8) -> bool:
        """Check if validation meets both success and confidence criteria."""
        return self.is_successful() and self.is_confident(confidence_threshold)
    
    def add_detail(self, key: str, value: Any) -> None:
        """Add detail information to the result."""
        self.details[key] = value
    
    def add_error(self, error_message: str) -> None:
        """Add error message to the result."""
        self.errors.append(error_message)
        if self.status == ValidationStatus.PASSED:
            self.status = ValidationStatus.FAILED
    
    def add_warning(self, warning_message: str) -> None:
        """Add warning message to the result."""
        self.warnings.append(warning_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary for serialization."""
        return {
            "validation_id": self.context.validation_id,
            "correlation_id": self.context.correlation_id,
            "status": self.status.value,
            "confidence_score": {
                "value": self.confidence_score.value,
                "components": self.confidence_score.components,
                "method": self.confidence_score.method,
                "reliability": self.confidence_score.reliability,
                "adjusted_score": self.confidence_score.adjusted_score()
            },
            "validation_type": self.validation_type.value,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "message": self.message,
            "details": self.details,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time_ms": self.execution_time_ms,
            "retry_count": self.retry_count,
            "from_cache": self.from_cache,
            "timestamp": self.context.timestamp.isoformat(),
            "metadata": self.context.metadata
        }


class ValidationError(Exception):
    """
    Custom exception for validation framework errors.
    
    This exception is raised when validation operations encounter errors
    that prevent them from completing, distinct from validation failures.
    """
    
    def __init__(
        self,
        message: str,
        validation_context: Optional[ValidationContext] = None,
        cause: Optional[Exception] = None,
        validation_type: Optional[ValidationType] = None
    ):
        super().__init__(message)
        self.message = message
        self.validation_context = validation_context
        self.cause = cause
        self.validation_type = validation_type
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": "ValidationError",
            "message": self.message,
            "validation_type": self.validation_type.value if self.validation_type else None,
            "validation_id": self.validation_context.validation_id if self.validation_context else None,
            "correlation_id": self.validation_context.correlation_id if self.validation_context else None,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }


class ValidationStrategy(ABC):
    """
    Abstract base class for all validation strategies.
    
    All validation implementations must inherit from this class and implement
    the validate method. This ensures consistent interfaces across all validation
    types while allowing for strategy-specific configurations.
    """
    
    def __init__(self, name: str, default_confidence_threshold: float = 0.8):
        """
        Initialize validation strategy.
        
        Args:
            name: Human-readable name for this validation strategy
            default_confidence_threshold: Default confidence threshold for this strategy
        """
        self.name = name
        self.default_confidence_threshold = default_confidence_threshold
        self.validation_type = ValidationType.CUSTOM  # Override in subclasses
    
    @abstractmethod
    async def validate(
        self,
        expected: Any,
        actual: Any,
        context: Optional[ValidationContext] = None,
        confidence_threshold: Optional[float] = None,
        **kwargs
    ) -> ValidationResult:
        """
        Perform validation with confidence scoring.
        
        Args:
            expected: Expected value or condition
            actual: Actual value to validate against expected
            context: Validation context with metadata and configuration
            confidence_threshold: Minimum confidence threshold for success
            **kwargs: Strategy-specific parameters
            
        Returns:
            ValidationResult with confidence scoring and detailed information
            
        Raises:
            ValidationError: If validation operation cannot be completed
        """
        pass
    
    def create_context(
        self,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationContext:
        """Create a validation context for this strategy."""
        context = ValidationContext(
            correlation_id=correlation_id,
            validation_type=self.validation_type
        )
        if metadata:
            context.metadata.update(metadata)
        return context
    
    def create_result(
        self,
        status: ValidationStatus,
        confidence_score: ConfidenceScore,
        context: ValidationContext,
        expected: Any = None,
        actual: Any = None,
        message: str = ""
    ) -> ValidationResult:
        """Create a validation result for this strategy."""
        return ValidationResult(
            status=status,
            confidence_score=confidence_score,
            validation_type=self.validation_type,
            context=context,
            expected_value=expected,
            actual_value=actual,
            message=message
        )
    
    def get_confidence_threshold(self, override: Optional[float] = None) -> float:
        """Get effective confidence threshold (override or default)."""
        return override if override is not None else self.default_confidence_threshold