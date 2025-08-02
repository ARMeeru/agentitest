# validation/registry.py
"""
Validation registry for managing and accessing validation strategies.

This module provides a centralized registry for all validation strategies,
making it easy to discover, configure, and use different validation approaches
in a consistent manner.
"""

import logging
from typing import Dict, Any, Optional, List, Type, Union
from enum import Enum

from .core import (
    ValidationStrategy,
    ValidationResult,
    ValidationContext,
    ValidationType,
    ValidationError
)
from .semantic import SemanticValidator
from .text_content import TextContentValidator, TextMatchingMode
from .visual_element import VisualElementValidator
from .dom_structure import DOMStructureValidator, DOMStructureType
from .accessibility import AccessibilityValidator, AccessibilityCategory, WCAGLevel
from .cache import ValidationResultCache
from .assertions import CustomAssertions


class ValidationMode(Enum):
    """Validation execution modes."""
    STRICT = "strict"          # Fail fast on first failure
    PERMISSIVE = "permissive"  # Continue despite failures
    CACHED = "cached"          # Use cached results when available
    FRESH = "fresh"            # Always perform fresh validation


class ValidationRegistry:
    """
    Central registry for validation strategies and execution.
    
    This registry provides a unified interface for accessing all validation
    strategies, managing caching, and coordinating validation execution.
    """
    
    def __init__(
        self,
        enable_caching: bool = True,
        cache_config: Optional[Dict[str, Any]] = None,
        default_confidence_threshold: float = 0.8,
        validation_mode: ValidationMode = ValidationMode.CACHED
    ):
        """
        Initialize validation registry.
        
        Args:
            enable_caching: Enable validation result caching
            cache_config: Configuration for validation cache
            default_confidence_threshold: Default confidence threshold
            validation_mode: Default validation execution mode
        """
        self.default_confidence_threshold = default_confidence_threshold
        self.validation_mode = validation_mode
        
        # Initialize cache
        if enable_caching:
            cache_settings = cache_config or {}
            self.cache = ValidationResultCache(**cache_settings)
        else:
            self.cache = None
        
        # Initialize validators
        self._validators: Dict[ValidationType, ValidationStrategy] = {}
        self._initialize_validators()
        
        # Initialize custom assertions
        self.assertions = CustomAssertions(
            default_confidence_threshold=default_confidence_threshold
        )
        
        # Statistics
        self._stats = {
            "validations_performed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_failures": 0,
            "validation_successes": 0
        }
    
    def _initialize_validators(self):
        """Initialize all validation strategies."""
        self._validators[ValidationType.SEMANTIC] = SemanticValidator(
            default_confidence_threshold=self.default_confidence_threshold
        )
        
        self._validators[ValidationType.TEXT_CONTENT] = TextContentValidator(
            default_confidence_threshold=self.default_confidence_threshold
        )
        
        self._validators[ValidationType.VISUAL_ELEMENT] = VisualElementValidator(
            default_confidence_threshold=self.default_confidence_threshold
        )
        
        self._validators[ValidationType.DOM_STRUCTURE] = DOMStructureValidator(
            default_confidence_threshold=self.default_confidence_threshold
        )
        
        self._validators[ValidationType.ACCESSIBILITY] = AccessibilityValidator(
            default_confidence_threshold=self.default_confidence_threshold
        )
    
    async def validate(
        self,
        validation_type: ValidationType,
        expected: Any,
        actual: Any,
        context: Optional[ValidationContext] = None,
        confidence_threshold: Optional[float] = None,
        use_cache: Optional[bool] = None,
        **validation_kwargs
    ) -> ValidationResult:
        """
        Perform validation using specified strategy.
        
        Args:
            validation_type: Type of validation to perform
            expected: Expected value or condition
            actual: Actual value to validate
            context: Validation context
            confidence_threshold: Minimum confidence threshold
            use_cache: Whether to use cached results (overrides default mode)
            **validation_kwargs: Strategy-specific parameters
            
        Returns:
            ValidationResult with confidence scoring
        """
        self._stats["validations_performed"] += 1
        
        # Determine if we should use cache
        should_use_cache = (
            self.cache is not None and
            (use_cache if use_cache is not None else self.validation_mode == ValidationMode.CACHED)
        )
        
        # Try to get from cache first
        cache_key = None
        if should_use_cache:
            cache_key = self.cache.generate_cache_key(
                validation_type=validation_type,
                expected=expected,
                actual=actual,
                context=context,
                **validation_kwargs
            )
            
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self._stats["cache_hits"] += 1
                logging.debug(f"Using cached validation result [{cache_key[:16]}...]")
                return cached_result
            else:
                self._stats["cache_misses"] += 1
        
        # Get validator
        validator = self._validators.get(validation_type)
        if not validator:
            raise ValidationError(f"No validator registered for type: {validation_type}")
        
        # Perform validation
        try:
            result = await validator.validate(
                expected=expected,
                actual=actual,
                context=context,
                confidence_threshold=confidence_threshold or self.default_confidence_threshold,
                **validation_kwargs
            )
            
            # Cache result if enabled
            if should_use_cache and cache_key:
                self.cache.set(cache_key, result)
            
            # Update statistics
            if result.is_successful():
                self._stats["validation_successes"] += 1
            else:
                self._stats["validation_failures"] += 1
            
            return result
            
        except Exception as e:
            self._stats["validation_failures"] += 1
            raise
    
    async def validate_semantic(
        self,
        expected: Union[str, List[str]],
        actual: str,
        context: Optional[ValidationContext] = None,
        confidence_threshold: Optional[float] = None,
        **kwargs
    ) -> ValidationResult:
        """Perform semantic validation."""
        return await self.validate(
            validation_type=ValidationType.SEMANTIC,
            expected=expected,
            actual=actual,
            context=context,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
    
    async def validate_text_content(
        self,
        expected: Union[str, List[str]],
        actual: str,
        matching_mode: TextMatchingMode = TextMatchingMode.FUZZY,
        context: Optional[ValidationContext] = None,
        confidence_threshold: Optional[float] = None,
        **kwargs
    ) -> ValidationResult:
        """Perform text content validation."""
        return await self.validate(
            validation_type=ValidationType.TEXT_CONTENT,
            expected=expected,
            actual=actual,
            context=context,
            confidence_threshold=confidence_threshold,
            matching_mode=matching_mode,
            **kwargs
        )
    
    async def validate_visual_element(
        self,
        expected: Dict[str, Any],
        actual: Optional[Dict[str, Any]],
        context: Optional[ValidationContext] = None,
        confidence_threshold: Optional[float] = None,
        **kwargs
    ) -> ValidationResult:
        """Perform visual element validation."""
        return await self.validate(
            validation_type=ValidationType.VISUAL_ELEMENT,
            expected=expected,
            actual=actual,
            context=context,
            confidence_threshold=confidence_threshold,
            **kwargs
        )
    
    async def validate_dom_structure(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        structure_type: DOMStructureType = DOMStructureType.HIERARCHY,
        context: Optional[ValidationContext] = None,
        confidence_threshold: Optional[float] = None,
        **kwargs
    ) -> ValidationResult:
        """Perform DOM structure validation."""
        return await self.validate(
            validation_type=ValidationType.DOM_STRUCTURE,
            expected=expected,
            actual=actual,
            context=context,
            confidence_threshold=confidence_threshold,
            structure_type=structure_type,
            **kwargs
        )
    
    async def validate_accessibility(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        category: AccessibilityCategory = AccessibilityCategory.PERCEIVABLE,
        wcag_level: WCAGLevel = WCAGLevel.AA,
        context: Optional[ValidationContext] = None,
        confidence_threshold: Optional[float] = None,
        **kwargs
    ) -> ValidationResult:
        """Perform accessibility validation."""
        return await self.validate(
            validation_type=ValidationType.ACCESSIBILITY,
            expected=expected,
            actual=actual,
            context=context,
            confidence_threshold=confidence_threshold,
            category=category,
            wcag_level=wcag_level,
            **kwargs
        )
    
    def get_validator(self, validation_type: ValidationType) -> Optional[ValidationStrategy]:
        """Get validator instance for a specific type."""
        return self._validators.get(validation_type)
    
    def register_validator(
        self,
        validation_type: ValidationType,
        validator: ValidationStrategy
    ) -> None:
        """Register a custom validator."""
        self._validators[validation_type] = validator
        logging.info(f"Registered custom validator for type: {validation_type.value}")
    
    def list_validators(self) -> List[ValidationType]:
        """List all registered validation types."""
        return list(self._validators.keys())
    
    def configure_validator(
        self,
        validation_type: ValidationType,
        **config_kwargs
    ) -> None:
        """Configure a validator with new settings."""
        validator = self._validators.get(validation_type)
        if not validator:
            raise ValidationError(f"No validator registered for type: {validation_type}")
        
        # Apply configuration (this would depend on validator implementation)
        for key, value in config_kwargs.items():
            if hasattr(validator, key):
                setattr(validator, key, value)
        
        logging.info(f"Configured validator {validation_type.value} with {len(config_kwargs)} settings")
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return None
    
    def clear_cache(self) -> None:
        """Clear validation cache."""
        if self.cache:
            self.cache.clear()
            logging.info("Validation cache cleared")
    
    def cleanup_cache(self) -> int:
        """Clean up expired cache entries."""
        if self.cache:
            return self.cache.cleanup_expired()
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats = self._stats.copy()
        
        # Add derived statistics
        total_validations = stats["validation_successes"] + stats["validation_failures"]
        if total_validations > 0:
            stats["success_rate"] = stats["validation_successes"] / total_validations
        else:
            stats["success_rate"] = 0.0
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            stats["cache_stats"] = cache_stats
        
        stats["registered_validators"] = len(self._validators)
        stats["validation_mode"] = self.validation_mode.value
        stats["default_confidence_threshold"] = self.default_confidence_threshold
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset registry statistics."""
        self._stats = {
            "validations_performed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_failures": 0,
            "validation_successes": 0
        }
        logging.info("Registry statistics reset")
    
    def create_context(
        self,
        validation_type: ValidationType,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationContext:
        """Create a validation context."""
        return ValidationContext(
            validation_type=validation_type,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
    
    def set_default_confidence_threshold(self, threshold: float) -> None:
        """Set default confidence threshold for all validators."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self.default_confidence_threshold = threshold
        
        # Update all validators
        for validator in self._validators.values():
            validator.default_confidence_threshold = threshold
        
        logging.info(f"Updated default confidence threshold to {threshold}")
    
    def set_validation_mode(self, mode: ValidationMode) -> None:
        """Set validation execution mode."""
        self.validation_mode = mode
        logging.info(f"Set validation mode to {mode.value}")
    
    async def validate_multiple(
        self,
        validations: List[Dict[str, Any]],
        fail_fast: bool = False,
        context: Optional[ValidationContext] = None
    ) -> List[ValidationResult]:
        """
        Perform multiple validations.
        
        Args:
            validations: List of validation specifications
            fail_fast: Stop on first failure
            context: Shared validation context
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        for i, validation_spec in enumerate(validations):
            validation_type = ValidationType(validation_spec["type"])
            expected = validation_spec["expected"]
            actual = validation_spec["actual"]
            
            # Create individual context
            individual_context = ValidationContext(
                validation_type=validation_type,
                correlation_id=context.correlation_id if context else None,
                metadata={
                    "batch_validation": True,
                    "batch_index": i,
                    "batch_size": len(validations),
                    **(validation_spec.get("metadata", {}))
                }
            )
            
            try:
                result = await self.validate(
                    validation_type=validation_type,
                    expected=expected,
                    actual=actual,
                    context=individual_context,
                    **validation_spec.get("kwargs", {})
                )
                
                results.append(result)
                
                # Fail fast if enabled and validation failed
                if fail_fast and not result.is_successful():
                    break
                    
            except Exception as e:
                # Create error result
                error_result = ValidationResult(
                    status=ValidationStatus.FAILED,
                    confidence_score=ConfidenceScore(value=0.0, method="error"),
                    validation_type=validation_type,
                    context=individual_context,
                    message=f"Validation error: {str(e)}"
                )
                
                results.append(error_result)
                
                if fail_fast:
                    break
        
        return results


# Global registry instance for easy access
_global_registry: Optional[ValidationRegistry] = None


def get_global_registry() -> ValidationRegistry:
    """Get or create global validation registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ValidationRegistry()
    return _global_registry


def set_global_registry(registry: ValidationRegistry) -> None:
    """Set global validation registry."""
    global _global_registry
    _global_registry = registry