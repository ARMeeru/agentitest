# validation/__init__.py
"""
Robust Test Validation Framework for AgentiTest

This package provides comprehensive validation strategies that replace simple string matching
with semantic validation, confidence scoring, and multiple validation approaches for reliable
test assertions.

Key Components:
- SemanticValidator: Confidence-scored semantic matching
- TextContentValidator: Advanced text content validation with fuzzy matching
- VisualElementValidator: Visual element presence and property validation
- DOMStructureValidator: DOM structure and hierarchy validation
- AccessibilityValidator: WCAG compliance and accessibility checks
- ValidationResultCache: Caching for deterministic and repeatable results
- CustomAssertions: Domain-specific assertion helpers

Usage:
    from validation import (
        SemanticValidator,
        TextContentValidator,
        ValidationResult,
        ValidationStrategy
    )
    
    validator = SemanticValidator()
    result = await validator.validate(
        actual="Page loaded successfully",
        expected="successful page load",
        confidence_threshold=0.8
    )
"""

from .core import (
    ValidationResult,
    ValidationStrategy,
    ValidationContext,
    ValidationStatus,
    ValidationType,
    ConfidenceScore,
    ValidationError as ValidationFrameworkError,
)

from .semantic import SemanticValidator
from .text_content import TextContentValidator, TextMatchingMode
from .visual_element import VisualElementValidator
from .dom_structure import DOMStructureValidator
from .accessibility import AccessibilityValidator
from .cache import ValidationResultCache
from .assertions import CustomAssertions

# Main validator registry for easy access
from .registry import ValidationRegistry, get_global_registry

__all__ = [
    # Core validation framework
    "ValidationResult",
    "ValidationStrategy", 
    "ValidationContext",
    "ValidationStatus",
    "ValidationType",
    "ConfidenceScore",
    "ValidationFrameworkError",
    
    # Validation strategies
    "SemanticValidator",
    "TextContentValidator",
    "TextMatchingMode", 
    "VisualElementValidator",
    "DOMStructureValidator",
    "AccessibilityValidator",
    
    # Support components
    "ValidationResultCache",
    "CustomAssertions",
    "ValidationRegistry",
    "get_global_registry",
]

# Framework version for compatibility tracking
__version__ = "1.0.0"