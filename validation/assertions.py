# validation/assertions.py
"""
Custom assertion helpers for common web testing patterns.

This module provides high-level assertion methods that combine multiple validation
strategies to create domain-specific assertions for web testing scenarios.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from functools import wraps

from .core import (
    ValidationResult,
    ValidationContext,
    ValidationStatus,
    ConfidenceScore,
    ValidationError,
    ValidationType
)
from .semantic import SemanticValidator
from .text_content import TextContentValidator, TextMatchingMode
from .visual_element import VisualElementValidator
from .dom_structure import DOMStructureValidator, DOMStructureType
from .accessibility import AccessibilityValidator, AccessibilityCategory, WCAGLevel


class AssertionError(Exception):
    """Custom assertion error with validation context."""
    
    def __init__(self, message: str, validation_result: Optional[ValidationResult] = None):
        super().__init__(message)
        self.message = message
        self.validation_result = validation_result


def validation_assertion(func: Callable) -> Callable:
    """Decorator for validation assertion methods."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            if isinstance(result, ValidationResult):
                if not result.is_successful():
                    raise AssertionError(
                        f"Assertion failed: {result.message}",
                        validation_result=result
                    )
                return result
            return result
        except ValidationError as e:
            raise AssertionError(f"Validation error: {str(e)}")
        except Exception as e:
            raise AssertionError(f"Assertion error: {str(e)}")
    
    return wrapper


class CustomAssertions:
    """
    Custom assertion helpers for web testing patterns.
    
    This class provides high-level assertion methods that encapsulate common
    web testing scenarios using the validation framework's strategies.
    """
    
    def __init__(
        self,
        default_confidence_threshold: float = 0.8,
        enable_detailed_logging: bool = True
    ):
        """
        Initialize custom assertions.
        
        Args:
            default_confidence_threshold: Default confidence threshold for all assertions
            enable_detailed_logging: Enable detailed logging of assertion results
        """
        self.default_confidence_threshold = default_confidence_threshold
        self.enable_detailed_logging = enable_detailed_logging
        
        # Initialize validators
        self.semantic_validator = SemanticValidator()
        self.text_validator = TextContentValidator()
        self.visual_validator = VisualElementValidator()
        self.dom_validator = DOMStructureValidator()
        self.accessibility_validator = AccessibilityValidator()
        
        # Common web testing patterns
        self.web_patterns = {
            "login_success": ["success", "welcome", "dashboard", "logged in", "account"],
            "login_failure": ["error", "invalid", "incorrect", "failed", "denied"],
            "form_validation": ["required", "invalid", "error", "please", "must"],
            "loading_states": ["loading", "please wait", "processing", "spinner"],
            "navigation_success": ["loaded", "page", "content", "title"],
            "search_results": ["results", "found", "matches", "search", "items"],
            "no_results": ["no results", "nothing found", "empty", "zero results"]
        }
    
    @validation_assertion
    async def assert_page_loaded(
        self,
        actual_content: str,
        expected_indicators: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Assert that a page has loaded successfully.
        
        Args:
            actual_content: Actual page content or response
            expected_indicators: Expected indicators of successful page load
            confidence_threshold: Minimum confidence threshold
            context: Validation context
            
        Returns:
            ValidationResult for the page load assertion
        """
        if expected_indicators is None:
            expected_indicators = ["loaded", "content", "page", "title", "body"]
        
        if context is None:
            context = ValidationContext(
                metadata={"assertion_type": "page_loaded"}
            )
        
        result = await self.semantic_validator.validate(
            expected=expected_indicators,
            actual=actual_content,
            context=context,
            confidence_threshold=confidence_threshold or self.default_confidence_threshold
        )
        
        if self.enable_detailed_logging:
            logging.info(f"Page load assertion: {result.status.value} [{context.validation_id}]")
        
        return result
    
    @validation_assertion
    async def assert_login_success(
        self,
        actual_response: str,
        user_context: Optional[Dict[str, str]] = None,
        confidence_threshold: Optional[float] = None,
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Assert successful user login.
        
        Args:
            actual_response: Actual login response or page content
            user_context: User context (username, expected welcome message, etc.)
            confidence_threshold: Minimum confidence threshold
            context: Validation context
            
        Returns:
            ValidationResult for the login success assertion
        """
        expected_patterns = self.web_patterns["login_success"].copy()
        
        # Add user-specific patterns if provided
        if user_context:
            username = user_context.get("username")
            if username:
                expected_patterns.extend([f"welcome {username}", f"hello {username}"])
        
        if context is None:
            context = ValidationContext(
                metadata={
                    "assertion_type": "login_success",
                    "user_context": user_context or {}
                }
            )
        
        result = await self.semantic_validator.validate(
            expected=expected_patterns,
            actual=actual_response,
            context=context,
            confidence_threshold=confidence_threshold or self.default_confidence_threshold,
            enable_partial_matching=True
        )
        
        if self.enable_detailed_logging:
            logging.info(f"Login success assertion: {result.status.value} [{context.validation_id}]")
        
        return result
    
    @validation_assertion
    async def assert_form_validation_error(
        self,
        actual_response: str,
        field_name: Optional[str] = None,
        error_type: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Assert form validation error is displayed.
        
        Args:
            actual_response: Actual form response or page content
            field_name: Specific field that should have error
            error_type: Type of validation error (required, invalid, etc.)
            confidence_threshold: Minimum confidence threshold
            context: Validation context
            
        Returns:
            ValidationResult for the form validation error assertion
        """
        expected_patterns = self.web_patterns["form_validation"].copy()
        
        # Add specific patterns based on field and error type
        if field_name:
            expected_patterns.extend([
                f"{field_name} required",
                f"{field_name} invalid",
                f"{field_name} error"
            ])
        
        if error_type:
            expected_patterns.append(f"{error_type}")
        
        if context is None:
            context = ValidationContext(
                metadata={
                    "assertion_type": "form_validation_error",
                    "field_name": field_name,
                    "error_type": error_type
                }
            )
        
        result = await self.semantic_validator.validate(
            expected=expected_patterns,
            actual=actual_response,
            context=context,
            confidence_threshold=confidence_threshold or self.default_confidence_threshold,
            ignore_case=True,
            enable_partial_matching=True
        )
        
        if self.enable_detailed_logging:
            logging.info(f"Form validation error assertion: {result.status.value} [{context.validation_id}]")
        
        return result
    
    @validation_assertion
    async def assert_search_results_displayed(
        self,
        actual_content: str,
        search_term: str,
        minimum_results: int = 1,
        confidence_threshold: Optional[float] = None,
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Assert search results are displayed for a search term.
        
        Args:
            actual_content: Actual search results page content
            search_term: The search term that was queried
            minimum_results: Minimum number of results expected
            confidence_threshold: Minimum confidence threshold
            context: Validation context
            
        Returns:
            ValidationResult for the search results assertion
        """
        expected_patterns = [
            f"results for {search_term}",
            f"found {search_term}",
            f"{search_term} results",
            "search results",
            "results found"
        ]
        
        if context is None:
            context = ValidationContext(
                metadata={
                    "assertion_type": "search_results_displayed",
                    "search_term": search_term,
                    "minimum_results": minimum_results
                }
            )
        
        result = await self.semantic_validator.validate(
            expected=expected_patterns,
            actual=actual_content,
            context=context,
            confidence_threshold=confidence_threshold or self.default_confidence_threshold,
            ignore_case=True,
            enable_partial_matching=True
        )
        
        if self.enable_detailed_logging:
            logging.info(f"Search results assertion: {result.status.value} [{context.validation_id}]")
        
        return result
    
    @validation_assertion
    async def assert_no_search_results(
        self,
        actual_content: str,
        search_term: str,
        confidence_threshold: Optional[float] = None,
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Assert no search results are found for a search term.
        
        Args:
            actual_content: Actual search results page content
            search_term: The search term that was queried
            confidence_threshold: Minimum confidence threshold
            context: Validation context
            
        Returns:
            ValidationResult for the no search results assertion
        """
        expected_patterns = self.web_patterns["no_results"].copy()
        expected_patterns.extend([
            f"no results for {search_term}",
            f"{search_term} not found",
            "0 results"
        ])
        
        if context is None:
            context = ValidationContext(
                metadata={
                    "assertion_type": "no_search_results",
                    "search_term": search_term
                }
            )
        
        result = await self.semantic_validator.validate(
            expected=expected_patterns,
            actual=actual_content,
            context=context,
            confidence_threshold=confidence_threshold or self.default_confidence_threshold,
            ignore_case=True,
            enable_partial_matching=True
        )
        
        if self.enable_detailed_logging:
            logging.info(f"No search results assertion: {result.status.value} [{context.validation_id}]")
        
        return result
    
    @validation_assertion
    async def assert_element_visible(
        self,
        element_properties: Dict[str, Any],
        expected_visibility: bool = True,
        confidence_threshold: Optional[float] = None,
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Assert element visibility state.
        
        Args:
            element_properties: Actual element properties
            expected_visibility: Expected visibility state
            confidence_threshold: Minimum confidence threshold
            context: Validation context
            
        Returns:
            ValidationResult for the element visibility assertion
        """
        expected = {"visible": expected_visibility}
        
        if context is None:
            context = ValidationContext(
                metadata={
                    "assertion_type": "element_visible",
                    "expected_visibility": expected_visibility
                }
            )
        
        result = await self.visual_validator.validate(
            expected=expected,
            actual=element_properties,
            context=context,
            confidence_threshold=confidence_threshold or self.default_confidence_threshold
        )
        
        if self.enable_detailed_logging:
            logging.info(f"Element visibility assertion: {result.status.value} [{context.validation_id}]")
        
        return result
    
    @validation_assertion
    async def assert_text_content_matches(
        self,
        actual_text: str,
        expected_text: Union[str, List[str]],
        matching_mode: TextMatchingMode = TextMatchingMode.FUZZY,
        confidence_threshold: Optional[float] = None,
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Assert text content matches expected text.
        
        Args:
            actual_text: Actual text content
            expected_text: Expected text or list of acceptable texts
            matching_mode: Text matching strategy to use
            confidence_threshold: Minimum confidence threshold
            context: Validation context
            
        Returns:
            ValidationResult for the text content assertion
        """
        if context is None:
            context = ValidationContext(
                metadata={
                    "assertion_type": "text_content_matches",
                    "matching_mode": matching_mode.value
                }
            )
        
        result = await self.text_validator.validate(
            expected=expected_text,
            actual=actual_text,
            context=context,
            confidence_threshold=confidence_threshold or self.default_confidence_threshold,
            matching_mode=matching_mode
        )
        
        if self.enable_detailed_logging:
            logging.info(f"Text content assertion: {result.status.value} [{context.validation_id}]")
        
        return result
    
    @validation_assertion
    async def assert_accessibility_compliance(
        self,
        page_data: Dict[str, Any],
        wcag_level: WCAGLevel = WCAGLevel.AA,
        category: AccessibilityCategory = AccessibilityCategory.PERCEIVABLE,
        confidence_threshold: Optional[float] = None,
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Assert accessibility compliance for a page.
        
        Args:
            page_data: Page accessibility data
            wcag_level: WCAG compliance level to validate against
            category: Accessibility category to validate
            confidence_threshold: Minimum confidence threshold
            context: Validation context
            
        Returns:
            ValidationResult for the accessibility compliance assertion
        """
        if context is None:
            context = ValidationContext(
                metadata={
                    "assertion_type": "accessibility_compliance",
                    "wcag_level": wcag_level.value,
                    "category": category.value
                }
            )
        
        result = await self.accessibility_validator.validate(
            expected={},  # Use defaults from validator
            actual=page_data,
            context=context,
            confidence_threshold=confidence_threshold or self.default_confidence_threshold,
            category=category,
            wcag_level=wcag_level
        )
        
        if self.enable_detailed_logging:
            logging.info(f"Accessibility compliance assertion: {result.status.value} [{context.validation_id}]")
        
        return result
    
    @validation_assertion
    async def assert_dom_structure_valid(
        self,
        dom_data: Dict[str, Any],
        structure_type: DOMStructureType = DOMStructureType.HIERARCHY,
        expected_structure: Optional[Dict[str, Any]] = None,
        confidence_threshold: Optional[float] = None,
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Assert DOM structure validity.
        
        Args:
            dom_data: DOM structure data
            structure_type: Type of structure validation to perform
            expected_structure: Expected structure specification
            confidence_threshold: Minimum confidence threshold
            context: Validation context
            
        Returns:
            ValidationResult for the DOM structure assertion
        """
        if context is None:
            context = ValidationContext(
                metadata={
                    "assertion_type": "dom_structure_valid",
                    "structure_type": structure_type.value
                }
            )
        
        result = await self.dom_validator.validate(
            expected=expected_structure or {},
            actual=dom_data,
            context=context,
            confidence_threshold=confidence_threshold or self.default_confidence_threshold,
            structure_type=structure_type
        )
        
        if self.enable_detailed_logging:
            logging.info(f"DOM structure assertion: {result.status.value} [{context.validation_id}]")
        
        return result
    
    async def assert_multiple_conditions(
        self,
        conditions: List[Dict[str, Any]],
        require_all: bool = True,
        confidence_threshold: Optional[float] = None,
        context: Optional[ValidationContext] = None
    ) -> ValidationResult:
        """
        Assert multiple conditions with combined confidence scoring.
        
        Args:
            conditions: List of condition specifications
            require_all: Whether all conditions must pass (AND) or any (OR)
            confidence_threshold: Minimum confidence threshold
            context: Validation context
            
        Returns:
            Combined ValidationResult for all conditions
        """
        if context is None:
            context = ValidationContext(
                metadata={
                    "assertion_type": "multiple_conditions",
                    "require_all": require_all,
                    "condition_count": len(conditions)
                }
            )
        
        results = []
        
        # Execute all conditions
        for i, condition in enumerate(conditions):
            condition_type = condition.get("type", "semantic")
            condition_context = ValidationContext(
                correlation_id=context.correlation_id,
                metadata={"parent_assertion": context.validation_id, "condition_index": i}
            )
            
            try:
                if condition_type == "semantic":
                    result = await self.semantic_validator.validate(
                        expected=condition["expected"],
                        actual=condition["actual"],
                        context=condition_context,
                        confidence_threshold=0.0  # We'll handle threshold at the end
                    )
                elif condition_type == "text_content":
                    result = await self.text_validator.validate(
                        expected=condition["expected"],
                        actual=condition["actual"],
                        context=condition_context,
                        confidence_threshold=0.0,
                        matching_mode=condition.get("matching_mode", TextMatchingMode.FUZZY)
                    )
                elif condition_type == "visual_element":
                    result = await self.visual_validator.validate(
                        expected=condition["expected"],
                        actual=condition["actual"],
                        context=condition_context,
                        confidence_threshold=0.0
                    )
                else:
                    raise ValidationError(f"Unknown condition type: {condition_type}")
                
                results.append(result)
                
            except Exception as e:
                # Create failure result for this condition
                failure_result = ValidationResult(
                    status=ValidationStatus.FAILED,
                    confidence_score=ConfidenceScore(value=0.0, method="condition_error"),
                    validation_type=ValidationType.CUSTOM,
                    context=condition_context,
                    message=f"Condition {i} failed: {str(e)}"
                )
                results.append(failure_result)
        
        # Combine results
        if require_all:
            # All conditions must pass
            passed_conditions = sum(1 for r in results if r.is_successful())
            overall_confidence = passed_conditions / len(results) if results else 0.0
            
            # Use minimum confidence of passing conditions
            if passed_conditions > 0:
                passing_confidences = [r.confidence_score.value for r in results if r.is_successful()]
                overall_confidence = min(passing_confidences) * (passed_conditions / len(results))
            
            status = ValidationStatus.PASSED if passed_conditions == len(results) else ValidationStatus.FAILED
        else:
            # Any condition can pass
            passed_conditions = sum(1 for r in results if r.is_successful())
            status = ValidationStatus.PASSED if passed_conditions > 0 else ValidationStatus.FAILED
            
            # Use maximum confidence of all conditions
            overall_confidence = max((r.confidence_score.value for r in results), default=0.0)
        
        # Check against threshold
        threshold = confidence_threshold or self.default_confidence_threshold
        final_status = ValidationStatus.PASSED if overall_confidence >= threshold and status == ValidationStatus.PASSED else ValidationStatus.FAILED
        
        # Create combined result
        combined_confidence = ConfidenceScore(
            value=overall_confidence,
            components={f"condition_{i}": r.confidence_score.value for i, r in enumerate(results)},
            method="multiple_conditions",
            reliability=sum(r.confidence_score.reliability for r in results) / len(results) if results else 0.0
        )
        
        message = f"Multiple conditions assertion: {final_status.value} "
        message += f"({passed_conditions}/{len(results)} conditions passed, "
        message += f"confidence: {overall_confidence:.3f})"
        
        combined_result = ValidationResult(
            status=final_status,
            confidence_score=combined_confidence,
            validation_type=ValidationType.CUSTOM,
            context=context,
            expected_value=conditions,
            actual_value=[r.actual_value for r in results],
            message=message
        )
        
        # Add individual results as details
        combined_result.add_detail("individual_results", [r.to_dict() for r in results])
        combined_result.add_detail("require_all", require_all)
        combined_result.add_detail("conditions_passed", passed_conditions)
        
        if self.enable_detailed_logging:
            logging.info(f"Multiple conditions assertion: {final_status.value} [{context.validation_id}]")
        
        return combined_result
    
    def create_context(
        self,
        assertion_type: str,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationContext:
        """Create a validation context for assertions."""
        context_metadata = {"assertion_type": assertion_type}
        if metadata:
            context_metadata.update(metadata)
        
        return ValidationContext(
            correlation_id=correlation_id,
            metadata=context_metadata
        )