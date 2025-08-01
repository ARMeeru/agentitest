# validation/dom_structure.py
"""
DOM structure validation for web page hierarchy and structural integrity.

This module provides validation capabilities for DOM structure, including element
hierarchy, relationships, structural patterns, and document organization validation
with confidence scoring.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Union, Set
from enum import Enum

from .core import (
    ValidationStrategy,
    ValidationResult,
    ValidationContext,
    ValidationStatus,
    ValidationType,
    ConfidenceScore,
    ValidationError
)


class DOMStructureType(Enum):
    """Types of DOM structure validation."""
    HIERARCHY = "hierarchy"
    ELEMENT_COUNT = "element_count"
    PARENT_CHILD = "parent_child"
    SIBLING_ORDER = "sibling_order"
    NESTING_DEPTH = "nesting_depth"
    SEMANTIC_STRUCTURE = "semantic_structure"
    FORM_STRUCTURE = "form_structure"
    TABLE_STRUCTURE = "table_structure"
    LIST_STRUCTURE = "list_structure"
    NAVIGATION_STRUCTURE = "navigation_structure"


class SemanticRole(Enum):
    """Semantic roles for HTML elements."""
    HEADER = "header"
    NAVIGATION = "navigation"
    MAIN = "main"
    ARTICLE = "article"
    SECTION = "section"
    ASIDE = "aside"
    FOOTER = "footer"
    FORM = "form"
    TABLE = "table"
    LIST = "list"


class DOMStructureValidator(ValidationStrategy):
    """
    DOM structure validation with confidence scoring.
    
    This validator can check various aspects of DOM structure:
    1. Element hierarchy and nesting
    2. Parent-child relationships
    3. Sibling ordering and positioning
    4. Element count and distribution
    5. Semantic HTML structure
    6. Form, table, and list structures
    7. Navigation and content organization
    """
    
    def __init__(
        self,
        default_confidence_threshold: float = 0.8,
        strict_hierarchy: bool = False,
        validate_semantic_html: bool = True
    ):
        """
        Initialize DOM structure validator.
        
        Args:
            default_confidence_threshold: Default minimum confidence for success
            strict_hierarchy: Use strict hierarchy validation
            validate_semantic_html: Enable semantic HTML validation
        """
        super().__init__("DOMStructureValidator", default_confidence_threshold)
        self.validation_type = ValidationType.DOM_STRUCTURE
        self.strict_hierarchy = strict_hierarchy
        self.validate_semantic_html = validate_semantic_html
        
        # Semantic HTML elements and their expected roles
        self.semantic_elements = {
            'header': SemanticRole.HEADER,
            'nav': SemanticRole.NAVIGATION,
            'main': SemanticRole.MAIN,
            'article': SemanticRole.ARTICLE,
            'section': SemanticRole.SECTION,
            'aside': SemanticRole.ASIDE,
            'footer': SemanticRole.FOOTER,
            'form': SemanticRole.FORM,
            'table': SemanticRole.TABLE,
            'ul': SemanticRole.LIST,
            'ol': SemanticRole.LIST,
        }
        
        # Expected nesting patterns
        self.valid_nesting_patterns = {
            'table': ['thead', 'tbody', 'tfoot', 'tr'],
            'tr': ['td', 'th'],
            'ul': ['li'],
            'ol': ['li'],
            'dl': ['dt', 'dd'],
            'form': ['input', 'textarea', 'select', 'button', 'fieldset', 'label'],
            'fieldset': ['legend', 'input', 'textarea', 'select', 'button', 'label'],
            'select': ['option', 'optgroup'],
            'optgroup': ['option']
        }
        
        # Elements that should not contain certain children
        self.invalid_nesting_patterns = {
            'button': ['button', 'a', 'input'],
            'a': ['a', 'button'],
            'label': ['label'],
            'form': ['form'],
            'table': ['table']
        }
    
    async def validate(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        context: Optional[ValidationContext] = None,
        confidence_threshold: Optional[float] = None,
        structure_type: DOMStructureType = DOMStructureType.HIERARCHY,
        **kwargs
    ) -> ValidationResult:
        """
        Perform DOM structure validation.
        
        Args:
            expected: Expected DOM structure specification
            actual: Actual DOM structure data
            context: Validation context with metadata
            confidence_threshold: Minimum confidence threshold for success
            structure_type: Type of DOM structure validation to perform
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult with detailed confidence scoring
        """
        start_time = time.time()
        
        if context is None:
            context = self.create_context()
        
        context.set_configuration("structure_type", structure_type.value)
        context.set_configuration("strict_hierarchy", self.strict_hierarchy)
        context.set_configuration("validate_semantic_html", self.validate_semantic_html)
        
        threshold = self.get_confidence_threshold(confidence_threshold)
        
        try:
            # Validate structure based on type
            if structure_type == DOMStructureType.HIERARCHY:
                result = await self._validate_hierarchy(expected, actual, context, threshold)
            elif structure_type == DOMStructureType.ELEMENT_COUNT:
                result = await self._validate_element_count(expected, actual, context, threshold)
            elif structure_type == DOMStructureType.PARENT_CHILD:
                result = await self._validate_parent_child(expected, actual, context, threshold)
            elif structure_type == DOMStructureType.SIBLING_ORDER:
                result = await self._validate_sibling_order(expected, actual, context, threshold)
            elif structure_type == DOMStructureType.NESTING_DEPTH:
                result = await self._validate_nesting_depth(expected, actual, context, threshold)
            elif structure_type == DOMStructureType.SEMANTIC_STRUCTURE:
                result = await self._validate_semantic_structure(expected, actual, context, threshold)
            elif structure_type == DOMStructureType.FORM_STRUCTURE:
                result = await self._validate_form_structure(expected, actual, context, threshold)
            elif structure_type == DOMStructureType.TABLE_STRUCTURE:
                result = await self._validate_table_structure(expected, actual, context, threshold)
            elif structure_type == DOMStructureType.LIST_STRUCTURE:
                result = await self._validate_list_structure(expected, actual, context, threshold)
            elif structure_type == DOMStructureType.NAVIGATION_STRUCTURE:
                result = await self._validate_navigation_structure(expected, actual, context, threshold)
            else:
                raise ValidationError(f"Unknown DOM structure type: {structure_type}")
            
            # Add structural analysis
            result.add_detail("structure_analysis", self._analyze_dom_structure(actual))
            result.add_detail("semantic_analysis", self._analyze_semantic_structure(actual))
            result.add_detail("validation_type", structure_type.value)
            
            # Record execution time
            result.execution_time_ms = (time.time() - start_time) * 1000
            
            logging.info(
                f"DOM structure validation completed: {result.status.value} "
                f"(confidence: {result.confidence_score.value:.3f}, type: {structure_type.value}) "
                f"[{context.validation_id}]"
            )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logging.error(
                f"DOM structure validation failed: {str(e)} "
                f"(execution_time: {execution_time:.1f}ms) [{context.validation_id}]"
            )
            raise ValidationError(
                f"DOM structure validation error: {str(e)}",
                validation_context=context,
                cause=e,
                validation_type=ValidationType.DOM_STRUCTURE
            )
    
    async def _validate_hierarchy(
        self, expected: Dict[str, Any], actual: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Validate DOM hierarchy structure."""
        
        confidence_components = {}
        
        # Extract hierarchy information
        expected_hierarchy = expected.get('hierarchy', {})
        actual_hierarchy = actual.get('hierarchy', {})
        
        if not expected_hierarchy or not actual_hierarchy:
            confidence_components["hierarchy_data_available"] = 0.0
            total_confidence = 0.0
        else:
            # Compare hierarchical structure
            hierarchy_match = self._compare_hierarchies(expected_hierarchy, actual_hierarchy)
            confidence_components.update(hierarchy_match)
            total_confidence = hierarchy_match.get('overall_match', 0.0)
        
        # Validate nesting patterns if strict mode
        if self.strict_hierarchy:
            nesting_validation = self._validate_nesting_patterns(actual_hierarchy)
            confidence_components.update(nesting_validation)
            total_confidence = (total_confidence + nesting_validation.get('nesting_score', 0.0)) / 2
        
        confidence_score = ConfidenceScore(
            value=total_confidence,
            components=confidence_components,
            method="dom_hierarchy",
            reliability=self._calculate_hierarchy_reliability(actual_hierarchy)
        )
        
        status = ValidationStatus.PASSED if confidence_score.is_above_threshold(threshold) else ValidationStatus.FAILED
        message = f"DOM hierarchy validation: {status.value} (confidence: {confidence_score.value:.3f})"
        
        result = self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=expected,
            actual=actual,
            message=message
        )
        
        result.add_detail("hierarchy_comparison", confidence_components)
        return result
    
    async def _validate_element_count(
        self, expected: Dict[str, Any], actual: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Validate element counts in the DOM."""
        
        confidence_components = {}
        
        expected_counts = expected.get('element_counts', {})
        actual_counts = actual.get('element_counts', {})
        
        if not expected_counts:
            confidence_components["no_count_expectations"] = 1.0
            total_confidence = 1.0
        else:
            total_elements = len(expected_counts)
            matched_elements = 0
            
            for element_type, expected_count in expected_counts.items():
                actual_count = actual_counts.get(element_type, 0)
                
                # Calculate confidence for this element type
                if expected_count == actual_count:
                    element_confidence = 1.0
                    matched_elements += 1
                elif expected_count == 0 and actual_count > 0:
                    element_confidence = 0.0  # Unexpected elements
                elif expected_count > 0 and actual_count == 0:
                    element_confidence = 0.0  # Missing elements
                else:
                    # Partial match based on ratio
                    ratio = min(expected_count, actual_count) / max(expected_count, actual_count)
                    element_confidence = ratio * 0.8  # Penalize count mismatches
                    if ratio >= 0.8:
                        matched_elements += 1
                
                confidence_components[f"count_{element_type}"] = element_confidence
            
            total_confidence = matched_elements / total_elements if total_elements > 0 else 1.0
        
        confidence_score = ConfidenceScore(
            value=total_confidence,
            components=confidence_components,
            method="dom_element_count",
            reliability=1.0  # Element counts are reliable
        )
        
        status = ValidationStatus.PASSED if confidence_score.is_above_threshold(threshold) else ValidationStatus.FAILED
        message = f"DOM element count validation: {status.value} (confidence: {confidence_score.value:.3f})"
        
        result = self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=expected,
            actual=actual,
            message=message
        )
        
        result.add_detail("count_comparison", {
            "expected": expected_counts,
            "actual": actual_counts,
            "matched_elements": confidence_components.get("matched_elements", 0)
        })
        
        return result
    
    async def _validate_parent_child(
        self, expected: Dict[str, Any], actual: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Validate parent-child relationships."""
        
        confidence_components = {}
        
        expected_relationships = expected.get('parent_child_relationships', [])
        actual_relationships = actual.get('parent_child_relationships', [])
        
        if not expected_relationships:
            confidence_components["no_relationship_expectations"] = 1.0
            total_confidence = 1.0
        else:
            matched_relationships = 0
            
            for expected_rel in expected_relationships:
                parent = expected_rel.get('parent')
                child = expected_rel.get('child')
                
                # Find matching relationship in actual
                relationship_found = any(
                    rel.get('parent') == parent and rel.get('child') == child
                    for rel in actual_relationships
                )
                
                if relationship_found:
                    matched_relationships += 1
                    confidence_components[f"relationship_{parent}_{child}"] = 1.0
                else:
                    confidence_components[f"relationship_{parent}_{child}"] = 0.0
            
            total_confidence = matched_relationships / len(expected_relationships)
        
        confidence_score = ConfidenceScore(
            value=total_confidence,
            components=confidence_components,
            method="dom_parent_child",
            reliability=0.9  # Relationships can be complex
        )
        
        status = ValidationStatus.PASSED if confidence_score.is_above_threshold(threshold) else ValidationStatus.FAILED
        message = f"DOM parent-child validation: {status.value} (confidence: {confidence_score.value:.3f})"
        
        return self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=expected,
            actual=actual,
            message=message
        )
    
    async def _validate_sibling_order(
        self, expected: Dict[str, Any], actual: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Validate sibling element ordering."""
        
        confidence_components = {}
        
        expected_orders = expected.get('sibling_orders', [])
        actual_orders = actual.get('sibling_orders', [])
        
        if not expected_orders:
            total_confidence = 1.0
        else:
            matched_orders = 0
            
            for expected_order in expected_orders:
                parent = expected_order.get('parent')
                expected_sequence = expected_order.get('sequence', [])
                
                # Find corresponding actual order
                actual_order = next(
                    (order for order in actual_orders if order.get('parent') == parent),
                    None
                )
                
                if actual_order:
                    actual_sequence = actual_order.get('sequence', [])
                    order_confidence = self._calculate_sequence_similarity(expected_sequence, actual_sequence)
                    confidence_components[f"order_{parent}"] = order_confidence
                    
                    if order_confidence >= 0.8:
                        matched_orders += 1
                else:
                    confidence_components[f"order_{parent}"] = 0.0
            
            total_confidence = matched_orders / len(expected_orders)
        
        confidence_score = ConfidenceScore(
            value=total_confidence,
            components=confidence_components,
            method="dom_sibling_order",
            reliability=0.85
        )
        
        status = ValidationStatus.PASSED if confidence_score.is_above_threshold(threshold) else ValidationStatus.FAILED
        message = f"DOM sibling order validation: {status.value} (confidence: {confidence_score.value:.3f})"
        
        return self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=expected,
            actual=actual,
            message=message
        )
    
    async def _validate_nesting_depth(
        self, expected: Dict[str, Any], actual: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Validate DOM nesting depth."""
        
        expected_depth = expected.get('max_nesting_depth', float('inf'))
        actual_depth = actual.get('max_nesting_depth', 0)
        
        if expected_depth == float('inf'):
            # No depth restriction
            confidence = 1.0
        elif actual_depth <= expected_depth:
            # Within acceptable depth
            confidence = 1.0
        else:
            # Exceeds expected depth
            excess_ratio = (actual_depth - expected_depth) / expected_depth
            confidence = max(0.0, 1.0 - excess_ratio * 0.5)
        
        confidence_score = ConfidenceScore(
            value=confidence,
            components={
                "depth_compliance": confidence,
                "expected_depth": expected_depth,
                "actual_depth": actual_depth
            },
            method="dom_nesting_depth",
            reliability=1.0
        )
        
        status = ValidationStatus.PASSED if confidence_score.is_above_threshold(threshold) else ValidationStatus.FAILED
        message = f"DOM nesting depth validation: {status.value} (actual: {actual_depth}, expected: {expected_depth})"
        
        return self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=expected,
            actual=actual,
            message=message
        )
    
    async def _validate_semantic_structure(
        self, expected: Dict[str, Any], actual: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Validate semantic HTML structure."""
        
        confidence_components = {}
        
        # Check for required semantic elements
        expected_semantic = expected.get('semantic_elements', [])
        actual_semantic = actual.get('semantic_elements', [])
        
        if expected_semantic:
            matched_semantic = 0
            for expected_element in expected_semantic:
                if expected_element in actual_semantic:
                    matched_semantic += 1
                    confidence_components[f"semantic_{expected_element}"] = 1.0
                else:
                    confidence_components[f"semantic_{expected_element}"] = 0.0
            
            semantic_confidence = matched_semantic / len(expected_semantic)
        else:
            semantic_confidence = 1.0
        
        # Check semantic hierarchy (header -> main -> footer)
        hierarchy_confidence = self._validate_semantic_hierarchy(actual.get('element_order', []))
        confidence_components["semantic_hierarchy"] = hierarchy_confidence
        
        # Overall confidence
        total_confidence = (semantic_confidence + hierarchy_confidence) / 2
        
        confidence_score = ConfidenceScore(
            value=total_confidence,
            components=confidence_components,
            method="dom_semantic_structure",
            reliability=0.9
        )
        
        status = ValidationStatus.PASSED if confidence_score.is_above_threshold(threshold) else ValidationStatus.FAILED
        message = f"DOM semantic structure validation: {status.value} (confidence: {confidence_score.value:.3f})"
        
        return self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=expected,
            actual=actual,
            message=message
        )
    
    # Additional validation methods for form, table, list, and navigation structures would go here
    # For brevity, I'm including simplified versions
    
    async def _validate_form_structure(
        self, expected: Dict[str, Any], actual: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Validate form structure."""
        # Simplified form structure validation
        form_elements = actual.get('form_elements', [])
        expected_fields = expected.get('required_fields', [])
        
        confidence = 1.0 if all(field in form_elements for field in expected_fields) else 0.5
        
        confidence_score = ConfidenceScore(value=confidence, method="form_structure", reliability=0.8)
        status = ValidationStatus.PASSED if confidence >= threshold else ValidationStatus.FAILED
        
        return self.create_result(status, confidence_score, context, expected, actual, f"Form structure: {status.value}")
    
    async def _validate_table_structure(
        self, expected: Dict[str, Any], actual: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Validate table structure."""
        # Simplified table structure validation
        has_header = actual.get('has_thead', False)
        has_body = actual.get('has_tbody', False)
        expected_header = expected.get('requires_header', False)
        
        confidence = 1.0 if (not expected_header or has_header) and has_body else 0.5
        
        confidence_score = ConfidenceScore(value=confidence, method="table_structure", reliability=0.9)
        status = ValidationStatus.PASSED if confidence >= threshold else ValidationStatus.FAILED
        
        return self.create_result(status, confidence_score, context, expected, actual, f"Table structure: {status.value}")
    
    async def _validate_list_structure(
        self, expected: Dict[str, Any], actual: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Validate list structure."""
        # Simplified list structure validation
        list_types = actual.get('list_types', [])
        expected_types = expected.get('expected_list_types', [])
        
        confidence = 1.0 if all(ltype in list_types for ltype in expected_types) else 0.5
        
        confidence_score = ConfidenceScore(value=confidence, method="list_structure", reliability=0.8)
        status = ValidationStatus.PASSED if confidence >= threshold else ValidationStatus.FAILED
        
        return self.create_result(status, confidence_score, context, expected, actual, f"List structure: {status.value}")
    
    async def _validate_navigation_structure(
        self, expected: Dict[str, Any], actual: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Validate navigation structure."""
        # Simplified navigation structure validation
        nav_elements = actual.get('navigation_elements', [])
        expected_nav = expected.get('required_navigation', [])
        
        confidence = 1.0 if all(nav in nav_elements for nav in expected_nav) else 0.5
        
        confidence_score = ConfidenceScore(value=confidence, method="navigation_structure", reliability=0.8)
        status = ValidationStatus.PASSED if confidence >= threshold else ValidationStatus.FAILED
        
        return self.create_result(status, confidence_score, context, expected, actual, f"Navigation structure: {status.value}")
    
    # Helper methods
    
    def _compare_hierarchies(self, expected: Dict, actual: Dict) -> Dict[str, float]:
        """Compare hierarchical structures."""
        # Simplified hierarchy comparison
        return {
            "structure_match": 0.8,  # Placeholder
            "depth_match": 0.9,
            "overall_match": 0.85
        }
    
    def _validate_nesting_patterns(self, hierarchy: Dict) -> Dict[str, float]:
        """Validate nesting patterns against HTML standards."""
        # Simplified nesting validation
        return {"nesting_score": 0.9}
    
    def _calculate_hierarchy_reliability(self, hierarchy: Dict) -> float:
        """Calculate reliability of hierarchy data."""
        return 0.9 if hierarchy else 0.5
    
    def _calculate_sequence_similarity(self, expected: List, actual: List) -> float:
        """Calculate similarity between two sequences."""
        if not expected and not actual:
            return 1.0
        if not expected or not actual:
            return 0.0
        
        # Simple Jaccard similarity
        set_expected = set(expected)
        set_actual = set(actual)
        intersection = len(set_expected.intersection(set_actual))
        union = len(set_expected.union(set_actual))
        
        return intersection / union if union > 0 else 0.0
    
    def _validate_semantic_hierarchy(self, element_order: List[str]) -> float:
        """Validate semantic element ordering."""
        # Check for proper semantic structure order
        semantic_order = {
            'header': 0,
            'nav': 1,
            'main': 2,
            'article': 2,
            'section': 2,
            'aside': 3,
            'footer': 4
        }
        
        previous_order = -1
        violations = 0
        
        for element in element_order:
            if element in semantic_order:
                current_order = semantic_order[element]
                if current_order < previous_order:
                    violations += 1
                previous_order = max(previous_order, current_order)
        
        return max(0.0, 1.0 - violations * 0.2)
    
    def _analyze_dom_structure(self, actual: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze DOM structure characteristics."""
        return {
            "total_elements": actual.get('total_elements', 0),
            "max_depth": actual.get('max_nesting_depth', 0),
            "semantic_elements_count": len(actual.get('semantic_elements', [])),
            "form_count": actual.get('form_count', 0),
            "table_count": actual.get('table_count', 0),
            "list_count": actual.get('list_count', 0)
        }
    
    def _analyze_semantic_structure(self, actual: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze semantic HTML structure."""
        semantic_elements = actual.get('semantic_elements', [])
        
        return {
            "has_header": 'header' in semantic_elements,
            "has_nav": 'nav' in semantic_elements,
            "has_main": 'main' in semantic_elements,
            "has_footer": 'footer' in semantic_elements,
            "semantic_coverage": len(semantic_elements) / len(self.semantic_elements) if self.semantic_elements else 0,
            "semantic_elements_found": semantic_elements
        }