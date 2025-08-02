# validation/visual_element.py
"""
Visual element validation for browser automation testing.

This module provides validation capabilities for visual elements on web pages,
including element presence, visibility, position, size, styling, and visual
properties validation with confidence scoring.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
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


class ElementProperty(Enum):
    """Visual element properties that can be validated."""
    PRESENT = "present"
    VISIBLE = "visible"
    ENABLED = "enabled"
    CLICKABLE = "clickable"
    TEXT_CONTENT = "text_content"
    ATTRIBUTE_VALUE = "attribute_value"
    CSS_PROPERTY = "css_property"
    POSITION = "position"
    SIZE = "size"
    BOUNDS = "bounds"
    COLOR = "color"
    FONT = "font"
    BACKGROUND = "background"
    OPACITY = "opacity"
    Z_INDEX = "z_index"


class VisibilityLevel(Enum):
    """Different levels of element visibility."""
    HIDDEN = "hidden"
    PARTIALLY_VISIBLE = "partially_visible"
    FULLY_VISIBLE = "fully_visible"
    VIEWPORT_VISIBLE = "viewport_visible"


class VisualElementValidator(ValidationStrategy):
    """
    Visual element validation with confidence scoring.
    
    This validator can check various visual properties of web elements:
    1. Element presence and visibility
    2. Element positioning and sizing
    3. CSS styling and visual properties
    4. Element state (enabled, clickable, etc.)
    5. Text content and attributes
    6. Visual appearance validation
    """
    
    def __init__(
        self,
        default_confidence_threshold: float = 0.9,
        enable_screenshot_validation: bool = False,
        strict_positioning: bool = False
    ):
        """
        Initialize visual element validator.
        
        Args:
            default_confidence_threshold: Default minimum confidence for success
            enable_screenshot_validation: Enable visual screenshot comparison
            strict_positioning: Use strict positioning validation
        """
        super().__init__("VisualElementValidator", default_confidence_threshold)
        self.validation_type = ValidationType.VISUAL_ELEMENT
        self.enable_screenshot_validation = enable_screenshot_validation
        self.strict_positioning = strict_positioning
        
        # Common CSS properties for validation
        self.common_css_properties = {
            'display', 'visibility', 'opacity', 'color', 'background-color',
            'font-size', 'font-family', 'font-weight', 'text-align', 'width',
            'height', 'margin', 'padding', 'border', 'position', 'top', 'left',
            'right', 'bottom', 'z-index', 'overflow', 'cursor'
        }
        
        # Tolerance levels for numerical comparisons
        self.position_tolerance = 5  # pixels
        self.size_tolerance = 2  # pixels
        self.opacity_tolerance = 0.1  # opacity units
    
    async def validate(
        self,
        expected: Dict[str, Any],
        actual: Optional[Dict[str, Any]],
        context: Optional[ValidationContext] = None,
        confidence_threshold: Optional[float] = None,
        browser_session = None,
        element_selector: Optional[str] = None,
        **kwargs
    ) -> ValidationResult:
        """
        Perform visual element validation.
        
        Args:
            expected: Expected element properties (dict with property names and values)
            actual: Actual element properties (dict or None if element not found)
            context: Validation context with metadata
            confidence_threshold: Minimum confidence threshold for success
            browser_session: Browser session for additional element inspection
            element_selector: CSS selector for the element being validated
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult with detailed confidence scoring
        """
        start_time = time.time()
        
        if context is None:
            context = self.create_context()
        
        context.set_configuration("element_selector", element_selector)
        context.set_configuration("browser_session_available", browser_session is not None)
        
        threshold = self.get_confidence_threshold(confidence_threshold)
        
        try:
            # If actual is None, element was not found
            if actual is None:
                if expected.get(ElementProperty.PRESENT.value, True):
                    # Expected present but not found
                    result = await self._handle_element_not_found(expected, context, threshold)
                else:
                    # Expected not present and indeed not found - success
                    result = await self._handle_element_correctly_absent(expected, context, threshold)
            else:
                # Element found, validate properties
                result = await self._validate_element_properties(
                    expected, actual, context, threshold, browser_session, element_selector
                )
            
            # Add element-specific details
            result.add_detail("element_selector", element_selector)
            result.add_detail("validation_properties", list(expected.keys()))
            result.add_detail("tolerances", {
                "position_tolerance": self.position_tolerance,
                "size_tolerance": self.size_tolerance,
                "opacity_tolerance": self.opacity_tolerance
            })
            
            # Record execution time
            result.execution_time_ms = (time.time() - start_time) * 1000
            
            logging.info(
                f"Visual element validation completed: {result.status.value} "
                f"(confidence: {result.confidence_score.value:.3f}) "
                f"[{context.validation_id}]"
            )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logging.error(
                f"Visual element validation failed: {str(e)} "
                f"(execution_time: {execution_time:.1f}ms) [{context.validation_id}]"
            )
            raise ValidationError(
                f"Visual element validation error: {str(e)}",
                validation_context=context,
                cause=e,
                validation_type=ValidationType.VISUAL_ELEMENT
            )
    
    async def _handle_element_not_found(
        self, expected: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Handle case where element is not found but was expected."""
        
        confidence_score = ConfidenceScore(
            value=0.0,
            components={"element_present": 0.0, "expected_present": 1.0},
            method="visual_element_presence",
            reliability=1.0
        )
        
        status = ValidationStatus.FAILED
        message = "Visual element validation FAILED: Element not found but was expected to be present"
        
        return self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=expected,
            actual=None,
            message=message
        )
    
    async def _handle_element_correctly_absent(
        self, expected: Dict[str, Any], context: ValidationContext, threshold: float
    ) -> ValidationResult:
        """Handle case where element is correctly absent."""
        
        confidence_score = ConfidenceScore(
            value=1.0,
            components={"element_absent": 1.0, "expected_absent": 1.0},
            method="visual_element_absence",
            reliability=1.0
        )
        
        status = ValidationStatus.PASSED
        message = "Visual element validation PASSED: Element correctly absent as expected"
        
        return self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=expected,
            actual=None,
            message=message
        )
    
    async def _validate_element_properties(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        context: ValidationContext,
        threshold: float,
        browser_session,
        element_selector: Optional[str]
    ) -> ValidationResult:
        """Validate individual element properties."""
        
        confidence_components = {}
        property_results = {}
        total_properties = len(expected)
        passed_properties = 0
        
        # Validate each expected property
        for prop_name, expected_value in expected.items():
            prop_result = await self._validate_single_property(
                prop_name, expected_value, actual, context
            )
            
            property_results[prop_name] = prop_result
            confidence_components[f"property_{prop_name}"] = prop_result["confidence"]
            
            if prop_result["passed"]:
                passed_properties += 1
        
        # Calculate overall confidence
        if total_properties == 0:
            overall_confidence = 1.0
        else:
            # Weighted average based on property importance
            property_weights = self._get_property_weights()
            weighted_sum = 0.0
            total_weight = 0.0
            
            for prop_name, prop_result in property_results.items():
                weight = property_weights.get(prop_name, 1.0)
                weighted_sum += prop_result["confidence"] * weight
                total_weight += weight
            
            overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Create confidence score
        confidence_score = ConfidenceScore(
            value=overall_confidence,
            components=confidence_components,
            method="visual_element_properties",
            reliability=self._calculate_element_reliability(actual, expected)
        )
        
        # Determine status
        status = ValidationStatus.PASSED if confidence_score.is_above_threshold(threshold) else ValidationStatus.FAILED
        
        # Create message
        message = f"Visual element validation: {status.value} "
        message += f"({passed_properties}/{total_properties} properties passed, "
        message += f"confidence: {confidence_score.value:.3f})"
        
        result = self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=expected,
            actual=actual,
            message=message
        )
        
        # Add detailed property results
        result.add_detail("property_results", property_results)
        result.add_detail("properties_passed", passed_properties)
        result.add_detail("properties_total", total_properties)
        result.add_detail("element_info", self._extract_element_info(actual))
        
        return result
    
    async def _validate_single_property(
        self, prop_name: str, expected_value: Any, actual: Dict[str, Any], context: ValidationContext
    ) -> Dict[str, Any]:
        """Validate a single element property."""
        
        prop_enum = None
        try:
            prop_enum = ElementProperty(prop_name)
        except ValueError:
            # Custom property, treat as generic
            pass
        
        actual_value = actual.get(prop_name)
        
        if prop_enum == ElementProperty.PRESENT:
            return self._validate_presence(expected_value, actual_value is not None)
        elif prop_enum == ElementProperty.VISIBLE:
            return self._validate_visibility(expected_value, actual_value)
        elif prop_enum == ElementProperty.POSITION:
            return self._validate_position(expected_value, actual_value)
        elif prop_enum == ElementProperty.SIZE:
            return self._validate_size(expected_value, actual_value)
        elif prop_enum == ElementProperty.COLOR:
            return self._validate_color(expected_value, actual_value)
        elif prop_enum == ElementProperty.TEXT_CONTENT:
            return self._validate_text_content(expected_value, actual_value)
        elif prop_name in self.common_css_properties:
            return self._validate_css_property(prop_name, expected_value, actual_value)
        else:
            # Generic property validation
            return self._validate_generic_property(prop_name, expected_value, actual_value)
    
    def _validate_presence(self, expected: bool, actual: bool) -> Dict[str, Any]:
        """Validate element presence."""
        match = expected == actual
        return {
            "passed": match,
            "confidence": 1.0 if match else 0.0,
            "expected": expected,
            "actual": actual,
            "validation_type": "presence"
        }
    
    def _validate_visibility(self, expected: Union[bool, str], actual: Any) -> Dict[str, Any]:
        """Validate element visibility."""
        if isinstance(expected, bool):
            # Simple visible/not visible
            actual_visible = actual not in [False, "hidden", "none", 0]
            match = expected == actual_visible
            confidence = 1.0 if match else 0.0
        else:
            # Specific visibility level
            confidence = self._calculate_visibility_confidence(expected, actual)
            match = confidence >= 0.8
        
        return {
            "passed": match,
            "confidence": confidence,
            "expected": expected,
            "actual": actual,
            "validation_type": "visibility"
        }
    
    def _validate_position(self, expected: Dict[str, int], actual: Dict[str, int]) -> Dict[str, Any]:
        """Validate element position."""
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return {
                "passed": False,
                "confidence": 0.0,
                "expected": expected,
                "actual": actual,
                "validation_type": "position",
                "error": "Position must be a dictionary with x/y or left/top coordinates"
            }
        
        # Extract coordinates
        exp_x = expected.get('x', expected.get('left', 0))
        exp_y = expected.get('y', expected.get('top', 0))
        act_x = actual.get('x', actual.get('left', 0))
        act_y = actual.get('y', actual.get('top', 0))
        
        # Calculate distance
        distance = ((exp_x - act_x) ** 2 + (exp_y - act_y) ** 2) ** 0.5
        
        # Confidence decreases with distance
        if distance <= self.position_tolerance:
            confidence = 1.0
        elif distance <= self.position_tolerance * 3:
            confidence = 0.7 - (distance - self.position_tolerance) / (self.position_tolerance * 2) * 0.3
        else:
            confidence = 0.0
        
        return {
            "passed": distance <= self.position_tolerance,
            "confidence": confidence,
            "expected": expected,
            "actual": actual,
            "validation_type": "position",
            "distance": distance,
            "tolerance": self.position_tolerance
        }
    
    def _validate_size(self, expected: Dict[str, int], actual: Dict[str, int]) -> Dict[str, Any]:
        """Validate element size."""
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return {
                "passed": False,
                "confidence": 0.0,
                "expected": expected,
                "actual": actual,
                "validation_type": "size",
                "error": "Size must be a dictionary with width/height"
            }
        
        exp_width = expected.get('width', 0)
        exp_height = expected.get('height', 0)
        act_width = actual.get('width', 0)
        act_height = actual.get('height', 0)
        
        # Calculate size differences
        width_diff = abs(exp_width - act_width)
        height_diff = abs(exp_height - act_height)
        
        # Confidence based on size accuracy
        width_conf = 1.0 if width_diff <= self.size_tolerance else max(0.0, 1.0 - width_diff / (exp_width + 1))
        height_conf = 1.0 if height_diff <= self.size_tolerance else max(0.0, 1.0 - height_diff / (exp_height + 1))
        
        confidence = (width_conf + height_conf) / 2
        passed = width_diff <= self.size_tolerance and height_diff <= self.size_tolerance
        
        return {
            "passed": passed,
            "confidence": confidence,
            "expected": expected,
            "actual": actual,
            "validation_type": "size",
            "width_difference": width_diff,
            "height_difference": height_diff,
            "tolerance": self.size_tolerance
        }
    
    def _validate_color(self, expected: str, actual: str) -> Dict[str, Any]:
        """Validate color values."""
        # Normalize color values for comparison
        exp_color = self._normalize_color(expected)
        act_color = self._normalize_color(actual)
        
        match = exp_color == act_color
        confidence = 1.0 if match else self._calculate_color_similarity(exp_color, act_color)
        
        return {
            "passed": confidence >= 0.9,
            "confidence": confidence,
            "expected": expected,
            "actual": actual,
            "validation_type": "color",
            "normalized_expected": exp_color,
            "normalized_actual": act_color
        }
    
    def _validate_text_content(self, expected: str, actual: str) -> Dict[str, Any]:
        """Validate text content."""
        if not isinstance(expected, str) or not isinstance(actual, str):
            return {
                "passed": False,
                "confidence": 0.0,
                "expected": expected,
                "actual": actual,
                "validation_type": "text_content",
                "error": "Text content must be strings"
            }
        
        # Use simple similarity for now (could be enhanced with semantic validation)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, expected.lower().strip(), actual.lower().strip()).ratio()
        
        return {
            "passed": similarity >= 0.8,
            "confidence": similarity,
            "expected": expected,
            "actual": actual,
            "validation_type": "text_content",
            "similarity": similarity
        }
    
    def _validate_css_property(self, prop_name: str, expected: Any, actual: Any) -> Dict[str, Any]:
        """Validate CSS property values."""
        if expected == actual:
            confidence = 1.0
        elif isinstance(expected, str) and isinstance(actual, str):
            # Normalize CSS values for comparison
            exp_normalized = expected.lower().strip().replace(' ', '')
            act_normalized = actual.lower().strip().replace(' ', '')
            confidence = 1.0 if exp_normalized == act_normalized else 0.5
        else:
            confidence = 0.0
        
        return {
            "passed": confidence >= 0.8,
            "confidence": confidence,
            "expected": expected,
            "actual": actual,
            "validation_type": f"css_{prop_name}"
        }
    
    def _validate_generic_property(self, prop_name: str, expected: Any, actual: Any) -> Dict[str, Any]:
        """Validate generic property values."""
        match = expected == actual
        
        # Try some intelligent comparison for common types
        if not match and isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            # Numerical tolerance
            tolerance = abs(expected * 0.05) if expected != 0 else 1
            match = abs(expected - actual) <= tolerance
            confidence = 1.0 if match else max(0.0, 1.0 - abs(expected - actual) / (abs(expected) + 1))
        else:
            confidence = 1.0 if match else 0.0
        
        return {
            "passed": match,
            "confidence": confidence,
            "expected": expected,
            "actual": actual,
            "validation_type": f"generic_{prop_name}"
        }
    
    def _calculate_visibility_confidence(self, expected: str, actual: Any) -> float:
        """Calculate confidence for visibility validation."""
        visibility_mapping = {
            "hidden": 0.0,
            "partially_visible": 0.5,
            "fully_visible": 1.0,
            "viewport_visible": 0.8
        }
        
        expected_score = visibility_mapping.get(expected, 0.5)
        
        if isinstance(actual, bool):
            actual_score = 1.0 if actual else 0.0
        elif isinstance(actual, str):
            actual_score = visibility_mapping.get(actual, 0.5)
        elif isinstance(actual, (int, float)):
            actual_score = min(1.0, max(0.0, actual))
        else:
            actual_score = 0.5
        
        # Calculate confidence based on similarity
        return 1.0 - abs(expected_score - actual_score)
    
    def _normalize_color(self, color: str) -> str:
        """Normalize color values for comparison."""
        if not color:
            return ""
        
        color = color.lower().strip()
        
        # Convert named colors to hex (simplified)
        named_colors = {
            "red": "#ff0000", "green": "#008000", "blue": "#0000ff",
            "white": "#ffffff", "black": "#000000", "yellow": "#ffff00",
            "cyan": "#00ffff", "magenta": "#ff00ff", "gray": "#808080"
        }
        
        if color in named_colors:
            return named_colors[color]
        
        # Normalize rgb() format
        if color.startswith('rgb('):
            # Extract numbers and convert to hex
            import re
            numbers = re.findall(r'\d+', color)
            if len(numbers) >= 3:
                r, g, b = int(numbers[0]), int(numbers[1]), int(numbers[2])
                return f"#{r:02x}{g:02x}{b:02x}"
        
        return color
    
    def _calculate_color_similarity(self, color1: str, color2: str) -> float:
        """Calculate similarity between color values."""
        # Very basic color similarity (could be enhanced with proper color space calculations)
        if color1 == color2:
            return 1.0
        
        # If both are hex colors, calculate RGB distance
        if color1.startswith('#') and color2.startswith('#') and len(color1) == 7 and len(color2) == 7:
            try:
                r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
                r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
                
                # Euclidean distance in RGB space
                distance = ((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)**0.5
                max_distance = (255**2 * 3)**0.5  # Maximum possible distance
                
                return 1.0 - (distance / max_distance)
            except ValueError:
                pass
        
        return 0.0  # Default to no similarity for unparseable colors
    
    def _get_property_weights(self) -> Dict[str, float]:
        """Get importance weights for different properties."""
        return {
            ElementProperty.PRESENT.value: 2.0,
            ElementProperty.VISIBLE.value: 1.8,
            ElementProperty.TEXT_CONTENT.value: 1.5,
            ElementProperty.POSITION.value: 1.2,
            ElementProperty.SIZE.value: 1.2,
            ElementProperty.ENABLED.value: 1.3,
            ElementProperty.CLICKABLE.value: 1.3,
            # CSS properties get lower weight
            **{prop: 1.0 for prop in self.common_css_properties}
        }
    
    def _calculate_element_reliability(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """Calculate reliability score for element validation."""
        reliability = 1.0
        
        # Reduce reliability if actual data seems incomplete
        if len(actual) < len(expected) / 2:
            reliability *= 0.8
        
        # Increase reliability if we have position and size data
        if 'position' in actual and 'size' in actual:
            reliability = min(1.0, reliability * 1.1)
        
        # Reduce reliability if element seems to be in transition
        if 'opacity' in actual and isinstance(actual['opacity'], (int, float)):
            if 0.1 < actual['opacity'] < 0.9:
                reliability *= 0.9  # Element might be animating
        
        return reliability
    
    def _extract_element_info(self, actual: Dict[str, Any]) -> Dict[str, Any]:
        """Extract useful element information for debugging."""
        info = {}
        
        # Basic properties
        for prop in ['tag_name', 'id', 'class', 'name']:
            if prop in actual:
                info[prop] = actual[prop]
        
        # Computed styles summary
        if 'computed_styles' in actual:
            styles = actual['computed_styles']
            info['computed_styles_summary'] = {
                'display': styles.get('display'),
                'visibility': styles.get('visibility'),
                'opacity': styles.get('opacity'),
                'position': styles.get('position')
            }
        
        # Geometry summary
        geometry_props = ['position', 'size', 'bounds']
        for prop in geometry_props:
            if prop in actual:
                info[prop] = actual[prop]
        
        return info