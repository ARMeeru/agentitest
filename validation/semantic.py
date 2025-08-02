# validation/semantic.py
"""
Semantic validation with confidence scoring.

This module provides semantic validation capabilities that go beyond simple string matching
to understand the meaning and intent of validation criteria. Uses multiple techniques
including similarity scoring, semantic matching, and contextual analysis.
"""

import re
import time
import logging
from typing import Dict, Any, Optional, List, Union
from difflib import SequenceMatcher

from .core import (
    ValidationStrategy,
    ValidationResult, 
    ValidationContext,
    ValidationStatus,
    ValidationType,
    ConfidenceScore,
    ValidationError
)


class SemanticValidator(ValidationStrategy):
    """
    Advanced semantic validation with confidence scoring.
    
    This validator uses multiple techniques to determine semantic similarity:
    1. Exact string matching
    2. Case-insensitive matching  
    3. Sequence similarity scoring
    4. Keyword extraction and matching
    5. Semantic pattern recognition
    6. Contextual relevance scoring
    """
    
    def __init__(
        self,
        default_confidence_threshold: float = 0.8,
        enable_fuzzy_matching: bool = True,
        enable_keyword_extraction: bool = True,
        enable_pattern_matching: bool = True
    ):
        """
        Initialize semantic validator.
        
        Args:
            default_confidence_threshold: Default minimum confidence for success
            enable_fuzzy_matching: Enable fuzzy string matching
            enable_keyword_extraction: Enable keyword-based validation
            enable_pattern_matching: Enable pattern-based validation
        """
        super().__init__("SemanticValidator", default_confidence_threshold)
        self.validation_type = ValidationType.SEMANTIC
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.enable_keyword_extraction = enable_keyword_extraction
        self.enable_pattern_matching = enable_pattern_matching
        
        # Common patterns for web testing
        self.success_patterns = [
            r"success(?:ful)?(?:ly)?",
            r"complete(?:d)?",
            r"load(?:ed|ing)?",
            r"found|present|displayed|visible",
            r"pass(?:ed)?",
            r"ok(?:ay)?",
            r"valid|correct"
        ]
        
        self.failure_patterns = [  
            r"fail(?:ed|ure)?",
            r"error|exception",
            r"not\s+found|missing|absent",
            r"invalid|incorrect",
            r"timeout|timed\s+out",
            r"unavailable|unreachable"
        ]
        
        # Keywords that indicate specific validation contexts
        self.context_keywords = {
            "navigation": ["navigate", "redirect", "url", "page", "link", "route"],
            "content": ["text", "content", "display", "show", "visible", "present"],
            "interaction": ["click", "type", "submit", "select", "choose", "input"],
            "state": ["load", "ready", "complete", "finish", "done", "available"],
            "search": ["search", "find", "results", "query", "filter", "match"],
            "form": ["form", "field", "input", "submit", "validate", "enter"]
        }
    
    async def validate(
        self,
        expected: Union[str, List[str]],
        actual: str,
        context: Optional[ValidationContext] = None,
        confidence_threshold: Optional[float] = None,
        ignore_case: bool = True,
        enable_partial_matching: bool = True,
        semantic_weight: float = 0.6,
        similarity_weight: float = 0.4,
        **kwargs
    ) -> ValidationResult:
        """
        Perform semantic validation with confidence scoring.
        
        Args:
            expected: Expected value(s) - string or list of acceptable strings
            actual: Actual value from the system under test
            context: Validation context with metadata
            confidence_threshold: Minimum confidence threshold for success  
            ignore_case: Whether to ignore case differences
            enable_partial_matching: Allow partial string matches
            semantic_weight: Weight for semantic analysis (0.0-1.0)
            similarity_weight: Weight for similarity scoring (0.0-1.0)
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult with detailed confidence scoring
        """
        start_time = time.time()
        
        if context is None:
            context = self.create_context()
        
        context.set_configuration("ignore_case", ignore_case)
        context.set_configuration("enable_partial_matching", enable_partial_matching)
        context.set_configuration("semantic_weight", semantic_weight)
        context.set_configuration("similarity_weight", similarity_weight)
        
        threshold = self.get_confidence_threshold(confidence_threshold)
        
        try:
            # Handle multiple expected values
            expected_list = expected if isinstance(expected, list) else [expected]
            
            # Calculate confidence scores for each expected value
            best_score = 0.0
            best_match = None
            all_scores = {}
            
            for exp_value in expected_list:
                score_result = await self._calculate_semantic_confidence(
                    exp_value, actual, context, ignore_case, enable_partial_matching,
                    semantic_weight, similarity_weight
                )
                all_scores[exp_value] = score_result
                
                if score_result["total_confidence"] > best_score:
                    best_score = score_result["total_confidence"]
                    best_match = exp_value
            
            # Create confidence score object
            confidence_score = ConfidenceScore(
                value=best_score,
                components=all_scores[best_match] if best_match else {},
                method="semantic_analysis",
                reliability=self._calculate_reliability(actual, context)
            )
            
            # Determine validation status
            status = ValidationStatus.PASSED if confidence_score.is_above_threshold(threshold) else ValidationStatus.FAILED
            
            # Create detailed message
            message = self._create_validation_message(
                expected_list, actual, best_match, confidence_score, status, threshold
            )
            
            # Create result
            result = self.create_result(
                status=status,
                confidence_score=confidence_score,
                context=context,
                expected=expected,
                actual=actual,
                message=message
            )
            
            # Add detailed information
            result.add_detail("all_confidence_scores", all_scores)
            result.add_detail("best_match", best_match)
            result.add_detail("threshold_used", threshold) 
            result.add_detail("validation_method", "semantic_analysis")
            result.add_detail("patterns_detected", self._detect_patterns(actual))
            result.add_detail("context_analysis", self._analyze_context(actual, context))
            
            # Record execution time
            result.execution_time_ms = (time.time() - start_time) * 1000
            
            logging.info(
                f"Semantic validation completed: {status.value} "
                f"(confidence: {confidence_score.value:.3f}, threshold: {threshold:.3f}) "
                f"[{context.validation_id}]"
            )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logging.error(
                f"Semantic validation failed: {str(e)} "
                f"(execution_time: {execution_time:.1f}ms) [{context.validation_id}]"
            )
            raise ValidationError(
                f"Semantic validation error: {str(e)}",
                validation_context=context,
                cause=e,
                validation_type=ValidationType.SEMANTIC
            )
    
    async def _calculate_semantic_confidence(
        self,
        expected: str,
        actual: str,
        context: ValidationContext,
        ignore_case: bool,
        enable_partial_matching: bool,
        semantic_weight: float,
        similarity_weight: float
    ) -> Dict[str, float]:
        """Calculate semantic confidence score with component breakdown."""
        
        # Prepare strings for comparison
        exp_str = expected.lower() if ignore_case else expected
        act_str = actual.lower() if ignore_case else actual
        
        confidence_components = {}
        
        # 1. Exact match scoring
        exact_match = exp_str == act_str
        confidence_components["exact_match"] = 1.0 if exact_match else 0.0
        
        # 2. Substring/partial match scoring  
        if enable_partial_matching:
            if exp_str in act_str:
                confidence_components["substring_match"] = 1.0
            elif act_str in exp_str:
                confidence_components["substring_match"] = 0.8
            else:
                confidence_components["substring_match"] = 0.0
        else:
            confidence_components["substring_match"] = 0.0
        
        # 3. Sequence similarity scoring
        if self.enable_fuzzy_matching:
            similarity_ratio = SequenceMatcher(None, exp_str, act_str).ratio()
            confidence_components["sequence_similarity"] = similarity_ratio
        else:
            confidence_components["sequence_similarity"] = 0.0
        
        # 4. Keyword extraction and matching
        if self.enable_keyword_extraction:
            keyword_score = self._calculate_keyword_confidence(exp_str, act_str)
            confidence_components["keyword_matching"] = keyword_score
        else:
            confidence_components["keyword_matching"] = 0.0
        
        # 5. Pattern-based scoring
        if self.enable_pattern_matching:
            pattern_score = self._calculate_pattern_confidence(exp_str, act_str)
            confidence_components["pattern_matching"] = pattern_score
        else:
            confidence_components["pattern_matching"] = 0.0
        
        # 6. Semantic context scoring
        context_score = self._calculate_context_confidence(exp_str, act_str, context)
        confidence_components["context_relevance"] = context_score
        
        # Calculate weighted total confidence
        # If exact match, give high weight
        if confidence_components["exact_match"] == 1.0:
            total_confidence = 1.0
        else:
            # Weighted combination of different confidence measures
            semantic_score = max(
                confidence_components["keyword_matching"],
                confidence_components["pattern_matching"],
                confidence_components["context_relevance"]
            )
            
            similarity_score = max(
                confidence_components["substring_match"],
                confidence_components["sequence_similarity"]
            )
            
            total_confidence = (semantic_score * semantic_weight) + (similarity_score * similarity_weight)
        
        confidence_components["total_confidence"] = total_confidence
        
        return confidence_components
    
    def _calculate_keyword_confidence(self, expected: str, actual: str) -> float:
        """Calculate confidence based on keyword matching."""
        # Extract keywords (simple word extraction)
        exp_words = set(re.findall(r'\b\w+\b', expected.lower()))
        act_words = set(re.findall(r'\b\w+\b', actual.lower()))
        
        if not exp_words:
            return 0.0
        
        # Calculate overlap
        overlap = len(exp_words.intersection(act_words))
        return overlap / len(exp_words)
    
    def _calculate_pattern_confidence(self, expected: str, actual: str) -> float:
        """Calculate confidence based on pattern matching."""
        max_score = 0.0
        
        # Check for success patterns if expected indicates success
        success_indicators = ["success", "complete", "load", "pass", "ok", "valid"]
        if any(indicator in expected for indicator in success_indicators):
            for pattern in self.success_patterns:
                if re.search(pattern, actual, re.IGNORECASE):
                    max_score = max(max_score, 0.9)
        
        # Check for failure patterns if expected indicates failure  
        failure_indicators = ["fail", "error", "not", "invalid", "missing"]
        if any(indicator in expected for indicator in failure_indicators):
            for pattern in self.failure_patterns:
                if re.search(pattern, actual, re.IGNORECASE):
                    max_score = max(max_score, 0.9)
        
        return max_score
    
    def _calculate_context_confidence(
        self, expected: str, actual: str, context: ValidationContext
    ) -> float:
        """Calculate confidence based on contextual relevance."""
        
        # Extract context from metadata
        test_name = context.test_name or ""
        test_method = context.test_method or ""
        metadata = context.metadata
        
        context_text = f"{test_name} {test_method} {str(metadata)}".lower()
        
        max_relevance = 0.0
        
        # Check context keyword relevance
        for context_type, keywords in self.context_keywords.items():
            if any(keyword in context_text for keyword in keywords):
                # Check if actual content matches this context
                if any(keyword in actual.lower() for keyword in keywords):
                    max_relevance = max(max_relevance, 0.7)
                # Check if expected content matches this context
                elif any(keyword in expected.lower() for keyword in keywords):
                    max_relevance = max(max_relevance, 0.5)
        
        return max_relevance
    
    def _calculate_reliability(self, actual: str, context: ValidationContext) -> float:
        """Calculate reliability score for this validation."""
        reliability = 1.0
        
        # Reduce reliability for very short responses
        if len(actual.strip()) < 3:
            reliability *= 0.6
        
        # Reduce reliability for very generic responses
        generic_responses = ["ok", "yes", "no", "done", "success", "fail"]
        if actual.lower().strip() in generic_responses:
            reliability *= 0.7
        
        # Increase reliability if we have good context
        if context.page_url or context.test_name or context.metadata:
            reliability = min(1.0, reliability * 1.1)
        
        return reliability
    
    def _detect_patterns(self, text: str) -> List[str]:
        """Detect patterns in the actual text."""
        patterns_found = []
        
        # Check success patterns
        for pattern in self.success_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                patterns_found.append(f"success_pattern: {pattern}")
        
        # Check failure patterns
        for pattern in self.failure_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                patterns_found.append(f"failure_pattern: {pattern}")
        
        return patterns_found
    
    def _analyze_context(self, actual: str, context: ValidationContext) -> Dict[str, Any]:
        """Analyze contextual information."""
        analysis = {
            "detected_context_types": [],
            "text_length": len(actual),
            "word_count": len(actual.split()),
            "has_url_context": bool(context.page_url),
            "has_test_context": bool(context.test_name or context.test_method),
            "metadata_keys": list(context.metadata.keys()) if context.metadata else []
        }
        
        # Detect context types based on content
        for context_type, keywords in self.context_keywords.items():
            if any(keyword in actual.lower() for keyword in keywords):
                analysis["detected_context_types"].append(context_type)
        
        return analysis
    
    def _create_validation_message(
        self,
        expected_list: List[str],
        actual: str,
        best_match: Optional[str],
        confidence_score: ConfidenceScore,
        status: ValidationStatus,
        threshold: float
    ) -> str:
        """Create detailed validation message."""
        
        if status == ValidationStatus.PASSED:
            message = f"Semantic validation PASSED with confidence {confidence_score.value:.3f} (threshold: {threshold:.3f})"
            if best_match:
                message += f"\nBest match: '{best_match}' vs actual: '{actual[:100]}...'"
        else:
            message = f"Semantic validation FAILED with confidence {confidence_score.value:.3f} (threshold: {threshold:.3f})"
            message += f"\nExpected one of: {expected_list}"
            message += f"\nActual: '{actual[:200]}...'"
            if confidence_score.components:
                top_component = max(confidence_score.components.items(), key=lambda x: x[1])
                message += f"\nHighest component score: {top_component[0]} = {top_component[1]:.3f}"
        
        return message