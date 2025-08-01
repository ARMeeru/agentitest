# validation/text_content.py
"""
Text content validation with fuzzy matching and advanced text analysis.

This module provides comprehensive text validation capabilities including fuzzy matching,
regular expression validation, text structure analysis, and content quality checks.
"""

import re
import time
import logging
from typing import Dict, Any, Optional, List, Union, Pattern
from difflib import SequenceMatcher
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


class TextMatchingMode(Enum):
    """Text matching modes for different validation scenarios."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    REGEX = "regex"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    WORDS = "words"
    LINES = "lines"


class TextContentValidator(ValidationStrategy):
    """
    Advanced text content validation with fuzzy matching.
    
    This validator provides multiple text matching strategies with confidence scoring:
    1. Exact text matching
    2. Fuzzy string matching with similarity thresholds
    3. Regular expression matching
    4. Partial content matching (contains, starts with, ends with)
    5. Word-based matching with stemming
    6. Line-based matching for structured text
    7. Text structure and quality analysis
    """
    
    def __init__(
        self,
        default_confidence_threshold: float = 0.8,
        default_fuzzy_threshold: float = 0.7,
        enable_normalization: bool = True,
        enable_stemming: bool = False
    ):
        """
        Initialize text content validator.
        
        Args:
            default_confidence_threshold: Default minimum confidence for success
            default_fuzzy_threshold: Default threshold for fuzzy matching
            enable_normalization: Enable text normalization (whitespace, case)
            enable_stemming: Enable word stemming for better matching
        """
        super().__init__("TextContentValidator", default_confidence_threshold)
        self.validation_type = ValidationType.TEXT_CONTENT
        self.default_fuzzy_threshold = default_fuzzy_threshold
        self.enable_normalization = enable_normalization
        self.enable_stemming = enable_stemming
        
        # Common text patterns for web content
        self.common_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?',
            "phone": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            "date": r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            "time": r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?\b',
            "number": r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            "currency": r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:USD|EUR|GBP)\b'
        }
        
        # Stop words for better text analysis (simplified set)
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'but', 'have', 'this', 'you', 'your'
        }
    
    async def validate(
        self,
        expected: Union[str, List[str], Pattern[str]],
        actual: str,
        context: Optional[ValidationContext] = None,
        confidence_threshold: Optional[float] = None,
        matching_mode: TextMatchingMode = TextMatchingMode.FUZZY,
        fuzzy_threshold: Optional[float] = None,
        ignore_case: bool = True,
        ignore_whitespace: bool = True,
        normalize_text: bool = True,
        **kwargs
    ) -> ValidationResult:
        """
        Perform text content validation with fuzzy matching.
        
        Args:
            expected: Expected text content (string, list of strings, or regex pattern)
            actual: Actual text content to validate
            context: Validation context with metadata
            confidence_threshold: Minimum confidence threshold for success
            matching_mode: Text matching strategy to use
            fuzzy_threshold: Threshold for fuzzy matching (0.0-1.0)
            ignore_case: Whether to ignore case differences
            ignore_whitespace: Whether to normalize whitespace
            normalize_text: Whether to apply full text normalization
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult with detailed confidence scoring
        """
        start_time = time.time()
        
        if context is None:
            context = self.create_context()
        
        # Store configuration
        context.set_configuration("matching_mode", matching_mode.value)
        context.set_configuration("fuzzy_threshold", fuzzy_threshold or self.default_fuzzy_threshold)
        context.set_configuration("ignore_case", ignore_case)
        context.set_configuration("ignore_whitespace", ignore_whitespace)
        context.set_configuration("normalize_text", normalize_text)
        
        threshold = self.get_confidence_threshold(confidence_threshold)
        fuzzy_thresh = fuzzy_threshold or self.default_fuzzy_threshold
        
        try:
            # Normalize text if enabled
            normalized_actual = self._normalize_text(actual, ignore_case, ignore_whitespace, normalize_text)
            
            # Handle different types of expected values
            if isinstance(expected, Pattern):
                # Regex pattern matching
                result = await self._validate_with_regex(
                    expected, normalized_actual, context, threshold
                )
            elif isinstance(expected, list):
                # Multiple expected values
                result = await self._validate_with_multiple_expected(
                    expected, normalized_actual, context, threshold, matching_mode, 
                    fuzzy_thresh, ignore_case, ignore_whitespace, normalize_text
                )
            else:
                # Single expected string
                normalized_expected = self._normalize_text(expected, ignore_case, ignore_whitespace, normalize_text)
                result = await self._validate_with_single_expected(
                    normalized_expected, normalized_actual, context, threshold, 
                    matching_mode, fuzzy_thresh
                )
            
            # Add analysis details
            result.add_detail("text_analysis", self._analyze_text_structure(actual))
            result.add_detail("pattern_matches", self._find_pattern_matches(actual))
            result.add_detail("text_quality", self._assess_text_quality(actual))
            result.add_detail("normalization_applied", {
                "ignore_case": ignore_case,
                "ignore_whitespace": ignore_whitespace,
                "normalize_text": normalize_text,
                "original_length": len(actual),
                "normalized_length": len(normalized_actual)
            })
            
            # Record execution time
            result.execution_time_ms = (time.time() - start_time) * 1000
            
            logging.info(
                f"Text content validation completed: {result.status.value} "
                f"(confidence: {result.confidence_score.value:.3f}, mode: {matching_mode.value}) "
                f"[{context.validation_id}]"
            )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logging.error(
                f"Text content validation failed: {str(e)} "
                f"(execution_time: {execution_time:.1f}ms) [{context.validation_id}]"
            )
            raise ValidationError(
                f"Text content validation error: {str(e)}",
                validation_context=context,
                cause=e,
                validation_type=ValidationType.TEXT_CONTENT
            )
    
    async def _validate_with_single_expected(
        self,
        expected: str,
        actual: str,
        context: ValidationContext,
        threshold: float,
        matching_mode: TextMatchingMode,
        fuzzy_threshold: float
    ) -> ValidationResult:
        """Validate against a single expected string."""
        
        confidence_components = {}
        
        # Apply matching strategy
        if matching_mode == TextMatchingMode.EXACT:
            match = expected == actual
            confidence_components["exact_match"] = 1.0 if match else 0.0
            total_confidence = confidence_components["exact_match"]
            
        elif matching_mode == TextMatchingMode.FUZZY:
            similarity = SequenceMatcher(None, expected, actual).ratio()
            confidence_components["fuzzy_similarity"] = similarity
            confidence_components["meets_fuzzy_threshold"] = 1.0 if similarity >= fuzzy_threshold else 0.0
            total_confidence = similarity
            
        elif matching_mode == TextMatchingMode.CONTAINS:
            contains = expected in actual
            confidence_components["contains_match"] = 1.0 if contains else 0.0
            # Also calculate partial similarity for confidence
            if not contains:
                words_expected = set(expected.lower().split())
                words_actual = set(actual.lower().split())
                word_overlap = len(words_expected.intersection(words_actual))
                confidence_components["word_overlap"] = word_overlap / len(words_expected) if words_expected else 0.0
                total_confidence = confidence_components["word_overlap"]
            else:
                total_confidence = 1.0
                
        elif matching_mode == TextMatchingMode.STARTS_WITH:
            starts = actual.startswith(expected)
            confidence_components["starts_with_match"] = 1.0 if starts else 0.0
            # Calculate prefix similarity
            prefix_len = min(len(expected), len(actual))
            prefix_similarity = SequenceMatcher(None, expected[:prefix_len], actual[:prefix_len]).ratio()
            confidence_components["prefix_similarity"] = prefix_similarity
            total_confidence = confidence_components["starts_with_match"] if starts else prefix_similarity
            
        elif matching_mode == TextMatchingMode.ENDS_WITH:
            ends = actual.endswith(expected)
            confidence_components["ends_with_match"] = 1.0 if ends else 0.0
            # Calculate suffix similarity
            suffix_len = min(len(expected), len(actual))
            suffix_similarity = SequenceMatcher(None, expected[-suffix_len:], actual[-suffix_len:]).ratio()
            confidence_components["suffix_similarity"] = suffix_similarity
            total_confidence = confidence_components["ends_with_match"] if ends else suffix_similarity
            
        elif matching_mode == TextMatchingMode.WORDS:
            words_conf = self._calculate_word_confidence(expected, actual)
            confidence_components.update(words_conf)
            total_confidence = words_conf["word_match_ratio"]
            
        elif matching_mode == TextMatchingMode.LINES:
            lines_conf = self._calculate_line_confidence(expected, actual)
            confidence_components.update(lines_conf)
            total_confidence = lines_conf["line_match_ratio"]
            
        else:
            # Default to fuzzy matching
            similarity = SequenceMatcher(None, expected, actual).ratio()
            confidence_components["default_fuzzy_similarity"] = similarity
            total_confidence = similarity
        
        # Create confidence score
        confidence_score = ConfidenceScore(
            value=total_confidence,
            components=confidence_components,
            method=f"text_content_{matching_mode.value}",
            reliability=self._calculate_text_reliability(actual, expected)
        )
        
        # Determine status
        status = ValidationStatus.PASSED if confidence_score.is_above_threshold(threshold) else ValidationStatus.FAILED
        
        # Create message
        message = self._create_text_validation_message(
            expected, actual, confidence_score, status, threshold, matching_mode
        )
        
        return self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=expected,
            actual=actual,
            message=message
        )
    
    async def _validate_with_multiple_expected(
        self,
        expected_list: List[str],
        actual: str,
        context: ValidationContext,
        threshold: float,
        matching_mode: TextMatchingMode,
        fuzzy_threshold: float,
        ignore_case: bool,
        ignore_whitespace: bool,
        normalize_text: bool
    ) -> ValidationResult:
        """Validate against multiple expected strings."""
        
        best_score = 0.0
        best_match = None
        all_scores = {}
        
        for expected in expected_list:
            normalized_expected = self._normalize_text(expected, ignore_case, ignore_whitespace, normalize_text)
            
            # Validate against this expected value
            single_result = await self._validate_with_single_expected(
                normalized_expected, actual, context, threshold, matching_mode, fuzzy_threshold
            )
            
            score = single_result.confidence_score.value
            all_scores[expected] = {
                "confidence": score,
                "components": single_result.confidence_score.components,
                "status": single_result.status.value
            }
            
            if score > best_score:
                best_score = score
                best_match = expected
        
        # Create combined confidence score
        confidence_score = ConfidenceScore(
            value=best_score,
            components=all_scores[best_match]["components"] if best_match else {},
            method=f"text_content_multiple_{matching_mode.value}",
            reliability=self._calculate_text_reliability(actual, best_match or "")
        )
        
        # Determine status
        status = ValidationStatus.PASSED if confidence_score.is_above_threshold(threshold) else ValidationStatus.FAILED
        
        # Create message
        message = f"Text validation against {len(expected_list)} options: {status.value} "
        message += f"(best match: '{best_match}' with confidence {best_score:.3f})"
        
        result = self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=expected_list,
            actual=actual,
            message=message
        )
        
        result.add_detail("all_matches", all_scores)
        result.add_detail("best_match", best_match)
        
        return result
    
    async def _validate_with_regex(
        self,
        pattern: Pattern[str],
        actual: str,
        context: ValidationContext,
        threshold: float
    ) -> ValidationResult:
        """Validate using regular expression pattern."""
        
        match = pattern.search(actual)
        confidence_components = {
            "regex_match": 1.0 if match else 0.0,
            "pattern": pattern.pattern
        }
        
        if match:
            confidence_components["match_start"] = match.start()
            confidence_components["match_end"] = match.end()
            confidence_components["matched_text"] = match.group(0)
            confidence_components["groups"] = match.groups() if match.groups() else []
            total_confidence = 1.0
        else:
            # Try to find partial matches for confidence scoring
            partial_matches = len(re.findall(r'\w+', pattern.pattern)) 
            if partial_matches > 0:
                actual_words = set(re.findall(r'\w+', actual.lower()))
                pattern_words = set(re.findall(r'\w+', pattern.pattern.lower()))
                overlap = len(actual_words.intersection(pattern_words))
                confidence_components["partial_word_match"] = overlap / partial_matches
                total_confidence = confidence_components["partial_word_match"]
            else:
                total_confidence = 0.0
        
        confidence_score = ConfidenceScore(
            value=total_confidence,
            components=confidence_components,
            method="text_content_regex",
            reliability=1.0  # Regex patterns are deterministic
        )
        
        status = ValidationStatus.PASSED if confidence_score.is_above_threshold(threshold) else ValidationStatus.FAILED
        
        message = f"Regex validation: {status.value} (pattern: '{pattern.pattern}')"
        if match:
            message += f" - matched: '{match.group(0)}'"
        
        return self.create_result(
            status=status,
            confidence_score=confidence_score,
            context=context,
            expected=pattern.pattern,
            actual=actual,
            message=message
        )
    
    def _normalize_text(
        self, text: str, ignore_case: bool, ignore_whitespace: bool, normalize_text: bool
    ) -> str:
        """Normalize text based on configuration."""
        if not self.enable_normalization and not normalize_text:
            return text
        
        normalized = text
        
        if ignore_case:
            normalized = normalized.lower()
        
        if ignore_whitespace:
            # Normalize whitespace: collapse multiple spaces, strip, normalize line endings
            normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        if normalize_text:
            # Additional normalization: remove punctuation, normalize unicode
            normalized = re.sub(r'[^\w\s]', '', normalized)
            normalized = normalized.encode('ascii', 'ignore').decode('ascii')
        
        return normalized
    
    def _calculate_word_confidence(self, expected: str, actual: str) -> Dict[str, float]:
        """Calculate confidence based on word-level matching."""
        exp_words = set(word.lower() for word in expected.split() if word.lower() not in self.stop_words)
        act_words = set(word.lower() for word in actual.split() if word.lower() not in self.stop_words)
        
        if not exp_words:
            return {"word_match_ratio": 1.0, "word_overlap_count": 0, "expected_word_count": 0}
        
        overlap = len(exp_words.intersection(act_words))
        ratio = overlap / len(exp_words)
        
        return {
            "word_match_ratio": ratio,
            "word_overlap_count": overlap,
            "expected_word_count": len(exp_words),
            "actual_word_count": len(act_words),
            "unique_words_overlap": overlap,
            "jaccard_similarity": overlap / len(exp_words.union(act_words)) if exp_words.union(act_words) else 0.0
        }
    
    def _calculate_line_confidence(self, expected: str, actual: str) -> Dict[str, float]:
        """Calculate confidence based on line-level matching."""
        exp_lines = [line.strip() for line in expected.split('\n') if line.strip()]
        act_lines = [line.strip() for line in actual.split('\n') if line.strip()]
        
        if not exp_lines:
            return {"line_match_ratio": 1.0, "line_overlap_count": 0, "expected_line_count": 0}
        
        matches = 0
        for exp_line in exp_lines:
            if any(SequenceMatcher(None, exp_line, act_line).ratio() > 0.8 for act_line in act_lines):
                matches += 1
        
        ratio = matches / len(exp_lines)
        
        return {
            "line_match_ratio": ratio,
            "line_overlap_count": matches,
            "expected_line_count": len(exp_lines),
            "actual_line_count": len(act_lines)
        }
    
    def _calculate_text_reliability(self, actual: str, expected: str) -> float:
        """Calculate reliability score for text validation."""
        reliability = 1.0
        
        # Reduce reliability for very short text
        if len(actual.strip()) < 5:
            reliability *= 0.7
        
        # Reduce reliability for very different lengths
        if expected and len(actual) > 0:
            length_ratio = min(len(expected), len(actual)) / max(len(expected), len(actual))
            if length_ratio < 0.5:
                reliability *= 0.8
        
        # Increase reliability for structured text
        if re.search(r'\b\w+\b.*\b\w+\b', actual):  # Multiple words
            reliability = min(1.0, reliability * 1.1)
        
        return reliability
    
    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure and characteristics."""
        return {
            "character_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.split('\n')),
            "sentence_count": len(re.findall(r'[.!?]+', text)),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "has_punctuation": bool(re.search(r'[.!?,:;]', text)),
            "has_numbers": bool(re.search(r'\d', text)),
            "has_uppercase": bool(re.search(r'[A-Z]', text)),
            "has_lowercase": bool(re.search(r'[a-z]', text)),
            "whitespace_ratio": len(re.findall(r'\s', text)) / len(text) if text else 0
        }
    
    def _find_pattern_matches(self, text: str) -> Dict[str, List[str]]:
        """Find common patterns in text."""
        matches = {}
        for pattern_name, pattern in self.common_patterns.items():
            found = re.findall(pattern, text, re.IGNORECASE)
            if found:
                matches[pattern_name] = found
        return matches
    
    def _assess_text_quality(self, text: str) -> Dict[str, Any]:
        """Assess text quality and characteristics."""
        words = text.split()
        
        return {
            "is_empty": len(text.strip()) == 0,
            "is_too_short": len(text.strip()) < 3,
            "is_too_long": len(text) > 10000,
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "has_repeated_words": len(words) != len(set(word.lower() for word in words)),
            "readability_estimate": min(100, len(words) / max(1, len(text.split('.')))),  # Simplified
            "contains_special_chars": bool(re.search(r'[^\w\s]', text)),
            "language_hints": self._detect_language_hints(text)
        }
    
    def _detect_language_hints(self, text: str) -> List[str]:
        """Detect potential language characteristics."""
        hints = []
        
        # Simple heuristics for language detection
        if re.search(r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b', text.lower()):
            hints.append("english_indicators")
        
        if re.search(r'[áéíóúñü]', text.lower()):
            hints.append("spanish_indicators")
        
        if re.search(r'[àâäéèêëîïôöùûüÿç]', text.lower()):
            hints.append("french_indicators")
        
        if len(re.findall(r'[A-Z]', text)) / len(text) > 0.1:
            hints.append("high_capitalization")
        
        return hints
    
    def _create_text_validation_message(
        self,
        expected: str,
        actual: str,
        confidence_score: ConfidenceScore,
        status: ValidationStatus,
        threshold: float,
        matching_mode: TextMatchingMode
    ) -> str:
        """Create detailed validation message for text content."""
        
        message = f"Text validation ({matching_mode.value}): {status.value} "
        message += f"(confidence: {confidence_score.value:.3f}, threshold: {threshold:.3f})"
        
        if status == ValidationStatus.FAILED:
            message += f"\nExpected: '{expected[:100]}...'"
            message += f"\nActual: '{actual[:100]}...'"
            
            # Add helpful details about the failure
            if confidence_score.components:
                best_component = max(confidence_score.components.items(), key=lambda x: x[1])
                message += f"\nBest match component: {best_component[0]} = {best_component[1]:.3f}"
        
        return message