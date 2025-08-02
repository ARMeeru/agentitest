import allure
import pytest
import logging
from typing import Optional, Dict, Any

from conftest import BaseAgentTest
from exceptions import (
    ValidationError,
    BrowserSessionError,
    create_error_context,
    log_error_with_context,
    get_degradation_status
)

# Import new validation framework
from validation import (
    ValidationRegistry,
    CustomAssertions,
    ValidationContext,
    ValidationType,
    TextMatchingMode,
    get_global_registry
)


# Enhanced validation methods using the new validation framework
class RobustValidationMixin:
    """Mixin class providing robust validation methods using the new validation framework."""
    
    def _ensure_validation_components(self):
        """Lazy initialization of validation components."""
        if not hasattr(self, 'validation_registry'):
            self.validation_registry = get_global_registry()
        if not hasattr(self, 'custom_assertions'):
            self.custom_assertions = CustomAssertions(default_confidence_threshold=0.8)
    
    async def validate_task_with_semantic_framework(
        self, llm, browser_session, task_instruction: str, 
        expected_outcomes: list, confidence_threshold: float = 0.8,
        context: Optional[ValidationContext] = None
    ) -> str:
        """Enhanced validation using semantic validation framework."""
        
        self._ensure_validation_components()
        
        if context is None:
            context = ValidationContext(
                validation_type=ValidationType.SEMANTIC,
                metadata={
                    "task_instruction": task_instruction,
                    "expected_outcomes": expected_outcomes,
                    "validation_method": "semantic_framework"
                }
            )
        
        try:
            # Execute the task using semantic validation method
            result = await self.validate_task(
                llm, browser_session, task_instruction, 
                expected_outcomes=expected_outcomes or ["success"],
                confidence_threshold=confidence_threshold,
                ignore_case=True,
                enable_partial_matching=True,
                validation_context=context.metadata
            )
            
            # Use semantic validator for confidence scoring
            validation_result = await self.validation_registry.validate_semantic(
                expected=expected_outcomes,
                actual=result,
                context=context,
                confidence_threshold=confidence_threshold,
                ignore_case=True,
                enable_partial_matching=True
            )
            
            if not validation_result.is_successful():
                raise ValidationError(
                    message=f"Semantic validation failed: {validation_result.message}",
                    expected_value=expected_outcomes,
                    actual_value=result,
                    validation_type="semantic_framework",
                    confidence_score=validation_result.confidence_score.value,
                    error_context=create_error_context(
                        component="Semantic Framework",
                        operation="semantic_validation",
                        metadata={
                            "validation_id": validation_result.context.validation_id,
                            "confidence_components": validation_result.confidence_score.components
                        }
                    )
                )
            
            logging.info(
                f"Semantic validation passed with confidence {validation_result.confidence_score.value:.3f} "
                f"[{context.validation_id}]"
            )
            
            return result
            
        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ValidationError(
                    message=f"Semantic framework validation error: {str(e)}",
                    expected_value=expected_outcomes,
                    actual_value="unknown",
                    validation_type="semantic_framework_error",
                    confidence_score=0.0,
                    error_context=context,
                    cause=e
                )
            raise
    
    async def validate_text_content_robust(
        self, actual_text: str, expected_text: str,
        matching_mode: TextMatchingMode = TextMatchingMode.FUZZY,
        confidence_threshold: float = 0.8,
        context: Optional[ValidationContext] = None
    ) -> str:
        """Robust text content validation using the text validation framework."""
        
        self._ensure_validation_components()
        
        if context is None:
            context = ValidationContext(
                validation_type=ValidationType.TEXT_CONTENT,
                metadata={
                    "matching_mode": matching_mode.value,
                    "validation_method": "text_content_framework"
                }
            )
        
        try:
            validation_result = await self.validation_registry.validate_text_content(
                expected=expected_text,
                actual=actual_text,
                matching_mode=matching_mode,
                context=context,
                confidence_threshold=confidence_threshold,
                ignore_case=True,
                normalize_text=True
            )
            
            if not validation_result.is_successful():
                raise ValidationError(
                    message=f"Text content validation failed: {validation_result.message}",
                    expected_value=expected_text,
                    actual_value=actual_text,
                    validation_type="text_content_framework",
                    confidence_score=validation_result.confidence_score.value,
                    error_context=create_error_context(
                        component="Text Content Framework",
                        operation="text_validation",
                        metadata={
                            "validation_id": validation_result.context.validation_id,
                            "matching_mode": matching_mode.value,
                            "confidence_components": validation_result.confidence_score.components
                        }
                    )
                )
            
            logging.info(
                f"Text content validation passed with confidence {validation_result.confidence_score.value:.3f} "
                f"[{context.validation_id}]"
            )
            
            return actual_text
            
        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ValidationError(
                    message=f"Text content framework validation error: {str(e)}",
                    expected_value=expected_text,
                    actual_value=actual_text,
                    validation_type="text_content_framework_error",
                    confidence_score=0.0,
                    error_context=context,
                    cause=e
                )
            raise
    
    async def assert_page_loaded_robust(
        self, actual_response: str, expected_indicators: Optional[list] = None,
        confidence_threshold: float = 0.8
    ) -> str:
        """Assert page loaded using custom assertions framework."""
        
        self._ensure_validation_components()
        
        try:
            validation_result = await self.custom_assertions.assert_page_loaded(
                actual_content=actual_response,
                expected_indicators=expected_indicators,
                confidence_threshold=confidence_threshold
            )
            
            logging.info(
                f"Page load assertion passed with confidence {validation_result.confidence_score.value:.3f} "
                f"[{validation_result.context.validation_id}]"
            )
            
            return actual_response
            
        except Exception as e:
            correlation_id = log_error_with_context(
                e, create_error_context(
                    component="Page Load Assertion",
                    operation="assert_page_loaded"
                )
            )
            raise ValidationError(
                message=f"Page load assertion failed: {str(e)}",
                expected_value=expected_indicators or ["loaded"],
                actual_value=actual_response,
                validation_type="page_load_assertion",
                confidence_score=0.0,
                error_context=create_error_context(
                    component="Page Load Assertion",
                    operation="assert_page_loaded",
                    metadata={"correlation_id": correlation_id}
                )
            )
    
    async def assert_search_results_robust(
        self, actual_content: str, search_term: str,
        minimum_results: int = 1, confidence_threshold: float = 0.8
    ) -> str:
        """Assert search results using custom assertions framework."""
        
        self._ensure_validation_components()
        
        try:
            validation_result = await self.custom_assertions.assert_search_results_displayed(
                actual_content=actual_content,
                search_term=search_term,
                minimum_results=minimum_results,
                confidence_threshold=confidence_threshold
            )
            
            logging.info(
                f"Search results assertion passed with confidence {validation_result.confidence_score.value:.3f} "
                f"[{validation_result.context.validation_id}]"
            )
            
            return actual_content
            
        except Exception as e:
            correlation_id = log_error_with_context(
                e, create_error_context(
                    component="Search Results Assertion",
                    operation="assert_search_results",
                    metadata={"search_term": search_term}
                )
            )
            raise ValidationError(
                message=f"Search results assertion failed: {str(e)}",
                expected_value=f"search results for '{search_term}'",
                actual_value=actual_content,
                validation_type="search_results_assertion",
                confidence_score=0.0,
                error_context=create_error_context(
                    component="Search Results Assertion",
                    operation="assert_search_results",
                    metadata={"correlation_id": correlation_id, "search_term": search_term}
                )
            )
    
    async def assert_no_search_results_robust(
        self, actual_content: str, search_term: str,
        confidence_threshold: float = 0.8
    ) -> str:
        """Assert no search results using custom assertions framework."""
        
        self._ensure_validation_components()
        
        try:
            # Add debug logging for the actual content being validated
            logging.info(f"Validating no search results for term '{search_term}' with content length: {len(actual_content)}")
            logging.debug(f"Actual content: {actual_content[:200]}...")
            
            validation_result = await self.custom_assertions.assert_no_search_results(
                actual_content=actual_content,
                search_term=search_term,
                confidence_threshold=confidence_threshold
            )
            
            logging.info(
                f"No search results assertion passed with confidence {validation_result.confidence_score.value:.3f} "
                f"[{validation_result.context.validation_id}]"
            )
            
            return actual_content
            
        except Exception as e:
            correlation_id = log_error_with_context(
                e, create_error_context(
                    component="No Search Results Assertion",
                    operation="assert_no_search_results",
                    metadata={"search_term": search_term}
                )
            )
            raise ValidationError(
                message=f"No search results assertion failed: {str(e)}",
                expected_value=f"no results for '{search_term}'",
                actual_value=actual_content,
                validation_type="no_search_results_assertion", 
                confidence_score=0.0,
                error_context=create_error_context(
                    component="No Search Results Assertion",
                    operation="assert_no_search_results",
                    metadata={"correlation_id": correlation_id, "search_term": search_term}
                )
            )


@allure.feature("Home Page Content")
class TestHomePageStats(BaseAgentTest, RobustValidationMixin):
    """Tests the content and navigation of the home page with robust validation framework."""

    EXPECTED_FORUM_RESULT = "forum_loaded"

    @allure.story("Main Navigation Links")
    @allure.title("Test Navigation to {link_text}")
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "link_text, expected_path_segment",
        [
            ("Google Workspace", "google-workspace"),
            ("AppSheet", "appsheet"),
            ("Looker & Looker Studio", "looker"),
            ("Google Cloud", "google-cloud"),  # Updated to match actual website structure
        ],
    )
    async def test_main_navigation(
        self, llm, browser_session, link_text, expected_path_segment
    ):
        """Tests navigation to main sections of the website using robust validation."""
        task = f"click on the '{link_text}' link in the main navigation, and then return the final URL of the page."
        
        # Use semantic framework validation for better confidence scoring
        result_url = await self.validate_task_with_semantic_framework(
            llm, browser_session, task, 
            expected_outcomes=[
                expected_path_segment,
                f"page containing {expected_path_segment}",
                f"navigated to {link_text}",
                f"URL with {expected_path_segment}"
            ],
            confidence_threshold=0.8
        )
        
        # Additional text content validation for URL verification
        await self.validate_text_content_robust(
            actual_text=result_url,
            expected_text=expected_path_segment,
            matching_mode=TextMatchingMode.CONTAINS,
            confidence_threshold=0.9
        )
        
        logging.info(f"Navigation to '{link_text}' successful - URL: {result_url}")

    @allure.story("Community Statistics")
    @allure.title("Test Forum Page Loads Successfully")
    @pytest.mark.asyncio
    async def test_forum_loads_successfully(self, llm, browser_session):
        """Tests that the forum loads successfully using robust validation."""
        task = "confirm that the Google Developer Community forum page has loaded successfully and displays community content. Return 'forum_loaded' if the page loads with community categories visible."
        
        # Use semantic validation for comprehensive page load validation
        result = await self.validate_task(
            llm, browser_session, task, 
            expected_outcomes=[self.EXPECTED_FORUM_RESULT, "forum_loaded", "community", "categories", "developer"],
            confidence_threshold=0.8,
            ignore_case=True,
            enable_partial_matching=True
        )
        
        # Additional validation using custom assertions
        await self.assert_page_loaded_robust(
            actual_response=result,
            expected_indicators=[
                "forum_loaded", 
                "community", 
                "categories", 
                "developer", 
                "loaded successfully"
            ],
            confidence_threshold=0.8
        )
        
        logging.info(f"Forum page loaded successfully with result: {result}")


@allure.feature("Search Functionality")
class TestSearch(BaseAgentTest, RobustValidationMixin):
    """Tests for the website's search functionality with robust validation framework."""

    EXPECTED_NO_RESULTS = "no_results_found"

    @allure.story("Searching for Terms")
    @allure.title("Search for '{term}'")
    @pytest.mark.asyncio
    @pytest.mark.parametrize("term", ["BigQuery", "Vertex AI"])
    async def test_search_for_term(self, llm, browser_session, term):
        """Tests searching for a term and verifying results are shown using robust validation."""
        task = f"locate the search input field, enter '{term}', submit the search by pressing enter, then confirm that search results for '{term}' are displayed on the page. Return 'search_results_displayed' if results are shown."
        
        # Execute search task with semantic validation
        result = await self.validate_task(
            llm, browser_session, task, 
            expected_outcomes=["search_results_displayed", "results found", f"results for {term}", "search complete"],
            confidence_threshold=0.8,
            ignore_case=True,
            enable_partial_matching=True
        )
        
        # Use text content validation for confirmation string
        await self.validate_text_content_robust(
            actual_text=result,
            expected_text="search_results_displayed",
            matching_mode=TextMatchingMode.CONTAINS,
            confidence_threshold=0.9
        )
        
        logging.info(f"Search for '{term}' completed successfully with results displayed")

    @allure.story("Searching for Non-Existent Term")
    @allure.title("Search for a Non-Existent Term")
    @pytest.mark.asyncio
    async def test_search_for_non_existent_term(self, llm, browser_session):
        """Tests searching for a non-existent term using robust validation."""
        
        term = "a_very_unlikely_search_term_xyz"
        task = f"find the search bar, type '{term}', press enter, and confirm that a 'no results' message is displayed. Return '{self.EXPECTED_NO_RESULTS}' if it is."
        
        try:
            # Execute search task with semantic validation for no results
            result = await self.validate_task(
                llm, browser_session, task, 
                expected_outcomes=[self.EXPECTED_NO_RESULTS, "no results", "nothing found", "0 results"],
                confidence_threshold=0.8,
                ignore_case=True,
                enable_partial_matching=True
            )
            
            # Use robust no search results assertion
            await self.assert_no_search_results_robust(
                actual_content=result,
                search_term=term,
                confidence_threshold=0.7
            )
            
            logging.info(f"No-results validation successful for term '{term}'")
            
        except Exception as e:
            context = create_error_context(
                component="Search Test",
                operation="no_results_validation",
                metadata={
                    "search_term": term,
                    "expected_response": self.EXPECTED_NO_RESULTS,
                    "test_type": "negative_search"
                }
            )
            
            correlation_id = log_error_with_context(e, context, level="error")
            
            allure.attach(
                f"No-Results Search Test Failed\n"
                f"Search Term: {term}\n"
                f"Expected: {self.EXPECTED_NO_RESULTS}\n"
                f"Correlation ID: {correlation_id}\n"
                f"Validation Framework: Robust Validation",
                name="No-Results Test Failure Details",
                attachment_type=allure.attachment_type.TEXT
            )
            raise
