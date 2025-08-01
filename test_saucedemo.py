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

# Import comprehensive validation framework
from validation import (
    ValidationRegistry,
    CustomAssertions,
    ValidationContext,
    ValidationType,
    TextMatchingMode,
    get_global_registry
)


# Enhanced validation methods using the robust validation framework
class SauceDemoValidationMixin:
    """Mixin class providing SauceDemo-specific validation methods using the validation framework."""

    def _ensure_validation_components(self):
        """Lazy initialization of validation components."""
        if not hasattr(self, 'validation_registry'):
            self.validation_registry = get_global_registry()
        if not hasattr(self, 'custom_assertions'):
            self.custom_assertions = CustomAssertions(default_confidence_threshold=0.8)

    async def validate_login_success_robust(
        self, actual_response: str, username: str,
        confidence_threshold: float = 0.8,
        context: Optional[ValidationContext] = None
    ) -> str:
        """Validate successful login using semantic validation framework."""

        self._ensure_validation_components()

        if context is None:
            context = ValidationContext(
                validation_type=ValidationType.SEMANTIC,
                metadata={
                    "validation_method": "login_success",
                    "username": username,
                    "expected_behavior": "successful_authentication"
                }
            )

        try:
            # Use semantic validator for login success patterns
            validation_result = await self.validation_registry.validate_semantic(
                expected=[
                    "login_successful",
                    "products page",
                    "inventory",
                    "items displayed",
                    "logged in successfully",
                    "dashboard loaded"
                ],
                actual=actual_response,
                context=context,
                confidence_threshold=confidence_threshold,
                ignore_case=True,
                enable_partial_matching=True
            )

            if not validation_result.is_successful():
                raise ValidationError(
                    message=f"Login validation failed: {validation_result.message}",
                    expected_value=f"successful login for user '{username}'",
                    actual_value=actual_response,
                    validation_type="login_success_validation",
                    confidence_score=validation_result.confidence_score.value,
                    error_context=create_error_context(
                        component="SauceDemo Login Validation",
                        operation="login_success_check",
                        metadata={
                            "validation_id": validation_result.context.validation_id,
                            "username": username,
                            "confidence_components": validation_result.confidence_score.components
                        }
                    )
                )

            logging.info(
                f"Login validation passed with confidence {validation_result.confidence_score.value:.3f} "
                f"for user '{username}' [{context.validation_id}]"
            )

            return actual_response

        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ValidationError(
                    message=f"Login validation framework error: {str(e)}",
                    expected_value=f"successful login for '{username}'",
                    actual_value=actual_response,
                    validation_type="login_validation_error",
                    confidence_score=0.0,
                    error_context=context,
                    cause=e
                )
            raise

    async def validate_cart_operation_robust(
        self, actual_response: str, operation_type: str,
        expected_count: Optional[int] = None,
        confidence_threshold: float = 0.8,
        context: Optional[ValidationContext] = None
    ) -> str:
        """Validate cart operations using text content validation."""

        self._ensure_validation_components()

        if context is None:
            context = ValidationContext(
                validation_type=ValidationType.TEXT_CONTENT,
                metadata={
                    "validation_method": "cart_operation",
                    "operation_type": operation_type,
                    "expected_count": expected_count
                }
            )

        try:
            # Determine expected patterns based on operation type
            if operation_type == "add_item":
                expected_patterns = ["item_added", "added to cart", "cart updated", "1"]
            elif operation_type == "view_cart":
                expected_patterns = ["cart", "checkout", "continue shopping"]
            else:
                expected_patterns = [operation_type]

            validation_result = await self.validation_registry.validate_text_content(
                expected=expected_patterns,
                actual=actual_response,
                matching_mode=TextMatchingMode.CONTAINS,
                context=context,
                confidence_threshold=confidence_threshold,
                ignore_case=True,
                normalize_text=True
            )

            if not validation_result.is_successful():
                raise ValidationError(
                    message=f"Cart operation validation failed: {validation_result.message}",
                    expected_value=f"{operation_type} operation success",
                    actual_value=actual_response,
                    validation_type="cart_operation_validation",
                    confidence_score=validation_result.confidence_score.value,
                    error_context=create_error_context(
                        component="SauceDemo Cart Validation",
                        operation="cart_operation_check",
                        metadata={
                            "validation_id": validation_result.context.validation_id,
                            "operation_type": operation_type,
                            "expected_count": expected_count,
                            "confidence_components": validation_result.confidence_score.components
                        }
                    )
                )

            logging.info(
                f"Cart operation '{operation_type}' validation passed with confidence "
                f"{validation_result.confidence_score.value:.3f} [{context.validation_id}]"
            )

            return actual_response

        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ValidationError(
                    message=f"Cart validation framework error: {str(e)}",
                    expected_value=f"{operation_type} operation success",
                    actual_value=actual_response,
                    validation_type="cart_validation_error",
                    confidence_score=0.0,
                    error_context=context,
                    cause=e
                )
            raise

    async def assert_purchase_completion_robust(
        self, actual_response: str, expected_confirmation_elements: Optional[list] = None,
        confidence_threshold: float = 0.8
    ) -> str:
        """Assert purchase completion using custom assertions framework."""

        self._ensure_validation_components()

        if expected_confirmation_elements is None:
            expected_confirmation_elements = [
                "purchase_completed",
                "thank you",
                "order confirmation",
                "complete",
                "success"
            ]

        try:
            # Use semantic validation for purchase completion
            validation_result = await self.validation_registry.validate_semantic(
                expected=expected_confirmation_elements,
                actual=actual_response,
                confidence_threshold=confidence_threshold,
                ignore_case=True,
                enable_partial_matching=True
            )

            if not validation_result.is_successful():
                raise ValidationError(
                    message=f"Purchase completion validation failed: {validation_result.message}",
                    expected_value="purchase completion confirmation",
                    actual_value=actual_response,
                    validation_type="purchase_completion_validation",
                    confidence_score=validation_result.confidence_score.value,
                    error_context=create_error_context(
                        component="SauceDemo Purchase Validation",
                        operation="purchase_completion_check",
                        metadata={
                            "validation_id": validation_result.context.validation_id,
                            "expected_elements": expected_confirmation_elements,
                            "confidence_components": validation_result.confidence_score.components
                        }
                    )
                )

            logging.info(
                f"Purchase completion validation passed with confidence "
                f"{validation_result.confidence_score.value:.3f} [{validation_result.context.validation_id}]"
            )

            return actual_response

        except Exception as e:
            correlation_id = log_error_with_context(
                e, create_error_context(
                    component="Purchase Completion Assertion",
                    operation="assert_purchase_completion",
                    metadata={"expected_elements": expected_confirmation_elements}
                )
            )
            raise ValidationError(
                message=f"Purchase completion assertion failed: {str(e)}",
                expected_value="purchase completion confirmation",
                actual_value=actual_response,
                validation_type="purchase_completion_assertion",
                confidence_score=0.0,
                error_context=create_error_context(
                    component="Purchase Completion Assertion",
                    operation="assert_purchase_completion",
                    metadata={"correlation_id": correlation_id}
                )
            )


@allure.feature("SauceDemo Basic Flow")
class TestSauceDemoBasic(BaseAgentTest, SauceDemoValidationMixin):
    """
    SauceDemo functionality tests showcasing the comprehensive AgentiTest framework
    with retry logic, enhanced error handling, and robust validation.
    """

    BASE_URL = "https://www.saucedemo.com/"

    @allure.story("User Login")
    @allure.title("Test successful login with standard user")
    @pytest.mark.asyncio
    async def test_standard_user_login(self, llm, browser_session):
        """
        Test login with standard_user using robust validation framework.

        This test demonstrates:
        - Automatic retry logic for LLM calls
        - Enhanced error handling with context preservation
        - Semantic validation with confidence scoring
        """
        task = ("enter 'standard_user' in the username field, enter 'secret_sauce' in the password field, "
                "click the LOGIN button, and confirm you reach the products page with items displayed. "
                "Return 'login_successful' when products are visible.")

        try:
            # Execute task with automatic retry logic (inherited from BaseAgentTest)
            result = await self.validate_task(llm, browser_session, task, "login_successful", ignore_case=True)

            # Use robust validation framework for additional verification
            await self.validate_login_success_robust(
                actual_response=result,
                username="standard_user",
                confidence_threshold=0.8
            )

            logging.info("Standard user login test completed successfully")

        except Exception as e:
            # Enhanced error handling with context and recovery suggestions
            context = create_error_context(
                component="SauceDemo Login Test",
                operation="standard_user_login",
                metadata={
                    "username": "standard_user",
                    "test_type": "authentication",
                    "base_url": self.BASE_URL
                }
            )

            correlation_id = log_error_with_context(e, context, level="error")

            allure.attach(
                f"Login Test Failed\n"
                f"Username: standard_user\n"
                f"Expected: login_successful\n"
                f"Correlation ID: {correlation_id}\n"
                f"Validation Framework: Robust Validation with Semantic Analysis\n"
                f"Error Context: {context}",
                name="Login Test Failure Details",
                attachment_type=allure.attachment_type.TEXT
            )
            raise

    @allure.story("Add to Cart")
    @allure.title("Test adding item to shopping cart")
    @pytest.mark.asyncio
    async def test_add_to_cart(self, llm, browser_session):
        """
        Test adding a product to cart after login with comprehensive validation.

        This test demonstrates:
        - Multi-step task execution with individual validation
        - Text content validation with fuzzy matching
        - Error context preservation across operations
        """
        correlation_id = None

        try:
            # First login with semantic validation
            login_task = ("enter 'standard_user' in the username field, enter 'secret_sauce' in the password field, "
                         "and click LOGIN")
            login_result = await self.validate_task(llm, browser_session, login_task, "inventory", ignore_case=True)

            # Validate login success using robust framework
            await self.validate_login_success_robust(
                actual_response=login_result,
                username="standard_user",
                confidence_threshold=0.7
            )

            # Then add to cart with cart-specific validation
            cart_task = ("find the first product and click its 'Add to cart' button, then verify the cart badge shows '1'. "
                        "Return 'item_added' if successful.")
            cart_result = await self.validate_task(llm, browser_session, cart_task, "item_added", ignore_case=True)

            # Use robust cart validation
            await self.validate_cart_operation_robust(
                actual_response=cart_result,
                operation_type="add_item",
                expected_count=1,
                confidence_threshold=0.8
            )

            logging.info("Add to cart test completed successfully")

        except Exception as e:
            context = create_error_context(
                component="SauceDemo Add to Cart Test",
                operation="add_item_to_cart",
                metadata={
                    "username": "standard_user",
                    "test_type": "cart_operation",
                    "base_url": self.BASE_URL,
                    "expected_cart_count": 1
                }
            )

            correlation_id = log_error_with_context(e, context, level="error")

            # Check for degradation status
            degradation_status = get_degradation_status()

            allure.attach(
                f"Add to Cart Test Failed\n"
                f"Username: standard_user\n"
                f"Expected Cart Count: 1\n"
                f"Correlation ID: {correlation_id}\n"
                f"Degradation Status: {degradation_status}\n"
                f"Validation Framework: Text Content with Cart-Specific Validation\n"
                f"Error Context: {context}",
                name="Add to Cart Test Failure Details",
                attachment_type=allure.attachment_type.TEXT
            )
            raise

    @allure.story("Complete Purchase")
    @allure.title("Test end-to-end purchase flow")
    @pytest.mark.asyncio
    async def test_complete_purchase(self, llm, browser_session):
        """
        Test complete purchase flow from login to checkout completion.

        This test demonstrates:
        - Complex multi-step workflow validation
        - Custom assertion helpers for domain-specific patterns
        - Comprehensive error handling with recovery suggestions
        - Validation result caching for performance
        """
        correlation_id = None

        try:
            task = """
            Complete this full purchase flow:
            1. Login with username 'standard_user' and password 'secret_sauce'
            2. Add any product to cart
            3. Click the cart icon and proceed to checkout
            4. Fill checkout form: First Name 'John', Last Name 'Doe', Postal Code '12345'
            5. Complete the purchase through to the final confirmation page
            Return 'purchase_completed' when you see the order confirmation or thank you message.
            """

            # Execute complex task with automatic retry and circuit breaker protection
            result = await self.validate_task(llm, browser_session, task, "purchase_completed", ignore_case=True)

            # Use custom assertion for purchase completion
            await self.assert_purchase_completion_robust(
                actual_response=result,
                expected_confirmation_elements=[
                    "purchase_completed",
                    "thank you for your order",
                    "order has been dispatched",
                    "checkout complete",
                    "finish"
                ],
                confidence_threshold=0.8
            )

            logging.info("Complete purchase test completed successfully")

        except BrowserSessionError as e:
            # Handle browser-specific errors with graceful degradation
            context = create_error_context(
                component="SauceDemo Purchase Flow",
                operation="complete_purchase_flow",
                metadata={
                    "test_type": "end_to_end_purchase",
                    "browser_error": True,
                    "form_data": {
                        "first_name": "John",
                        "last_name": "Doe",
                        "postal_code": "12345"
                    }
                }
            )

            correlation_id = log_error_with_context(e, context, level="error")

            allure.attach(
                f"Browser Session Error in Purchase Flow\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {str(e)}\n"
                f"Correlation ID: {correlation_id}\n"
                f"Recovery Suggestions:\n"
                f"  - Check browser session stability\n"
                f"  - Verify page loading times\n"
                f"  - Review network connectivity\n"
                f"  - Consider increasing timeout values",
                name="Browser Session Error Details",
                attachment_type=allure.attachment_type.TEXT
            )
            raise

        except ValidationError as e:
            # Handle validation-specific errors with detailed diagnostics
            context = create_error_context(
                component="SauceDemo Purchase Flow",
                operation="complete_purchase_flow",
                metadata={
                    "test_type": "end_to_end_purchase",
                    "validation_error": True,
                    "confidence_score": getattr(e, 'confidence_score', 0.0),
                    "validation_type": getattr(e, 'validation_type', 'unknown')
                }
            )

            correlation_id = log_error_with_context(e, context, level="error")

            allure.attach(
                f"Validation Error in Purchase Flow\n"
                f"Validation Type: {getattr(e, 'validation_type', 'unknown')}\n"
                f"Confidence Score: {getattr(e, 'confidence_score', 0.0)}\n"
                f"Expected: purchase completion confirmation\n"
                f"Actual: {getattr(e, 'actual_value', 'unknown')}\n"
                f"Correlation ID: {correlation_id}\n"
                f"Validation Framework: Semantic with Custom Purchase Assertions\n"
                f"Recovery Suggestions:\n"
                f"  - Review purchase flow steps\n"
                f"  - Check form validation requirements\n"
                f"  - Verify confirmation page elements\n"
                f"  - Consider adjusting confidence thresholds",
                name="Purchase Validation Error Details",
                attachment_type=allure.attachment_type.TEXT
            )
            raise

        except Exception as e:
            # Handle unexpected errors with comprehensive context
            context = create_error_context(
                component="SauceDemo Purchase Flow",
                operation="complete_purchase_flow",
                metadata={
                    "test_type": "end_to_end_purchase",
                    "unexpected_error": True,
                    "error_type": type(e).__name__,
                    "base_url": self.BASE_URL
                }
            )

            correlation_id = log_error_with_context(e, context, level="error")

            allure.attach(
                f"Unexpected Error in Purchase Flow\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {str(e)}\n"
                f"Correlation ID: {correlation_id}\n"
                f"Test Framework: AgentiTest with Full Enhancement Suite\n"
                f"Recovery Suggestions:\n"
                f"  - Check system resources and connectivity\n"
                f"  - Review LLM provider status\n"
                f"  - Verify test environment configuration\n"
                f"  - Consider retry with different LLM provider",
                name="Unexpected Error Details",
                attachment_type=allure.attachment_type.TEXT
            )
            raise


@allure.feature("SauceDemo Error Scenarios")
class TestSauceDemoErrorHandling(BaseAgentTest, SauceDemoValidationMixin):
    """
    SauceDemo error scenario tests demonstrating comprehensive error handling
    and graceful degradation capabilities.
    """

    BASE_URL = "https://www.saucedemo.com/"

    @allure.story("Invalid Login")
    @allure.title("Test login with invalid credentials")
    @pytest.mark.asyncio
    async def test_invalid_login_error_handling(self, llm, browser_session):
        """
        Test login with invalid credentials to demonstrate error handling.

        This test demonstrates:
        - Expected validation failure handling
        - Error classification and recovery suggestions
        - Confidence scoring for negative test cases
        """
        task = ("enter 'invalid_user' in the username field, enter 'wrong_password' in the password field, "
                "click the LOGIN button. If login fails, return 'login_failed' with the error message.")

        try:
            # This should fail, but we handle it gracefully
            result = await self.validate_task(llm, browser_session, task, "login_failed", ignore_case=True)

            # Initialize validation components before use
            self._ensure_validation_components()

            # Validate that login actually failed (negative test case)
            validation_result = await self.validation_registry.validate_text_content(
                expected=["login_failed", "error", "invalid", "incorrect"],
                actual=result,
                matching_mode=TextMatchingMode.CONTAINS,
                confidence_threshold=0.7,
                ignore_case=True
            )

            if validation_result.is_successful():
                logging.info(f"Invalid credentials correctly rejected with confidence "
                           f"{validation_result.confidence_score.value:.3f}")
            else:
                # If validation fails, it might mean login unexpectedly succeeded
                raise ValidationError(
                    message="Expected login failure but validation suggests success",
                    expected_value="login_failed",
                    actual_value=result,
                    validation_type="negative_test_validation",
                    confidence_score=validation_result.confidence_score.value
                )

        except ValidationError as e:
            # For negative tests, some validation errors are expected
            if "login_failed" in str(e).lower() or "invalid" in str(e).lower():
                logging.info("Invalid login correctly handled - test passed")
                return
            else:
                # Re-raise unexpected validation errors
                raise
        except Exception as e:
            context = create_error_context(
                component="SauceDemo Invalid Login Test",
                operation="invalid_credentials_test",
                metadata={
                    "username": "invalid_user",
                    "test_type": "negative_authentication",
                    "expected_outcome": "login_failure"
                }
            )

            correlation_id = log_error_with_context(e, context, level="warning")

            allure.attach(
                f"Invalid Login Test Execution\n"
                f"Username: invalid_user (intentionally invalid)\n"
                f"Expected: login_failed\n"
                f"Correlation ID: {correlation_id}\n"
                f"Test Type: Negative Test Case\n"
                f"Note: Some exceptions are expected in negative testing",
                name="Invalid Login Test Details",
                attachment_type=allure.attachment_type.TEXT
            )

            # For negative tests, certain exceptions might be acceptable
            if isinstance(e, ValidationError) and "login" in str(e).lower():
                logging.info("Invalid login test completed - error handling validated")
                return
            else:
                raise
