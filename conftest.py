import base64
import logging
import os
import platform
import sys
from importlib.metadata import version
from typing import AsyncGenerator, Dict, Optional, Union, List, Any

import allure
import pytest
from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.utils import get_browser_use_version
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from core.retry import (
    LLMProvider,
    CircuitBreakerOpenException,
    LLMAPIException,
    with_llm_retry,
    get_correlation_id,
    get_circuit_breaker_status,
    retry_manager
)

# Import enhanced error handling
from exceptions import (
    ConfigurationError,
    LLMProviderError,
    BrowserSessionError,
    ValidationError,
    create_error_context,
    log_error_with_context,
    configure_error_logging
)

# Load environment variables from .env file
load_dotenv()

# Configure structured error logging with security features
configure_error_logging(level="INFO", format_type="json", enable_security=True)

# Try to initialize secure credential management
try:
    from core.credential_manager import get_credential_loader
    _credential_loader = get_credential_loader()
    SECURE_CREDENTIALS_AVAILABLE = True
except ImportError:
    _credential_loader = None
    SECURE_CREDENTIALS_AVAILABLE = False


def get_api_key_secure(provider: str, env_var: str) -> Optional[str]:
    """Get API key using secure credential loader if available, otherwise environment."""
    if SECURE_CREDENTIALS_AVAILABLE and _credential_loader:
        return _credential_loader.get_credential(
            f"{provider}_api_key",
            required=False,
            fallback_env_var=env_var
        )
    else:
        return os.getenv(env_var)


def create_llm_instance():
    # Factory function to create the appropriate LLM instance based on environment configuration
    # Integrates circuit breaker pattern to prevent cascade failures after consecutive LLM provider failures
    # Reads the LLM_PROVIDER environment variable to determine which provider to use
    # Supported values: "gemini", "openai", "anthropic", "azure", "groq"
    # Returns: An instance of the appropriate LLM provider
    # Raises: ValueError, ImportError, EnvironmentError, CircuitBreakerOpenException
    # Get the provider from environment, validate it's not empty
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    # Check circuit breaker before attempting to create instance
    if not retry_manager.check_circuit_breaker(provider):
        raise CircuitBreakerOpenException(provider)

    if not provider:
        raise ConfigurationError(
            message="LLM_PROVIDER environment variable is empty",
            config_key="LLM_PROVIDER",
            expected_format="One of: 'gemini', 'openai', 'anthropic', 'azure', 'groq'",
            error_context=create_error_context(
                component="LLM Configuration",
                operation="provider_validation"
            )
        )

    # Define supported providers for better error messages
    supported_providers = ["gemini", "openai", "anthropic", "azure", "groq"]

    if provider not in supported_providers:
        raise ConfigurationError(
            message=f"Invalid LLM_PROVIDER: '{provider}'",
            config_key="LLM_PROVIDER",
            expected_format=f"One of: {', '.join(supported_providers)}",
            error_context=create_error_context(
                component="LLM Configuration",
                operation="provider_validation",
                metadata={"provided_value": provider, "supported_providers": supported_providers}
            )
        )

    try:
        if provider == "gemini":
            # Try to import the required module
            try:
                from browser_use.llm import ChatGoogle
            except ImportError as e:
                raise ConfigurationError(
                    message="Failed to import ChatGoogle for Gemini provider",
                    config_key="GEMINI_DEPENDENCIES",
                    expected_format="pip install browser_use[gemini]",
                    error_context=create_error_context(
                        component="LLM Configuration",
                        operation="import_validation",
                        provider="gemini"
                    ),
                    cause=e
                )

            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
            api_key = get_api_key_secure("gemini", "GOOGLE_API_KEY")

            # Validate API key
            if not api_key or api_key == "YOUR_API_KEY":
                raise ConfigurationError(
                    message="GOOGLE_API_KEY is not properly configured for Gemini provider",
                    config_key="GOOGLE_API_KEY",
                    expected_format="Valid API key from Google AI Studio",
                    error_context=create_error_context(
                        component="LLM Configuration",
                        operation="api_key_validation",
                        provider="gemini",
                        metadata={"help_url": "https://makersuite.google.com/app/apikey"}
                    )
                )

            # Basic API key validation (removed specific format validation for security)

            try:
                llm_instance = ChatGoogle(model=model_name, api_key=api_key)
                retry_manager.record_success(provider)
                return llm_instance
            except Exception as e:
                retry_manager.record_failure(provider, e)
                context = create_error_context(
                    component="LLM Provider",
                    operation="instance_creation",
                    provider=provider
                )
                llm_error = LLMProviderError(
                    message=f"Failed to create Gemini instance: {str(e)}",
                    provider=provider,
                    error_context=context,
                    cause=e
                )
                log_error_with_context(llm_error, context)
                raise llm_error

        elif provider == "openai":
            # Try to import the required module
            try:
                from browser_use.llm import ChatOpenAI
            except ImportError as e:
                raise ImportError(
                    f"Failed to import ChatOpenAI for OpenAI provider. "
                    f"Please ensure browser_use[openai] is installed. "
                    f"You can install it with: pip install browser_use[openai]\n"
                    f"Original error: {str(e)}"
                )

            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            api_key = get_api_key_secure("openai", "OPENAI_API_KEY")

            # Validate API key
            if not api_key or api_key == "YOUR_API_KEY":
                raise ConfigurationError(
                    message="OPENAI_API_KEY is not properly configured for OpenAI provider",
                    config_key="OPENAI_API_KEY",
                    expected_format="Valid API key from OpenAI Platform",
                    error_context=create_error_context(
                        component="LLM Configuration",
                        operation="api_key_validation",
                        provider="openai",
                        metadata={"help_url": "https://platform.openai.com/api-keys"}
                    )
                )

            # Basic API key validation (removed specific format validation for security)

            try:
                llm_instance = ChatOpenAI(model=model_name, api_key=api_key)
                retry_manager.record_success(provider)
                return llm_instance
            except Exception as e:
                retry_manager.record_failure(provider, e)
                raise LLMAPIException(provider, f"Failed to create OpenAI instance: {str(e)}") from e

        elif provider == "anthropic":
            # Try to import the required module
            try:
                from browser_use.llm import ChatAnthropic
            except ImportError as e:
                raise ImportError(
                    f"Failed to import ChatAnthropic for Anthropic provider. "
                    f"Please ensure browser_use[anthropic] is installed. "
                    f"You can install it with: pip install browser_use[anthropic]\n"
                    f"Original error: {str(e)}"
                )

            model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            api_key = get_api_key_secure("anthropic", "ANTHROPIC_API_KEY")

            # Validate API key
            if not api_key or api_key == "YOUR_API_KEY":
                raise ConfigurationError(
                    message="ANTHROPIC_API_KEY is not properly configured for Anthropic provider",
                    config_key="ANTHROPIC_API_KEY",
                    expected_format="Valid API key from Anthropic Console",
                    error_context=create_error_context(
                        component="LLM Configuration",
                        operation="api_key_validation",
                        provider="anthropic",
                        metadata={"help_url": "https://console.anthropic.com/account/keys"}
                    )
                )

            # Basic API key validation (removed specific format validation for security)

            try:
                llm_instance = ChatAnthropic(model=model_name, api_key=api_key)
                retry_manager.record_success(provider)
                return llm_instance
            except Exception as e:
                retry_manager.record_failure(provider, e)
                raise LLMAPIException(provider, f"Failed to create Anthropic instance: {str(e)}") from e

        elif provider == "azure":
            # Try to import the required module
            try:
                from browser_use.llm import ChatAzureOpenAI
            except ImportError as e:
                raise ImportError(
                    f"Failed to import ChatAzureOpenAI for Azure provider. "
                    f"Please ensure browser_use[azure] is installed. "
                    f"You can install it with: pip install browser_use[azure]\n"
                    f"Original error: {str(e)}"
                )

            model_name = os.getenv("AZURE_MODEL", "gpt-4o-mini")
            api_key = get_api_key_secure("azure", "AZURE_API_KEY")
            endpoint = os.getenv("AZURE_ENDPOINT")
            deployment = os.getenv("AZURE_DEPLOYMENT")
            api_version = os.getenv("AZURE_API_VERSION", "2024-10-21")

            # Validate required parameters
            if not api_key or api_key == "YOUR_API_KEY":
                raise ConfigurationError(
                    message="AZURE_API_KEY is not properly configured for Azure provider",
                    config_key="AZURE_API_KEY",
                    expected_format="Valid API key from Azure Portal",
                    error_context=create_error_context(
                        component="LLM Configuration",
                        operation="api_key_validation",
                        provider="azure",
                        metadata={"help_url": "https://portal.azure.com"}
                    )
                )

            # Basic API key validation (removed specific format validation for security)

            if not endpoint:
                raise ConfigurationError(
                    message="AZURE_ENDPOINT is required for Azure provider",
                    config_key="AZURE_ENDPOINT",
                    expected_format="https://your-resource.openai.azure.com/",
                    error_context=create_error_context(
                        component="LLM Configuration",
                        operation="endpoint_validation",
                        provider="azure"
                    )
                )

            # Azure can use either deployment name or model name
            if not deployment and not model_name:
                raise ValueError(
                    "Either AZURE_DEPLOYMENT or AZURE_MODEL must be set for Azure provider. "
                    "Please configure your deployment name or model name."
                )

            try:
                llm_instance = ChatAzureOpenAI(
                    model=model_name,
                    api_key=api_key,
                    azure_endpoint=endpoint,
                    azure_deployment=deployment,
                    api_version=api_version
                )
                retry_manager.record_success(provider)
                return llm_instance
            except Exception as e:
                retry_manager.record_failure(provider, e)
                raise LLMAPIException(provider, f"Failed to create Azure OpenAI instance: {str(e)}") from e

        elif provider == "groq":
            # Try to import the required module
            try:
                from browser_use.llm import ChatGroq
            except ImportError as e:
                raise ImportError(
                    f"Failed to import ChatGroq for Groq provider. "
                    f"Please ensure browser_use[groq] is installed. "
                    f"You can install it with: pip install browser_use[groq]\n"
                    f"Original error: {str(e)}"
                )

            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            api_key = get_api_key_secure("groq", "GROQ_API_KEY")

            # Validate API key
            if not api_key or api_key == "YOUR_API_KEY":
                raise ConfigurationError(
                    message="GROQ_API_KEY is not properly configured for Groq provider",
                    config_key="GROQ_API_KEY",
                    expected_format="Valid API key from Groq Console",
                    error_context=create_error_context(
                        component="LLM Configuration",
                        operation="api_key_validation",
                        provider="groq",
                        metadata={"help_url": "https://console.groq.com/keys"}
                    )
                )

            # Basic API key validation (removed specific format validation for security)

            try:
                llm_instance = ChatGroq(model=model_name, api_key=api_key)
                retry_manager.record_success(provider)
                return llm_instance
            except Exception as e:
                retry_manager.record_failure(provider, e)
                raise LLMAPIException(provider, f"Failed to create Groq instance: {str(e)}") from e

    except Exception as e:
        # If it's already one of our custom framework errors, re-raise it
        if isinstance(e, (ConfigurationError, LLMProviderError)):
            raise
        # If it's a standard error type, convert to framework error
        elif isinstance(e, (ValueError, ImportError, EnvironmentError)):
            raise ConfigurationError(
                message=f"Failed to initialize {provider} LLM provider: {str(e)}",
                config_key=f"{provider.upper()}_CONFIGURATION",
                error_context=create_error_context(
                    component="LLM Configuration",
                    operation="provider_initialization",
                    provider=provider
                ),
                cause=e
            )
        # Otherwise, wrap it with more context
        else:
            raise ConfigurationError(
                message=f"Unexpected error initializing {provider} LLM provider: {str(e)}",
                error_context=create_error_context(
                    component="LLM Configuration",
                    operation="provider_initialization",
                    provider=provider
                ),
                cause=e
            )


@pytest.fixture(scope="session")
def browser_version_info(browser_profile: BrowserProfile) -> Dict[str, str]:
    # Fixture to get Playwright and browser version info
    try:
        playwright_version = version("playwright")
        with sync_playwright() as p:
            browser_type_name = (
                browser_profile.channel.value if browser_profile.channel else "chromium"
            )
            browser = p[browser_type_name].launch()
            browser_version = browser.version
            browser.close()
        return {
            "playwright_version": playwright_version,
            "browser_version": browser_version,
        }
    except Exception as e:
        logging.warning(f"Could not determine Playwright/browser version: {e}")
        return {
            "playwright_version": "N/A",
            "browser_version": "N/A",
        }


@pytest.fixture(scope="session", autouse=True)
def environment_reporter(
    request: pytest.FixtureRequest,
    llm,
    browser_profile: BrowserProfile,
    browser_version_info: Dict[str, str],
):
    # Fixture to write environment details to a properties file for reporting
    # This runs once per session and is automatically used
    # By default, this creates environment.properties for Allure
    allure_dir = request.config.getoption("--alluredir")
    if not allure_dir or not isinstance(allure_dir, str):
        return

    ENVIRONMENT_PROPERTIES_FILENAME = "environment.properties"
    properties_file = os.path.join(allure_dir, ENVIRONMENT_PROPERTIES_FILENAME)

    # Ensure the directory exists, with permission handling
    try:
        os.makedirs(allure_dir, exist_ok=True)
    except PermissionError:
        logging.error(f"Permission denied to create report directory: {allure_dir}")
        return  # Exit if we can't create the directory

    # Get LLM provider and model name dynamically
    llm_provider = getattr(llm, 'provider', 'unknown')
    # Map 'google' to 'gemini' for consistency with LLM_PROVIDER env var
    if llm_provider == 'google':
        llm_provider = 'gemini'
    llm_model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))

    env_props = {
        "operating_system": f"{platform.system()} {platform.release()}",
        "python_version": sys.version.split(" ")[0],
        "browser_use_version": get_browser_use_version(),
        "playwright_version": browser_version_info["playwright_version"],
        "browser_type": (
            browser_profile.channel.value if browser_profile.channel else "chromium"
        ),
        "browser_version": browser_version_info["browser_version"],
        "headless_mode": str(browser_profile.headless),
        "llm_provider": llm_provider,
        "llm_model": llm_model_name,
    }

    try:
        with open(properties_file, "w") as f:
            for key, value in env_props.items():
                f.write(f"{key}={value}\n")
    except IOError as e:
        logging.error(f"Failed to write environment properties file: {e}")


@pytest.fixture(scope="session")
def llm():
    # Session-scoped fixture to initialize the language model using the factory function
    # This fixture will fail early with clear error messages if there are configuration issues
    try:
        return create_llm_instance()
    except (ConfigurationError, LLMProviderError) as e:
        # Log the error with structured logging
        correlation_id = log_error_with_context(e, e.error_context, level="error")
        # Re-raise to fail the test session with clear error
        raise pytest.UsageError(
            f"\n\nLLM Configuration Error [correlation_id: {correlation_id}]:\n{e.get_actionable_message()}\n\n"
            "Please check your environment configuration and try again.\n"
        )
    except (ValueError, ImportError, EnvironmentError) as e:
        # Handle legacy exceptions for backward compatibility
        context = create_error_context(
            component="LLM Fixture",
            operation="llm_initialization"
        )
        correlation_id = log_error_with_context(e, context, level="error")
        raise pytest.UsageError(
            f"\n\nLLM Configuration Error [correlation_id: {correlation_id}]:\n{str(e)}\n\n"
            "Please check your environment configuration and try again.\n"
        )


@pytest.fixture(scope="session")
def browser_profile() -> BrowserProfile:
    # Session-scoped fixture for browser profile configuration
    headless_mode = os.getenv("HEADLESS", "True").lower() in ("true", "1", "t")
    return BrowserProfile(headless=headless_mode)


@pytest.fixture(scope="function")
async def browser_session(
    browser_profile: BrowserProfile,
) -> AsyncGenerator[BrowserSession, None]:
    # Function-scoped fixture to manage the browser session's lifecycle
    session = None
    try:
        session = BrowserSession(browser_profile=browser_profile)
        yield session
    except Exception as e:
        # Convert browser session errors to framework exceptions
        context = create_error_context(
            component="Browser Session",
            operation="session_creation",
            metadata={"browser_type": browser_profile.channel.value if browser_profile.channel else "chromium"}
        )
        browser_error = BrowserSessionError(
            message=f"Failed to create browser session: {str(e)}",
            browser_type=browser_profile.channel.value if browser_profile.channel else "chromium",
            error_context=context,
            cause=e
        )
        log_error_with_context(browser_error, context)
        raise browser_error
    finally:
        if session:
            try:
                await session.close()
            except Exception as e:
                # Log session cleanup errors but don't fail the test
                context = create_error_context(
                    component="Browser Session",
                    operation="session_cleanup"
                )
                log_error_with_context(e, context, level="warning")


# --- Base Test Class for Agent-based Tests ---


class BaseAgentTest:
    # Base class for agent-based tests to reduce boilerplate

    BASE_URL = "https://discuss.google.dev/"

    def setup_method(self):
        # Initialize fresh semantic validation components for each test method
        from validation import ValidationRegistry, CustomAssertions
        # Create fresh instances to avoid state contamination between tests
        self.validation_registry = ValidationRegistry()
        self.custom_assertions = CustomAssertions(default_confidence_threshold=0.8)

    def teardown_method(self):
        # Clean up validation components after each test
        if hasattr(self, 'validation_registry'):
            # Clear any cached validation results
            if hasattr(self.validation_registry, 'cache'):
                self.validation_registry.cache.clear()
        
        # Reset validation components
        self.validation_registry = None
        self.custom_assertions = None

    def _ensure_validation_components(self):
        # Lazy initialization of validation components if not already set up
        if not hasattr(self, 'validation_registry') or self.validation_registry is None:
            from validation import ValidationRegistry, CustomAssertions
            # Create fresh instances to avoid state contamination
            self.validation_registry = ValidationRegistry()
            self.custom_assertions = CustomAssertions(default_confidence_threshold=0.8)

    async def validate_task(
        self,
        llm,
        browser_session: BrowserSession,
        task_instruction: str,
        expected_outcomes: Optional[Union[str, List[str]]] = None,
        confidence_threshold: float = 0.8,
        ignore_case: bool = True,
        enable_partial_matching: bool = True,
        validation_context: Optional[Dict[str, Any]] = None
    ) -> str:
        # Enhanced validation using semantic validation framework
        # Replaces simple string matching with confidence-scored semantic validation
        # Args: llm, browser_session, task_instruction, expected_outcomes, confidence_threshold, ignore_case, enable_partial_matching, validation_context
        # Returns: The final text result from the agent after semantic validation
        
        # Ensure validation components are initialized
        self._ensure_validation_components()
        
        from validation import ValidationContext, ValidationType
        
        full_task = f"Go to {self.BASE_URL}, then {task_instruction}"

        result_text = await run_agent_task(full_task, llm, browser_session)

        if result_text is None:
            raise ValidationError(
                message="Agent did not return a final result",
                validation_type="result_presence",
                expected_value="non-null result",
                actual_value="null",
                error_context=create_error_context(
                    component="Semantic Agent Validation",
                    operation="result_validation"
                )
            )

        # If no expected outcomes provided, return result without validation
        if not expected_outcomes:
            return result_text

        # Convert single string to list for consistent processing
        if isinstance(expected_outcomes, str):
            expected_outcomes = [expected_outcomes]

        # Create validation context
        context = ValidationContext(
            validation_type=ValidationType.SEMANTIC,
            metadata={
                "task_instruction": task_instruction,
                "base_url": self.BASE_URL,
                "validation_method": "semantic_framework",
                "ignore_case": ignore_case,
                "enable_partial_matching": enable_partial_matching,
                **(validation_context or {})
            }
        )

        try:
            # Use semantic validator for confidence scoring
            validation_result = await self.validation_registry.validate_semantic(
                expected=expected_outcomes,
                actual=result_text,
                context=context,
                confidence_threshold=confidence_threshold,
                ignore_case=ignore_case,
                enable_partial_matching=enable_partial_matching
            )

            if not validation_result.is_successful():
                raise ValidationError(
                    message=f"Semantic validation failed: {validation_result.message}",
                    expected_value=expected_outcomes,
                    actual_value=result_text,
                    validation_type="semantic_framework",
                    confidence_score=validation_result.confidence_score.value,
                    error_context=create_error_context(
                        component="Semantic Agent Validation",
                        operation="semantic_validation",
                        metadata={
                            "validation_id": validation_result.context.validation_id,
                            "confidence_components": validation_result.confidence_score.components,
                            "confidence_threshold": confidence_threshold
                        }
                    )
                )

            logging.info(
                f"Semantic validation passed with confidence {validation_result.confidence_score.value:.3f} "
                f"(threshold: {confidence_threshold:.3f}) [{context.validation_id}]"
            )

            return result_text

        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ValidationError(
                    message=f"Semantic validation error: {str(e)}",
                    expected_value=expected_outcomes,
                    actual_value=result_text,
                    validation_type="semantic_framework",
                    error_context=create_error_context(
                        component="Semantic Agent Validation",
                        operation="semantic_validation_error",
                        metadata={"original_error": str(e)}
                    )
                ) from e
            raise

    # Legacy method for backward compatibility - DEPRECATED
    async def validate_task_legacy(
        self,
        llm,
        browser_session: BrowserSession,
        task_instruction: str,
        expected_substring: Optional[str] = None,
        ignore_case: bool = False,
    ) -> str:
        """
        DEPRECATED: Legacy string matching validation method.
        
        This method is deprecated and provided only for backward compatibility.
        Use validate_task() with semantic validation instead.
        
        Will be removed in a future version.
        """
        import warnings
        warnings.warn(
            "validate_task_legacy() is deprecated. Use validate_task() with semantic validation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Convert to new semantic validation approach
        expected_outcomes = [expected_substring] if expected_substring else None
        return await self.validate_task(
            llm, browser_session, task_instruction,
            expected_outcomes=expected_outcomes,
            confidence_threshold=0.8,
            ignore_case=ignore_case,
            enable_partial_matching=True
        )


# --- Allure Hook for Step-by-Step Reporting ---


async def record_step(agent: Agent):
    # Hook function that captures and records agent activity at each step
    history = agent.state.history
    if not history:
        return

    last_action = history.model_actions()[-1] if history.model_actions() else {}
    action_name = next(iter(last_action)) if last_action else "No action"
    action_params = last_action.get(action_name, {})

    step_title = f"Action: {action_name}"
    if action_params:
        param_str = ", ".join(f"{k}={v}" for k, v in action_params.items())
        step_title += f"({param_str})"

    with allure.step(step_title):
        # Attach Agent Thoughts
        thoughts = history.model_thoughts()
        if thoughts:
            allure.attach(
                str(thoughts[-1]),
                name="Agent Thoughts",
                attachment_type=allure.attachment_type.TEXT,
            )

        # Attach URL
        url = history.urls()[-1] if history.urls() else "N/A"
        allure.attach(
            url,
            name="URL",
            attachment_type=allure.attachment_type.URI_LIST,
        )

        # Attach Step Duration
        last_history_item = history.history[-1] if history.history else None
        if last_history_item and last_history_item.metadata:
            duration = last_history_item.metadata.duration_seconds
            allure.attach(
                f"{duration:.2f}s",
                name="Step Duration",
                attachment_type=allure.attachment_type.TEXT,
            )

        # Attach Screenshot
        if agent.browser_session:
            try:
                screenshot_b64 = await agent.browser_session.take_screenshot()
                if screenshot_b64:
                    screenshot_bytes = base64.b64decode(screenshot_b64)
                    allure.attach(
                        screenshot_bytes,
                        name="Screenshot after Action",
                        attachment_type=allure.attachment_type.PNG,
                    )
            except Exception as e:
                logging.warning(f"Failed to take or attach screenshot: {e}")


# --- Helper Function to Run Agent ---


async def _run_agent_task_impl(
    task_description: str,
    llm,
    browser_session: BrowserSession,
) -> Optional[str]:
    # Internal implementation - Initializes and runs the browser agent for a given task using an active browser session
    # Implements retry logic with correlation ID tracking for better debugging and monitoring
    correlation_id = get_correlation_id()

    logging.info(f"Running task: {task_description} [correlation_id: {correlation_id}]")

    # Determine provider for retry logic
    provider_name = getattr(llm, 'provider', 'unknown')
    if provider_name == 'google':
        provider_name = 'gemini'  # Map google to gemini for consistency

    try:
        provider_enum = LLMProvider(provider_name)
    except ValueError:
        logging.warning(f"Unknown provider {provider_name}, using default retry behavior")
        provider_enum = LLMProvider.GEMINI  # Default fallback

    # Wrap agent execution with retry logic
    @with_llm_retry(provider_enum, correlation_id)
    async def execute_agent():
        try:
            agent = Agent(
                task=task_description,
                llm=llm,
                browser_session=browser_session,
                name=f"Agent for '{task_description[:50]}...'",
            )

            result = await agent.run(on_step_end=record_step)

            if not result or not result.final_result():
                raise LLMAPIException(provider_name, "Agent returned empty or invalid result")

            return result
        except Exception as e:
            # Convert browser/agent exceptions to retryable LLM exceptions for certain cases
            if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                context = create_error_context(
                    correlation_id=correlation_id,
                    component="Agent Execution",
                    operation="agent_run",
                    provider=provider_name
                )
                llm_error = LLMProviderError(
                    message=f"Agent execution failed: {str(e)}",
                    provider=provider_name,
                    error_context=context,
                    cause=e
                )
                raise llm_error from e
            # Re-raise non-retryable exceptions as-is
            raise

    try:
        result = await execute_agent()

        final_text = result.final_result()
        # Sanitize final text for Allure report
        from core.security import sanitize_for_allure
        safe_final_text = sanitize_for_allure(final_text)
        
        allure.attach(
            safe_final_text,
            name="Agent Final Output",
            attachment_type=allure.attachment_type.TEXT,
        )

        # Attach correlation ID for tracking
        allure.attach(
            correlation_id,
            name="Correlation ID",
            attachment_type=allure.attachment_type.TEXT,
        )

        logging.info(f"Task finished successfully [correlation_id: {correlation_id}]")
        return final_text

    except CircuitBreakerOpenException as e:
        error_msg = f"Circuit breaker is open for provider {provider_name}. Task cannot be executed."
        context = create_error_context(
            correlation_id=correlation_id,
            component="Agent Execution",
            operation="circuit_breaker_check",
            provider=provider_name
        )

        log_error_with_context(e, context, level="error")

        from core.security import sanitize_for_allure
        safe_message = sanitize_for_allure(e.get_actionable_message())
        allure.attach(
            safe_message,
            name="Circuit Breaker Error",
            attachment_type=allure.attachment_type.TEXT,
        )

        raise RuntimeError(error_msg) from e

    except (LLMAPIException, LLMProviderError) as e:
        error_msg = f"LLM API error after retries: {str(e)}"

        # Ensure we have error context
        if hasattr(e, 'error_context') and e.error_context:
            context = e.error_context
            context.correlation_id = correlation_id  # Ensure correlation ID is set
        else:
            context = create_error_context(
                correlation_id=correlation_id,
                component="Agent Execution",
                operation="llm_api_call",
                provider=provider_name
            )

        log_error_with_context(e, context, level="error")

        # Attach comprehensive error details to Allure report with sanitization
        from core.security import sanitize_for_allure
        error_details = e.get_actionable_message() if hasattr(e, 'get_actionable_message') else str(e)
        safe_error_details = sanitize_for_allure(f"{error_details}\nCorrelation ID: {correlation_id}")
        allure.attach(
            safe_error_details,
            name="LLM API Error Details",
            attachment_type=allure.attachment_type.TEXT,
        )

        raise RuntimeError(error_msg) from e


async def run_agent_task(
    task_description: str,
    llm,
    browser_session: BrowserSession,
) -> Optional[str]:
    """
    Secure wrapper for agent task execution that prevents sensitive data leakage in Allure reports.
    
    This function sanitizes the task description before creating Allure steps to ensure
    no API keys, credentials, or other sensitive data are exposed in test reports.
    """
    # Sanitize task description to prevent sensitive data from appearing in Allure
    from core.security import sanitize_for_allure
    safe_task_description = sanitize_for_allure(task_description)
    
    # Use Allure step with sanitized description - no function parameters are captured
    with allure.step(f"Running browser agent with task: {safe_task_description}"):
        # Call the actual implementation
        return await _run_agent_task_impl(task_description, llm, browser_session)
