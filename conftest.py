import base64
import logging
import os
import platform
import sys
from importlib.metadata import version
from typing import AsyncGenerator, Dict, Optional

import allure
import pytest
from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.utils import get_browser_use_version
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

# Load environment variables from .env file
load_dotenv()


def create_llm_instance():
    """
    Factory function to create the appropriate LLM instance based on environment configuration.

    Reads the LLM_PROVIDER environment variable to determine which provider to use.
    Supported values: "gemini", "openai", "anthropic", "azure", "groq"

    Returns:
        An instance of the appropriate LLM provider

    Raises:
        ValueError: If LLM_PROVIDER contains an unsupported value or required API keys are missing
        ImportError: If the required provider module is not available
        EnvironmentError: If configuration issues are detected
    """
    # Get the provider from environment, validate it's not empty
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    if not provider:
        raise ValueError(
            "LLM_PROVIDER environment variable is empty. "
            "Please set it to one of: 'gemini', 'openai', 'anthropic', 'azure', 'groq'"
        )

    # Define supported providers for better error messages
    supported_providers = ["gemini", "openai", "anthropic", "azure", "groq"]

    if provider not in supported_providers:
        raise ValueError(
            f"Invalid LLM_PROVIDER: '{provider}'. "
            f"Supported values are: {', '.join(supported_providers)}. "
            "Please check your .env file or environment variables."
        )

    try:
        if provider == "gemini":
            # Try to import the required module
            try:
                from browser_use.llm import ChatGoogle
            except ImportError as e:
                raise ImportError(
                    f"Failed to import ChatGoogle for Gemini provider. "
                    f"Please ensure browser_use[gemini] is installed. "
                    f"You can install it with: pip install browser_use[gemini]\n"
                    f"Original error: {str(e)}"
                )

            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
            api_key = os.getenv("GOOGLE_API_KEY")

            # Validate API key
            if not api_key or api_key == "YOUR_API_KEY":
                raise ValueError(
                    "GOOGLE_API_KEY is not properly configured for Gemini provider. "
                    "Please set a valid API key in your .env file or environment variables. "
                    "You can get an API key from: https://makersuite.google.com/app/apikey"
                )

            # Validate API key format (Gemini keys are typically 39-40 characters)
            if len(api_key) < 30 or len(api_key) > 50:
                raise ValueError(
                    "GOOGLE_API_KEY appears to be invalid (incorrect length). "
                    "Gemini API keys are typically 39-40 characters long. "
                    "Please check your API key configuration."
                )

            return ChatGoogle(model=model_name, api_key=api_key)

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
            api_key = os.getenv("OPENAI_API_KEY")

            # Validate API key
            if not api_key or api_key == "YOUR_API_KEY":
                raise ValueError(
                    "OPENAI_API_KEY is not properly configured for OpenAI provider. "
                    "Please set a valid API key in your .env file or environment variables. "
                    "You can get an API key from: https://platform.openai.com/api-keys"
                )

            # Validate API key format (OpenAI keys start with 'sk-' and are ~51 characters)
            if not api_key.startswith("sk-"):
                raise ValueError(
                    "OPENAI_API_KEY appears to be invalid. "
                    "OpenAI API keys must start with 'sk-'. "
                    "Please check your API key configuration."
                )
            
            if len(api_key) < 45 or len(api_key) > 60:
                raise ValueError(
                    "OPENAI_API_KEY appears to be invalid (incorrect length). "
                    "OpenAI API keys are typically around 51 characters long. "
                    "Please check your API key configuration."
                )

            return ChatOpenAI(model=model_name, api_key=api_key)

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
            api_key = os.getenv("ANTHROPIC_API_KEY")

            # Validate API key
            if not api_key or api_key == "YOUR_API_KEY":
                raise ValueError(
                    "ANTHROPIC_API_KEY is not properly configured for Anthropic provider. "
                    "Please set a valid API key in your .env file or environment variables. "
                    "You can get an API key from: https://console.anthropic.com/account/keys"
                )

            # Validate API key format (Anthropic keys start with 'sk-ant-' and are ~108 characters)
            if not api_key.startswith("sk-ant-"):
                raise ValueError(
                    "ANTHROPIC_API_KEY appears to be invalid. "
                    "Anthropic API keys must start with 'sk-ant-'. "
                    "Please check your API key configuration."
                )
            
            if len(api_key) < 100 or len(api_key) > 120:
                raise ValueError(
                    "ANTHROPIC_API_KEY appears to be invalid (incorrect length). "
                    "Anthropic API keys are typically around 108 characters long. "
                    "Please check your API key configuration."
                )

            return ChatAnthropic(model=model_name, api_key=api_key)

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
            api_key = os.getenv("AZURE_API_KEY")
            endpoint = os.getenv("AZURE_ENDPOINT")
            deployment = os.getenv("AZURE_DEPLOYMENT")
            api_version = os.getenv("AZURE_API_VERSION", "2024-10-21")

            # Validate required parameters
            if not api_key or api_key == "YOUR_API_KEY":
                raise ValueError(
                    "AZURE_API_KEY is not properly configured for Azure provider. "
                    "Please set a valid API key in your .env file or environment variables. "
                    "You can get an API key from your Azure portal."
                )
            
            # Validate API key format (Azure keys are typically 32 hex characters)
            if len(api_key) < 30 or len(api_key) > 40:
                raise ValueError(
                    "AZURE_API_KEY appears to be invalid (incorrect length). "
                    "Azure API keys are typically 32 characters long. "
                    "Please check your API key configuration."
                )

            if not endpoint:
                raise ValueError(
                    "AZURE_ENDPOINT is required for Azure provider. "
                    "Please set your Azure OpenAI endpoint URL in your .env file. "
                    "Example: https://your-resource.openai.azure.com/"
                )

            # Azure can use either deployment name or model name
            if not deployment and not model_name:
                raise ValueError(
                    "Either AZURE_DEPLOYMENT or AZURE_MODEL must be set for Azure provider. "
                    "Please configure your deployment name or model name."
                )

            return ChatAzureOpenAI(
                model=model_name,
                api_key=api_key,
                azure_endpoint=endpoint,
                azure_deployment=deployment,
                api_version=api_version
            )

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
            api_key = os.getenv("GROQ_API_KEY")

            # Validate API key
            if not api_key or api_key == "YOUR_API_KEY":
                raise ValueError(
                    "GROQ_API_KEY is not properly configured for Groq provider. "
                    "Please set a valid API key in your .env file or environment variables. "
                    "You can get an API key from: https://console.groq.com/keys"
                )

            # Validate API key format (Groq keys start with 'gsk_' and are ~56 characters)
            if not api_key.startswith("gsk_"):
                raise ValueError(
                    "GROQ_API_KEY appears to be invalid. "
                    "Groq API keys must start with 'gsk_'. "
                    "Please check your API key configuration."
                )
            
            if len(api_key) < 50 or len(api_key) > 65:
                raise ValueError(
                    "GROQ_API_KEY appears to be invalid (incorrect length). "
                    "Groq API keys are typically around 56 characters long. "
                    "Please check your API key configuration."
                )

            return ChatGroq(model=model_name, api_key=api_key)

    except Exception as e:
        # If it's already one of our custom errors, re-raise it
        if isinstance(e, (ValueError, ImportError, EnvironmentError)):
            raise
        # Otherwise, wrap it with more context
        raise EnvironmentError(
            f"Failed to initialize {provider} LLM provider. "
            f"Error: {str(e)}"
        )


@pytest.fixture(scope="session")
def browser_version_info(browser_profile: BrowserProfile) -> Dict[str, str]:
    """
    Fixture to get Playwright and browser version info.
    """
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
    """
    Fixture to write environment details to a properties file for reporting.
    This runs once per session and is automatically used.
    By default, this creates `environment.properties` for Allure.
    """
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
    """Session-scoped fixture to initialize the language model using the factory function.

    This fixture will fail early with clear error messages if there are configuration issues.
    """
    try:
        return create_llm_instance()
    except (ValueError, ImportError, EnvironmentError) as e:
        # Log the error for better visibility
        logging.error(f"Failed to initialize LLM: {str(e)}")
        # Re-raise to fail the test session with clear error
        raise pytest.UsageError(
            f"\n\nLLM Configuration Error:\n{str(e)}\n\n"
            "Please check your environment configuration and try again.\n"
        )


@pytest.fixture(scope="session")
def browser_profile() -> BrowserProfile:
    """Session-scoped fixture for browser profile configuration."""
    headless_mode = os.getenv("HEADLESS", "True").lower() in ("true", "1", "t")
    return BrowserProfile(headless=headless_mode)


@pytest.fixture(scope="function")
async def browser_session(
    browser_profile: BrowserProfile,
) -> AsyncGenerator[BrowserSession, None]:
    """Function-scoped fixture to manage the browser session's lifecycle."""
    session = BrowserSession(browser_profile=browser_profile)
    yield session
    await session.close()


# --- Base Test Class for Agent-based Tests ---


class BaseAgentTest:
    """Base class for agent-based tests to reduce boilerplate."""

    BASE_URL = "https://www.googlecloudcommunity.com/"

    async def validate_task(
        self,
        llm,
        browser_session: BrowserSession,
        task_instruction: str,
        expected_substring: Optional[str] = None,
        ignore_case: bool = False,
    ) -> str:
        """
        Runs a task with the agent, prepends the BASE_URL, and performs common assertions.

        Args:
            llm: The language model instance.
            browser_session: The browser session instance.
            task_instruction: The specific instruction for the agent, without the "Go to URL" part.
            expected_substring: An optional string to assert is present in the agent's result.
            ignore_case: If True, the substring check will be case-insensitive.

        Returns:
            The final text result from the agent for any further custom assertions.
        """
        full_task = f"Go to {self.BASE_URL}, then {task_instruction}"

        result_text = await run_agent_task(full_task, llm, browser_session)

        assert result_text is not None, "Agent did not return a final result."

        if expected_substring:
            result_to_check = result_text.lower() if ignore_case else result_text
            substring_to_check = (
                expected_substring.lower() if ignore_case else expected_substring
            )
            assert (
                substring_to_check in result_to_check
            ), f"Assertion failed: Expected '{expected_substring}' not found in agent result: '{result_text}'"

        return result_text


# --- Allure Hook for Step-by-Step Reporting ---


async def record_step(agent: Agent):
    """Hook function that captures and records agent activity at each step."""
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


@allure.step("Running browser agent with task: {task_description}")
async def run_agent_task(
    task_description: str,
    llm,
    browser_session: BrowserSession,
) -> Optional[str]:
    """Initializes and runs the browser agent for a given task using an active browser session."""
    logging.info(f"Running task: {task_description}")

    agent = Agent(
        task=task_description,
        llm=llm,
        browser_session=browser_session,
        name=f"Agent for '{task_description[:50]}...'",
    )

    result = await agent.run(on_step_end=record_step)

    final_text = result.final_result()
    allure.attach(
        final_text,
        name="Agent Final Output",
        attachment_type=allure.attachment_type.TEXT,
    )

    logging.info("Task finished.")
    return final_text
