import allure
import pytest

from conftest import BaseAgentTest


@allure.feature("Home Page Content")
class TestHomePageStats(BaseAgentTest):
    """Tests the content and navigation of the home page."""

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
            ("Google Cloud Security", "security"),
        ],
    )
    async def test_main_navigation(
        self, llm, browser_session, link_text, expected_path_segment
    ):
        """Tests navigation to main sections of the website."""
        task = f"click on the '{link_text}' link in the main navigation, and then return the final URL of the page."
        result_url = await self.validate_task(
            llm, browser_session, task, expected_path_segment
        )
        assert (
            expected_path_segment in result_url
        ), f"Agent did not navigate to the correct page for {link_text}. URL was: {result_url}"

    @allure.story("Community Statistics")
    @allure.title("Test Forum Statistics Are Not Present")
    @pytest.mark.asyncio
    async def test_forum_loads_successfully(self, llm, browser_session):
        """Tests that the forum loads successfully (replacing legacy stats test)."""
        task = "confirm that the Google Developer Community forum page has loaded successfully and displays community content. Return 'forum_loaded' if the page loads with community categories visible."
        await self.validate_task(
            llm, browser_session, task, "forum_loaded", ignore_case=True
        )


@allure.feature("Search Functionality")
class TestSearch(BaseAgentTest):
    """Tests for the website's search functionality."""

    EXPECTED_NO_RESULTS = "no_results_found"

    @allure.story("Searching for Terms")
    @allure.title("Search for '{term}'")
    @pytest.mark.asyncio
    @pytest.mark.parametrize("term", ["BigQuery", "Vertex AI"])
    async def test_search_for_term(self, llm, browser_session, term):
        """Tests searching for a term and verifying results are shown."""
        task = f"locate the search input field, enter '{term}', submit the search by pressing enter, then confirm that search results for '{term}' are displayed on the page. Return 'search_results_displayed' if results are shown."
        expected_response = "search_results_displayed"
        await self.validate_task(llm, browser_session, task, expected_response, ignore_case=True)

    @allure.story("Searching for Non-Existent Term")
    @allure.title("Search for a Non-Existent Term")
    @pytest.mark.asyncio
    async def test_search_for_non_existent_term(self, llm, browser_session):
        """Tests searching for a term that should not have results."""
        term = "a_very_unlikely_search_term_xyz"
        task = f"find the search bar, type '{term}', press enter, and confirm that a 'no results' message is displayed. Return '{self.EXPECTED_NO_RESULTS}' if it is."
        await self.validate_task(
            llm, browser_session, task, self.EXPECTED_NO_RESULTS, ignore_case=True
        )
