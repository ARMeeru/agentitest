import allure
import pytest
from conftest import BaseAgentTest


@allure.feature("SauceDemo Basic Flow")
class TestSauceDemoBasic(BaseAgentTest):
    # Basic SauceDemo functionality test to showcase the framework.

    BASE_URL = "https://www.saucedemo.com/"

    @allure.story("User Login")
    @allure.title("Test successful login with standard user")
    @pytest.mark.asyncio
    async def test_standard_user_login(self, llm, browser_session):
        # Test login with standard_user.
        task = "enter 'standard_user' in the username field, enter 'secret_sauce' in the password field, click the LOGIN button, and confirm you reach the products page with items displayed. Return 'login_successful' when products are visible."
        await self.validate_task(llm, browser_session, task, "login_successful", ignore_case=True)

    @allure.story("Add to Cart")
    @allure.title("Test adding item to shopping cart")
    @pytest.mark.asyncio
    async def test_add_to_cart(self, llm, browser_session):
        # Test adding a product to cart after login.
        # First login
        login_task = "enter 'standard_user' in the username field, enter 'secret_sauce' in the password field, and click LOGIN"
        await self.validate_task(llm, browser_session, login_task, "inventory", ignore_case=True)

        # Then add to cart
        cart_task = "find the first product and click its 'Add to cart' button, then verify the cart badge shows '1'. Return 'item_added' if successful."
        await self.validate_task(llm, browser_session, cart_task, "item_added", ignore_case=True)

    @allure.story("Complete Purchase")
    @allure.title("Test end-to-end purchase flow")
    @pytest.mark.asyncio
    async def test_complete_purchase(self, llm, browser_session):
        # Test complete purchase flow from login to checkout completion.
        task = """
        Complete this full purchase flow:
        1. Login with username 'standard_user' and password 'secret_sauce'
        2. Add any product to cart
        3. Click the cart icon and proceed to checkout
        4. Fill checkout form: First Name 'John', Last Name 'Doe', Postal Code '12345'
        5. Complete the purchase through to the final confirmation page
        Return 'purchase_completed' when you see the order confirmation or thank you message.
        """
        await self.validate_task(llm, browser_session, task, "purchase_completed", ignore_case=True)
