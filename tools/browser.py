"""
Browser interaction tools.

This module provides tools for interacting with web browsers and web pages.
"""

from tools.base import Tool


class BrowserTool(Tool):
    """
    A tool that allows controlling a web browser for web interactions.

    This tool enables language models to interact with web pages by performing
    actions like navigating to URLs, clicking elements, typing text, and retrieving
    content from web pages using Playwright.
    """

    def __init__(self):
        """
        Initialize the BrowserTool with its name, description, and parameter schema.

        Sets up the tool with a detailed parameter schema that defines different
        actions that can be performed (navigate, click, type, get_text, quit) and
        their respective required parameters.
        """
        super().__init__(
            name="browser_control",
            description="Control a web browser to navigate and interact with web pages",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform: 'navigate', 'click', 'type', 'quit'",
                        "enum": ["navigate", "click", "type", "get_text", "quit"],
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to (for 'navigate' action)",
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the target element (for 'click', 'type', 'get_text' actions)",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type (for 'type' action)",
                    },
                },
                "required": ["action"],
            },
        )
        self.browser = None
        self.page = None

    def execute(self, **kwargs):
        """
        Executes browser actions using Playwright.

        Supports various browser actions including navigation, clicking elements,
        typing text, retrieving text, and closing the browser.

        Args:
            **kwargs: Keyword arguments including:
                action (str): The action to perform ('navigate', 'click', 'type', 'get_text', 'quit')
                url (str, optional): URL to navigate to (for 'navigate' action)
                selector (str, optional): CSS selector for element (for 'click', 'type', 'get_text' actions)
                text (str, optional): Text to type (for 'type' action)

        Returns:
            str: Result of the browser action or error message
        """
        try:
            # Import here to avoid requiring playwright for the entire module
            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                return "Error: Playwright is not installed. Please install it with 'pip install playwright' and then run 'playwright install'"

            action = kwargs.get("action")

            # Handle quit action separately
            if action == "quit":
                if self.browser:
                    self.browser.close()
                    if hasattr(self, "playwright"):
                        self.playwright.stop()
                    self.browser = None
                    self.page = None
                return "Browser closed successfully"

            # Initialize browser if not already done
            if self.browser is None or self.page is None:
                self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(headless=True)
                self.page = self.browser.new_page()

            # At this point, self.page should never be None
            if self.page is None:
                return "Error: Failed to initialize browser page"

            if action == "navigate":
                url = kwargs.get("url")
                if not url:
                    return "Error: URL is required for navigate action"
                self.page.goto(url)
                page_text = self.page.inner_text("body")
                return f"Navigated to {url}. Page content: {page_text[:500]}..."

            elif action in ["click", "type", "get_text"]:
                selector = kwargs.get("selector")
                if not selector:
                    return "Error: Selector is required for element actions"

                try:
                    # Wait for element to be present
                    self.page.wait_for_selector(
                        selector, state="visible", timeout=10000
                    )

                    if action == "click":
                        self.page.click(selector)
                        return f"Clicked element with selector: {selector}"
                    elif action == "type":
                        text = kwargs.get("text")
                        if not text:
                            return "Error: Text is required for type action"
                        self.page.fill(selector, text)
                        return f"Typed '{text}' into element with selector: {selector}"
                    elif action == "get_text":
                        elements = self.page.query_selector_all(selector)
                        texts = [element.inner_text() for element in elements]
                        return f"Found {len(texts)} elements. Text content: {texts}"
                except Exception as e:
                    return f"Error interacting with element: {str(e)}"

            return f"Error: Invalid action specified: {action}"

        except Exception as e:
            return f"Error: {str(e)}"
