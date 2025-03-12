"""
Chain of Thought (CoT) Agent implementation with tool calling capabilities.

This module implements a Chain of Thought reasoning agent that can use large language
models (LLMs) from various providers (OpenAI, Anthropic, Ollama) to solve problems
step by step. The agent can leverage external tools to perform actions like mathematical
calculations, web browsing, file operations, and shell command execution.

The implementation provides:
1. A base Tool class and several specific tool implementations
2. An abstract LLMProvider class with concrete implementations for different LLM services
3. The CoTAgent class that manages the step-by-step reasoning process
"""

import sys  # System-specific parameters and functions
import dotenv  # For loading environment variables from a .env file
import json  # For JSON serialization/deserialization
import os  # For interacting with the operating system and environment variables
from abc import ABC, abstractmethod  # For creating abstract base classes
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,
)  # OpenAI tool parameter type
from anthropic.types.tool_use_block import ToolUseBlock  # Anthropic tool use block type
from anthropic.types.text_block import TextBlock  # Anthropic text block type
from anthropic.types.tool_param import ToolParam  # Anthropic tool parameter type

dotenv.load_dotenv()


# Base Tool class for defining external tools
class Tool:
    """
    Base class for defining external tools that can be called by language models.

    This class provides a common interface for all tools, including their definition
    and execution. Subclasses must implement the execute method to define the tool's
    specific behavior.

    Tools enable language models to perform actions beyond text generation, such as
    mathematical calculations, web browsing, file operations, and more.
    """

    def __init__(self, name, description, parameters):
        """
        Initialize a Tool with its metadata.

        Args:
            name (str): The name of the tool, used to identify it in tool calls.
            description (str): A description of what the tool does, provided to the LLM.
            parameters (dict): JSON Schema object defining the input parameters for the tool.
        """
        self.name = name
        self.description = description
        self.parameters = parameters

    @property
    def definition(self):
        """
        Returns the tool definition in the format expected by LLM providers.

        This property is used when passing tools to LLMs, following the function
        calling format popularized by OpenAI.

        Returns:
            dict: A dictionary containing the tool definition.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs):
        """
        Executes the tool functionality with the provided arguments.

        This is an abstract method that must be implemented by subclasses
        to define the tool's specific behavior.

        Args:
            **kwargs: Keyword arguments corresponding to the tool's parameters.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")


# Example Tool: Calculator
class CalculatorTool(Tool):
    """
    A tool that performs mathematical calculations by evaluating expressions.

    This tool allows language models to solve mathematical problems by passing
    expressions as strings which are then evaluated using Python's eval function.
    It handles basic arithmetic operations, functions, and mathematical expressions.
    """

    def __init__(self):
        """
        Initialize the CalculatorTool with its name, description, and parameter schema.

        Sets up the tool with a name, description, and a parameter schema defining
        that it accepts a mathematical expression as a string.
        """
        super().__init__(
            name="calculator",
            description="Performs mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": 'The mathematical expression to evaluate (e.g., "2 + 3")',
                    }
                },
                "required": ["expression"],
            },
        )

    def execute(self, expression):
        """
        Evaluates a mathematical expression and returns the result.

        Uses Python's eval function to evaluate the provided expression string.
        Note that this can be potentially unsafe with arbitrary expressions.

        Args:
            expression (str): A string containing a mathematical expression to evaluate.

        Returns:
            str: The result of the evaluation or an error message.
        """
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"


# Example Tool: Browser Control
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


# Shell Command Execution Tool
class ExecuteShellTool(Tool):
    """
    A tool that executes shell commands and returns their output.

    This tool enables language models to interact with the operating system
    by executing shell commands. It returns the stdout, stderr, and return code
    of the executed command, with a timeout to prevent hanging.

    Note: This tool should be used with caution as it can potentially execute
    dangerous commands with system-wide effects.
    """

    def __init__(self):
        """
        Initialize the ExecuteShellTool with its name, description, and parameter schema.

        Sets up the tool with a parameter schema that requires a command string
        to be executed in the shell.
        """
        super().__init__(
            name="execute_shell",
            description="Execute a shell command and return its output (use with caution)",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                },
                "required": ["command"],
            },
        )

    def execute(self, command, timeout=10):
        """
        Executes a shell command and returns the result.

        Runs the specified command in a shell environment with a timeout
        to prevent hanging processes. Returns a JSON object containing
        the command's success status, return code, stdout, and stderr.

        Args:
            command (str): Shell command to execute
            timeout (int, optional): Maximum execution time in seconds. Defaults to 10.

        Returns:
            str: JSON string containing command execution results or error information
        """
        try:
            import subprocess
            from pathlib import Path

            # Create a process with timeout
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return json.dumps(
                {
                    "success": process.returncode == 0,
                    "return_code": process.returncode,
                    "stdout": process.stdout.strip(),
                    "stderr": process.stderr.strip(),
                }
            )

        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout} seconds"}
        except Exception as e:
            return {"error": str(e)}


# File Reading Tool
class FileReadTool(Tool):
    """
    A tool that reads files from the filesystem and returns their contents.

    This tool allows language models to read files from the local filesystem,
    with options to limit file size, specify encoding, and list directory contents.
    It includes safety checks to prevent reading sensitive system files.
    """

    def __init__(self):
        """
        Initialize the FileReadTool with its name, description, and parameter schema.

        Sets up the tool with a parameter schema that defines file reading options
        including the file path, maximum size to read, encoding, and whether to list
        directory contents instead of reading a file.
        """
        super().__init__(
            name="file_read",
            description="Read files from the filesystem",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "max_size": {
                        "type": "integer",
                        "description": "Maximum number of bytes to read (default: 100KB)",
                        "default": 102400,
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8",
                    },
                    "list_dir": {
                        "type": "boolean",
                        "description": "If true and path is a directory, list its contents instead of reading",
                        "default": False,
                    },
                },
                "required": ["path"],
            },
        )

    def execute(self, path, max_size=102400, encoding="utf-8", list_dir=False):
        """
        Reads a file or lists directory contents.

        Safely reads a file from the filesystem, with size limits and encoding options.
        Can alternatively list the contents of a directory. Includes safety checks
        to prevent reading sensitive system files.

        Args:
            path (str): Path to the file or directory to read
            max_size (int, optional): Maximum file size in bytes to read. Defaults to 102400 (100KB).
            encoding (str, optional): File encoding to use. Defaults to "utf-8".
            list_dir (bool, optional): If True and path is a directory, list its contents. Defaults to False.

        Returns:
            str: JSON string containing file contents or directory listing or error information
        """
        try:
            from pathlib import Path

            file_path = Path(path).expanduser().resolve()

            # Security check - prevent reading sensitive system files
            sensitive_paths = ["/etc/passwd", "/etc/shadow", "/etc/sudoers", "/etc/ssh"]
            for sensitive in sensitive_paths:
                if str(file_path).startswith(sensitive):
                    return json.dumps(
                        {
                            "error": f"Access denied: Cannot read sensitive system file {file_path}"
                        }
                    )

            # Check if path exists
            if not file_path.exists():
                return json.dumps({"error": f"Path does not exist: {file_path}"})

            # Handle directory listing
            if file_path.is_dir():
                if list_dir:
                    dir_contents = []
                    for item in file_path.iterdir():
                        dir_contents.append(
                            {
                                "name": item.name,
                                "type": "directory" if item.is_dir() else "file",
                                "size": item.stat().st_size if item.is_file() else None,
                                "modified": item.stat().st_mtime,
                            }
                        )
                    return json.dumps(
                        {
                            "is_directory": True,
                            "path": str(file_path),
                            "contents": dir_contents,
                        }
                    )
                else:
                    return json.dumps(
                        {
                            "error": f"Path is a directory: {file_path}. Set list_dir=true to list contents."
                        }
                    )

            # Handle file reading
            if not file_path.is_file():
                return json.dumps({"error": f"Path is not a regular file: {file_path}"})

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > max_size:
                return json.dumps(
                    {
                        "error": f"File too large: {file_size} bytes (max: {max_size} bytes)"
                    }
                )

            # Read file content
            content = file_path.read_text(encoding=encoding)

            return json.dumps(
                {"path": str(file_path), "size": file_size, "content": content}
            )

        except UnicodeDecodeError:
            return json.dumps(
                {
                    "error": f"Cannot decode file with encoding {encoding}. The file might be binary."
                }
            )
        except PermissionError:
            return json.dumps({"error": f"Permission denied: Cannot read {path}"})
        except Exception as e:
            return json.dumps({"error": f"Error reading file: {str(e)}"})


# Google Search Tool using SerpAPI
class GoogleSearchTool(Tool):
    """
    A tool that performs Google searches using the SerpAPI service.

    This tool enables language models to search the internet for real-time information
    by querying Google through SerpAPI. It returns search results including organic
    results, featured snippets, and other relevant search features.

    API key should be set in the SERPAPI_KEY environment variable.
    """

    def __init__(self):
        """
        Initialize the GoogleSearchTool with its name, description, and parameter schema.

        Sets up the tool with a parameter schema that defines search options including
        the query, number of results, and optional location and language settings.
        """
        super().__init__(
            name="google_search",
            description="Search Google for information using SerpAPI",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to send to Google",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10, max: 100)",
                        "default": 10,
                    },
                    "location": {
                        "type": "string",
                        "description": "Location for geographically specific results (e.g., 'New York, New York, United States')",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language for results (e.g., 'en' for English)",
                        "default": "en",
                    },
                },
                "required": ["query"],
            },
        )

    def execute(self, query, num_results=10, location=None, language="en"):
        """
        Performs a Google search using SerpAPI and returns the results.

        Makes a request to SerpAPI with the specified search parameters and returns
        a formatted version of the search results, including organic results,
        featured snippets, and other relevant search information.

        Args:
            query (str): The search query to send to Google
            num_results (int, optional): Number of results to return. Defaults to 10.
            location (str, optional): Location for geographically specific results. Defaults to None.
            language (str, optional): Language for results. Defaults to "en".

        Returns:
            str: JSON string containing search results or error information
        """
        try:
            import requests

            # Get API key from environment
            api_key = os.getenv("SERPAPI_KEY")
            if not api_key:
                print(
                    "SERPAPI_KEY environment variable not set. Please set your SerpAPI key."
                )
                return json.dumps(
                    {
                        "error": "SERPAPI_KEY environment variable not set. Please set your SerpAPI key."
                    }
                )

            # Build request parameters
            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "num": min(num_results, 100),  # Cap at 100 results
                "hl": language,
            }

            # Add optional location parameter
            if location:
                params["location"] = location

            # Make request to SerpAPI
            response = requests.get("https://serpapi.com/search", params=params)

            # Check if request was successful
            if response.status_code != 200:
                return json.dumps(
                    {
                        "error": f"SerpAPI request failed with status code {response.status_code}: {response.text}"
                    }
                )

            # Parse JSON response
            search_results = response.json()

            # Extract and format relevant information
            formatted_results = {
                "query": query,
                "organic_results": [],
                "answer_box": None,
                "knowledge_graph": None,
                "related_questions": [],
            }

            # Add organic results
            if "organic_results" in search_results:
                formatted_results["organic_results"] = [
                    {
                        "title": result.get("title"),
                        "link": result.get("link"),
                        "snippet": result.get("snippet"),
                        "position": result.get("position"),
                    }
                    for result in search_results["organic_results"][:num_results]
                ]

            # Add answer box if present
            if "answer_box" in search_results:
                answer_box = search_results["answer_box"]
                formatted_results["answer_box"] = {
                    "title": answer_box.get("title"),
                    "answer": answer_box.get("answer") or answer_box.get("snippet"),
                    "type": answer_box.get("type"),
                }

            # Add knowledge graph if present
            if "knowledge_graph" in search_results:
                kg = search_results["knowledge_graph"]
                formatted_results["knowledge_graph"] = {
                    "title": kg.get("title"),
                    "description": kg.get("description"),
                    "type": kg.get("type"),
                }

            # Add related questions if present
            if "related_questions" in search_results:
                formatted_results["related_questions"] = [
                    {
                        "question": q.get("question"),
                        "answer": q.get("answer"),
                    }
                    for q in search_results["related_questions"]
                ]

            return json.dumps(formatted_results, ensure_ascii=False)

        except ImportError:
            return json.dumps(
                {
                    "error": "Required package 'requests' is not installed. Install it with 'pip install requests'."
                }
            )
        except Exception as e:
            return json.dumps({"error": f"Error performing Google search: {str(e)}"})


# LLM Provider base class - abstraction layer for different LLM services
class LLMProvider(ABC):
    """
    Abstract base class for language model providers (OpenAI, Anthropic, Ollama, etc.).

    Defines a common interface for different language model providers, allowing
    the CoTAgent to work with any supported LLM provider interchangeably.
    Subclasses must implement the chat method to define provider-specific behavior.
    """

    @abstractmethod
    def chat(self, messages, tools=None) -> list[dict]:
        """
        Send a chat request to the LLM and return the response.

        This abstract method must be implemented by subclasses to handle
        communication with specific LLM providers.

        Args:
            messages (list): A list of message dictionaries with "role" and "content" fields
            tools (list, optional): A list of Tool instances to be made available to the LLM

        Returns:
            list[dict]: A list of message dictionaries containing the model's response(s)
        """
        pass


# Ollama Provider implementation
class OllamaProvider(LLMProvider):
    """
    LLM Provider implementation for Ollama models like Llama 3.1, Mistral Nemo, and others.

    Ollama's message structure follows a specific format:

    1. Message Format:
       - Each message is a dict with "role" and "content" fields
       - Role can be: "user", "assistant", or "tool"
       - Content contains the message text (string)
       - The message history is passed as a list of these message objects

    2. Tool Calls:
       - When a model wants to call a tool, the response includes a "tool_calls" field
       - Each tool call contains:
         - "function": An object with "name" and "arguments" (dict)
         - "arguments" contains the parameters to be passed to the function
       - When responding to a tool call, you provide a message with:
         - "role": "tool"
         - "name": The name of the function that was called
         - "content": The result of the function call

    3. Response Structure:
       - response["message"] contains the model's response
       - It includes fields like "role", "content", and optionally "tool_calls"
       - The response message format is consistent with the input message format
       - If a tool is called, response["message"]["tool_calls"] will be present

    4. Tool Call Flow:
       - Model generates a response with tool_calls
       - Application executes the tool(s) based on arguments
       - Result is sent back as a "tool" role message
       - Model generates a new response incorporating tool results

    For more details, see: https://ollama.com/blog/tool-support
    """

    def __init__(self, model_name):
        import ollama

        self.client = ollama.Client()
        self.model_name = model_name

    def chat(self, messages, tools=[]):
        """
        Send a chat request to the Ollama API and return the response.

        If tools are provided and the model calls a tool, this method will
        execute the tool with the provided arguments and return both the
        model's message and the tool result.

        Args:
            messages: List of message dicts with "role" and "content" fields
            tools: List of Tool instances to be passed to the model

        Returns:
            List of message dicts including the model's response and any tool results
        """
        response = self.client.chat(
            model=self.model_name, messages=messages, tools=self.format_tools(tools)
        )

        # Check if the model wants to use a tool
        if "tool_calls" in response["message"]:
            result_messages = [response["message"]]

            # Process each tool call
            for tool_call in response["message"]["tool_calls"]:
                # Execute the tool
                tool_result = self.execute_tool(tools, tool_call)
                result_messages.append(tool_result)

            return result_messages

        return [response["message"]]

    def format_tools(self, tools):
        """
        Convert our tool format to Ollama's format.

        Ollama expects tools in the following format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }

        Args:
            tools: List of Tool instances to convert

        Returns:
            List of tools in Ollama's format
        """
        return [tool.definition for tool in tools]

    def execute_tool(self, tools: list[Tool], tool_call: dict) -> dict:
        """
        Execute a tool based on the tool call from the model.

        Args:
            tools: List of available Tool instances
            tool_call: Tool call object from the model response

        Returns:
            A message dict with the tool execution result in the format:
            {
                "role": "tool",
                "name": "tool_name",
                "content": "tool result"
            }
        """
        for tool in tools:
            if tool.name == tool_call["function"]["name"]:
                tool_result = tool.execute(**tool_call["function"]["arguments"])
                return {
                    "role": "tool",
                    "content": tool_result,
                    "name": tool_call["function"]["name"],
                }
        return {"error": f"Tool '{tool_call['function']['name']}' not found."}


# OpenAI Provider implementation
class OpenAIProvider(LLMProvider):
    """
    LLM Provider implementation for OpenAI models like GPT-3.5-Turbo and GPT-4.

    OpenAI's message structure follows a specific format:

    1. Message Format:
       - Each message is a dict with "role" and "content" fields
       - Role can be: "system", "user", "assistant", or "tool"
       - Content contains the message text (string) or content blocks (for multimodal)
       - Messages can have optional "name" field to identify specific users/assistants

    2. Tool Calls:
       - When a model wants to call a tool, the response includes a "tool_calls" field
       - Each tool call contains:
         - "id": A unique identifier for the tool call
         - "function": An object with "name" and "arguments" (JSON string)
       - When responding to a tool call, you provide a message with:
         - "role": "tool"
         - "tool_call_id": The ID of the tool call being responded to
         - "name": The name of the function that was called
         - "content": The result of the function call

    3. Response Structure:
       - response.choices[0].message contains the model's response
       - It includes fields like "role", "content", and optionally "tool_calls"
       - response.usage contains token usage statistics
         - "prompt_tokens": Number of tokens in the input
         - "completion_tokens": Number of tokens in the output
         - "total_tokens": Total tokens used

    4. Tool Call Flow:
       - Model generates a response with tool_calls
       - Application executes the tool(s) based on arguments
       - Result is sent back as a "tool" message
       - Model generates a new response incorporating tool results

    For more details, see: https://platform.openai.com/docs/guides/function-calling
    """

    def __init__(self, model_name, api_key=None):
        import openai

        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def chat(self, messages, tools: list[Tool] = []):
        """
        Send a chat request to the OpenAI API and return the response.

        If tools are provided and the model calls a tool, this method will:
        1. Execute the tool with the provided arguments
        2. Call the model again with the tool result
        3. Return both the original assistant message and the final response

        Args:
            messages: List of message dicts with "role" and "content" fields
            tools: List of Tool instances to be passed to the model

        Returns:
            List of message dicts including the model's response and any tool results
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.format_tools(tools),
        )

        # Get the message from the response
        message = response.choices[0].message

        # Convert to dict if it's an object
        message_dict = (
            message.model_dump() if hasattr(message, "model_dump") else message
        )

        # Check for tool calls - try both attribute and dictionary access
        tool_calls = None
        try:
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = message.tool_calls
        except AttributeError:
            if message_dict.get("tool_calls"):
                tool_calls = message_dict["tool_calls"]

        # Handle tool calls if present
        if tool_calls:
            # Add the assistant message to results
            result_messages = [message_dict]

            for tool_call in tool_calls:
                # Print tool call in real time
                print(f"Tool call: {tool_call}")
                print("-" * 50)

                # Execute the tool
                tool_result = self.execute_tool(tools, tool_call)
                # print(tool_result)
                # print("-" * 50)

                # Call the LLM again with the tool result
                tool_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        *messages,
                        message_dict,
                        tool_result,
                    ],
                )

                # Get the second response from OpenAI
                tool_response_message = tool_response.choices[0].message
                tool_response_dict = (
                    tool_response_message.model_dump()
                    if hasattr(tool_response_message, "model_dump")
                    else tool_response_message
                )

                # Add the follow-up response to the results
                result_messages.append(tool_response_dict)

            return result_messages

        # If no tool calls, return the message as a list
        return [message_dict]

    def format_tools(self, tools: list[Tool] = []) -> list[ChatCompletionToolParam]:
        """
        Convert our tool format to OpenAI's format.

        OpenAI expects tools in the following format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }

        Args:
            tools: List of Tool instances to convert

        Returns:
            List of tools in OpenAI's format
        """
        # OpenAI uses a slightly different format than Ollama
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def execute_tool(self, tools: list[Tool], tool_call) -> dict:
        """
        Execute a tool based on the tool call from the model.

        Handles tool calls in both object format and dictionary format for compatibility
        with different versions of the OpenAI API.

        Args:
            tools: List of available Tool instances
            tool_call: Tool call object from the model response (can be object or dict)

        Returns:
            A message dict with the tool execution result in the format:
            {
                "role": "tool",
                "content": "tool result",
                "tool_call_id": "id of the tool call",
            }
        """
        try:
            # Try object-style access first (newer OpenAI SDK)
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments
            tool_call_id = tool_call.id
        except (AttributeError, TypeError):
            # Fall back to dictionary-style access
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]

        for tool in tools:
            if tool.name == function_name:
                # Parse arguments (could be a JSON string or already a dict)
                args = (
                    json.loads(function_args)
                    if isinstance(function_args, str)
                    else function_args
                )
                return {
                    "role": "tool",
                    "content": tool.execute(**args),
                    "tool_call_id": tool_call_id,
                }
        return {"error": f"Tool '{function_name}' not found."}


# Anthropic Provider implementation
class AnthropicProvider(LLMProvider):
    """
    LLM Provider implementation for Anthropic Claude models.

    Anthropic's message structure follows a specific format:

    1. Message Format:
       - Messages are exchanged as alternating 'user' and 'assistant' roles
       - Each message has a 'role' and 'content'
       - Content can be a string or an array of content blocks (e.g., text, images, tool use)

    2. Content Block Types:
       - TextBlock: Simple text content ({"type": "text", "text": "content"})
       - ToolUseBlock: Used when Claude wants to call a tool
         ({"type": "tool_use", "id": "tool_id", "name": "tool_name", "input": {...}})
       - Images and other media types are also supported

    3. Response Structure:
       - id: Unique identifier for the response
       - model: The Claude model used
       - type: Always "message"
       - role: Always "assistant"
       - content: Array of content blocks
       - stop_reason: Why generation stopped (e.g., "end_turn", "max_tokens", "tool_use")
       - stop_sequence: The sequence that triggered stopping (if applicable)
       - usage: Token usage statistics

    4. Tool Use Flow:
       - Claude requests to use a tool via a ToolUseBlock
       - The application executes the tool and returns results
       - Results are provided back as a tool_result message

    For more details, see: https://docs.anthropic.com/claude/reference/messages_post
    """

    def __init__(self, model_name, api_key=None):
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model_name = model_name

    def chat(self, messages, tools=[]):
        # Create message with tools if provided
        if tools:
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                tools=self.format_tools(tools),
                max_tokens=1024,
            )
        else:
            response = self.client.messages.create(
                model=self.model_name, messages=messages, max_tokens=1024
            )

        content = []
        # Check if Claude wants to use a tool
        if response.stop_reason == "tool_use":
            # Return in a format consistent with our interface
            tool_result = None
            for block in response.content:
                content.append(block.model_dump())
                if isinstance(block, ToolUseBlock):
                    tool_use = block
                    for tool in tools:
                        if tool.name == tool_use.name:
                            # print(f"Use tool: {tool.name}")
                            # print(f"Tool input: {tool_use.input}")
                            tool_result = tool.execute(**tool_use.input)
                            # print(f"Tool result: {tool_result}")
            return [
                {
                    "role": "assistant",
                    "content": content,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_result,
                        }
                    ],
                },
            ]
        else:
            for block in response.content:
                content.append(block.model_dump())
            return [
                {
                    "role": "assistant",
                    "content": content,
                }
            ]

    def format_tools(self, tools: list[Tool]) -> list[ToolParam]:
        """
        Convert our tool format to Anthropic's format.

        Anthropic expects tools in the following format:
        {
            "name": "tool_name",
            "description": "tool description",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

        Args:
            tools: List of Tool instances to convert

        Returns:
            List of tools in Anthropic's format
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": tool.parameters.get("properties", {}),
                    "required": tool.parameters.get("required", []),
                },
            }
            for tool in tools
        ]


# CoTAgent class to manage the reasoning process
class CoTAgent:
    """
    Agent that implements Chain of Thought (CoT) reasoning with language models.

    The CoTAgent class orchestrates a step-by-step reasoning process using language models
    to solve complex problems. It manages the conversational context, tool calling, and
    the overall reasoning flow, breaking problems down into logical steps before arriving
    at a final answer.

    This agent can work with different LLM providers (OpenAI, Anthropic, Ollama) and
    can use various tools to augment the language model's capabilities.

    Architecture and Flow:
    ```
    +-----------------+     +-------------------+     +----------------------+
    | Problem/Question | --> | Initial Prompt    | --> | Language Model       |
    +-----------------+     +-------------------+     | Provider (LLMProvider)|
                                                      +----------------------+
                                                               |
                  +-------------------------------------------+
                  |
                  v
    +---------------------------+     +----------------------+
    | Response Analysis         | <-- | Tool Execution       |
    | - Check for tool calls    | --> | (if needed)          |
    | - Extract reasoning steps |     +----------------------+
    | - Check for final answer  |             ^
    +---------------------------+             |
                  |                           |
                  v                           |
    +---------------------------+             |
    | Continue Reasoning?       |-------------+
    | - Yes: Next step          |
    | - No: Final answer        |
    +---------------------------+
                  |
                  v
    +---------------------------+
    | Final Answer              |
    | Extracted & Returned      |
    +---------------------------+
    ```

    Flow Process:
    1. User provides a problem/question
    2. Agent builds initial prompt with CoT instructions
    3. LLM provider processes the prompt
    4. Agent analyzes response for:
       - Tool calls (executes if needed)
       - Reasoning steps
       - Final answer indication
    5. If reasoning is incomplete, repeat steps 3-4
    6. Once complete, extract and return final answer
    """

    def __init__(self, provider, tools=None, max_steps=10, prompt_builder=None):
        """
        Initializes the CoT agent.

        Args:
            provider (LLMProvider): An LLM provider instance.
            tools (list, optional): List of Tool instances. Defaults to None (empty list).
            max_steps (int, optional): Maximum reasoning steps to prevent infinite loops. Defaults to 10.
            prompt_builder (callable, optional): Custom function to build the initial prompt.
                                               Defaults to the internal _default_prompt_builder.
        """
        self.provider = provider
        self.tools = tools or []
        self.max_steps = max_steps
        self.prompt_builder = prompt_builder or self._default_prompt_builder
        self.history = []
        self.system_prompt = "You are a helpful AI assistant that solves problems using step-by-step reasoning. "

    def _default_prompt_builder(self, problem):
        """
        Default prompt builder that instructs the model to reason step-by-step.

        Creates a prompt that guides the LLM to break down the problem, show its
        reasoning for each step, consider different approaches, and conclude with
        a final answer.

        Args:
            problem (str): The problem or question to solve.

        Returns:
            str: The formatted prompt with instructions for step-by-step reasoning.
        """
        return (
            f"Problem to solve: {problem}\n\n"
            "Reasoning approach:\n"
            "1. Analyze the problem and identify key components\n"
            "2. Develop a step-by-step solution strategy\n"
            "3. Execute each step with clear reasoning\n"
            "4. Evaluate your solution and consider alternative approaches\n"
            "5. If you need to use tools, clearly state which tool you need and why\n\n"
            "When you reach a conclusion, format your final answer as:\n"
            "Final Answer: [concise solution with key insights]\n\n"
            "After providing your final answer, respond with empty content to indicate you've completed the task.\n\n"
            "Begin your reasoning now."
        )

    def solve_problem(self, problem):
        """
        Solves the given problem using CoT reasoning and tool calling.

        This method manages the entire reasoning process, including:
        1. Initializing the conversation with the problem statement
        2. Conducting a multi-step reasoning process with the LLM
        3. Using tools when needed to gather information or perform actions
        4. Collecting a final answer once reasoning is complete

        Args:
            problem (str): The problem or question to solve.

        Returns:
            str: The final answer extracted from the LLM's reasoning process.
        """
        # Initialize conversation with system prompt and user prompt
        initial_prompt = self.prompt_builder(problem)
        self.history = [
            {"role": "user", "content": initial_prompt},
        ]

        print("Starting to solve problem:", problem)
        print("-" * 50)

        # Reasoning loop
        for step in range(self.max_steps):
            messages = self.provider.chat(messages=self.history, tools=self.tools)
            self.history.extend(messages)
            # print(messages)
            agent_finished = False

            for message in messages:
                if message["role"] == "assistant":
                    content = message["content"]
                    if isinstance(content, list):
                        for block in content:
                            if block["type"] == "text":
                                step_info = f"Step {step + 1}: {block['text']}"
                                break
                            else:
                                print(block)
                    else:
                        step_info = f"Step {step + 1}: {content}"

                    print(step_info)
                    print("-" * 50)

                    if content == []:
                        agent_finished = True
                        break

            if agent_finished:
                break

        self.history.append(
            {
                "role": "user",
                "content": "Summarize the solution from the steps above",
            }
        )

        messages = self.provider.chat(messages=self.history, tools=self.tools)
        self.history.extend(messages)

        # Handle different content structures from various providers
        content = messages[0]["content"]

        # Handle Anthropic's content block array format
        if isinstance(content, list):
            result = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    result += block.get("text", "")
                elif isinstance(block, dict) and "text" in block:
                    result += block["text"]
            return result
        # Handle simple text content (OpenAI style)
        elif isinstance(content, dict) and "text" in content:
            return content["text"]
        # Handle plain string content
        else:
            return content

    def execute_tool(self, tool_name, tool_args):
        """
        Executes the specified tool with given arguments.

        This is a utility method used by the agent to call tools during
        the reasoning process.

        Args:
            tool_name (str): The name of the tool to execute.
            tool_args (dict): Arguments to pass to the tool.

        Returns:
            str: The result returned by the tool or an error message.
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.execute(**tool_args)
        return f"Error: Tool '{tool_name}' not found."


# Example Usage
if __name__ == "__main__":
    """
    Main entry point for the CoT Agent when run as a script.

    This section demonstrates how to use the CoT Agent with different LLM providers.
    It initializes the agent with various tools and selects the LLM provider based on
    environment variables or command line arguments.

    Environment variables:
        LLM_PROVIDER: The LLM provider to use ("openai", "anthropic", or "ollama")
        OPENAI_MODEL: The OpenAI model to use (default: "gpt-4o")
        ANTHROPIC_MODEL: The Anthropic model to use (default: "claude-3-7-sonnet-20250219")
        OLLAMA_MODEL: The Ollama model to use (default: "llama3.2")

    Usage:
        python cot_agent.py "What is 2+2?"

    Returns:
        Prints the agent's step-by-step reasoning and final answer to the console.
    """
    # Define tools
    calculator = CalculatorTool()
    browser = BrowserTool()
    execute_shell = ExecuteShellTool()
    file_read = FileReadTool()
    google_search = GoogleSearchTool()

    tools = [calculator, browser, execute_shell, file_read, google_search]

    # Check for provider type in command line args or environment
    provider_type = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider_type == "openai":
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        provider = OpenAIProvider(model_name)
    elif provider_type == "anthropic":
        model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
        provider = AnthropicProvider(model_name)
    else:  # default to ollama
        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        provider = OllamaProvider(model_name)

    # Initialize the agent with the selected provider
    agent = CoTAgent(
        provider=provider,
        tools=tools,
        max_steps=10,
    )

    # Define a problem from command line args
    problem = sys.argv[1] if len(sys.argv) > 1 else "What is 2+2?"

    # Solve the problem
    answer = agent.solve_problem(problem)

    # Display the final answer
    print(f"\nFinal Answer: {answer}")
