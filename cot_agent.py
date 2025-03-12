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
4. An interactive ChatInterface with readline support and provider/model management

Features:
- Interactive command line chat with command history
- Support for multiple LLM providers (OpenAI, Anthropic, Ollama)
- Ability to change providers and models at runtime
- Step-by-step reasoning with tool use capabilities
- Command-based interface for configuration
"""

import datetime
import sys
import traceback  # System-specific parameters and functions
import dotenv  # For loading environment variables from a .env file
import json  # For JSON serialization/deserialization
import os  # For interacting with the operating system and environment variables
import platform  # For detecting operating system
import time  # For timing and delays
import threading  # For background tasks

# Import readline (Unix) or pyreadline3 (Windows)
readline_available = True
try:
    if platform.system() == "Windows":
        import pyreadline3 as readline
    else:
        import readline  # For command history and line editing
except ImportError:
    print(
        "Warning: readline or pyreadline3 is not installed. Command history will be disabled."
    )
    readline_available = False

# Import tools from the tools module
from tools import (
    CalculatorTool,
    BrowserTool,
    ExecuteShellTool,
    FileReadTool,
    GoogleSearchTool,
    GoogleFinanceTool,
    GoogleFlightsTool,
    GoogleNewsTool,
)

# Import providers from the providers module
from providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
)

dotenv.load_dotenv()


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
    +---------------------------+
    | Planning Phase            |
    | - Analyze problem         |
    | - Break into sub-problems |
    | - Create solution plan    |
    | - Identify needed tools   |
    +---------------------------+
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
    3. Language model begins with planning phase:
       - Analyze and understand the problem
       - Break it down into sub-problems
       - Create a specific execution plan
       - Identify tools that might be needed
    4. LLM provider processes the prompt
    5. Agent analyzes response for:
       - Tool calls (executes if needed)
       - Extract reasoning steps
       - Check for final answer indication
    6. If reasoning is incomplete, repeat steps 4-5
    7. Once complete, extract and return final answer
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
        self.system_prompt = f"""
        You are a helpful AI assistant that solves problems using step-by-step reasoning.
        Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.
        You first create a clear plan before execution.
        You think step by step through both planning and execution phases.
        You provide a final answer after completing your reasoning.
        Optimize for speed and efficiency.
        """
        self.chat_history = []  # Store all chat interactions for reference

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
            "Approach:\n"
            "1. Planning Phase:\n"
            "   a. Analyze problem and identify key components\n"
            "   b. Break down the problem into sub-problems\n"
            "   c. Create a specific plan with clear steps for solving each sub-problem\n"
            "   d. Identify potential tools needed for execution\n"
            "2. Execution Phase:\n"
            "   a. Follow your plan step-by-step with clear reasoning\n"
            "   b. Adjust the plan if needed based on new insights\n"
            "   c. Use tools when needed (state which tool and why)\n"
            "3. When complete, provide: Final Answer: [concise solution]\n"
            "4. End with empty content to signal completion\n\n"
            "Begin your planning and reasoning now."
        )

    def solve_problem(self, problem, show_thinking=False):
        """
        Solves the given problem using CoT reasoning and tool calling.

        This method manages the entire reasoning process, including:
        1. Initializing the conversation with the problem statement
        2. Conducting a multi-step reasoning process with the LLM
        3. Using tools when needed to gather information or perform actions
        4. Collecting a final answer once reasoning is complete

        Args:
            problem (str): The problem or question to solve.
            show_thinking (bool, optional): Whether to display step-by-step reasoning. Defaults to False.

        Returns:
            str: The final answer extracted from the LLM's reasoning process.
        """
        # Initialize conversation with system prompt and user prompt
        initial_prompt = self.prompt_builder(problem)
        self.history = [
            {"role": "user", "content": initial_prompt},
        ]

        # Add to chat history
        self.chat_history.append({"role": "user", "content": problem})

        if show_thinking:
            print("Starting to solve problem:", problem)
            print("-" * 50)

        # Reasoning loop
        for step in range(self.max_steps):
            messages = self.provider.chat(messages=self.history, tools=self.tools)
            self.history.extend(messages)
            agent_finished = False
            yield messages

            # Display thinking steps if enabled
            if show_thinking:
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

            # Check if the agent is finished
            for message in messages:
                if message["role"] == "assistant" and self.provider.is_finished(
                    message
                ):
                    if show_thinking:
                        print("Agent finished")
                    agent_finished = True
                    break

            if agent_finished:
                break

        self.history.append(
            {
                "role": "user",
                "content": """
                Provide the most appropriate answer to type of question based on the findings in the previous steps.
                If the question requires a long answer, provide a detailed answer.
                Prefer concise if the question is short.
                """,
            }
        )

        messages = self.provider.chat(messages=self.history, tools=self.tools)
        self.history.extend(messages)
        # Handle different content structures from various providers
        content = messages[0]["content"]
        yield messages
        # # Handle Anthropic's content block array format
        # if isinstance(content, list):
        #     result = ""
        #     for block in content:
        #         if isinstance(block, dict) and block.get("type") == "text":
        #             result += block.get("text", "")
        #         elif isinstance(block, dict) and "text" in block:
        #             result += block["text"]
        #     return result
        # # Handle simple text content (OpenAI style)
        # elif isinstance(content, dict) and "text" in content:
        #     return content["text"]
        # # Handle plain string content
        # else:
        #     return content

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

    def clear_history(self):
        """Clears the conversation history."""
        self.history = []
        return "Conversation history cleared."


class ChatInterface:
    """
    Interactive chat interface for the CoT Agent.

    Provides a command-line interface with history support, commands for changing
    providers and models, and persistent chat history.
    """

    def __init__(self):
        """Initialize the chat interface with default settings."""
        # Define available tools
        self.tools = [
            CalculatorTool(),
            BrowserTool(),
            ExecuteShellTool(),
            FileReadTool(),
            GoogleSearchTool(),
            GoogleFinanceTool(),
            GoogleFlightsTool(),
            GoogleNewsTool(),
        ]

        # Set up readline history
        self.history_file = os.path.expanduser("~/.cot_agent_history")
        self._setup_readline()

        # Provider configuration
        self.providers = {
            "openai": {
                "default_model": os.getenv("OPENAI_MODEL", "gpt-4o"),
                "available_models": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
                "class": OpenAIProvider,
            },
            "anthropic": {
                "default_model": os.getenv(
                    "ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219"
                ),
                "available_models": [
                    "claude-3-7-sonnet-20250219",
                    "claude-3-5-sonnet-20240620",
                    "claude-3-haiku-20240307",
                ],
                "class": AnthropicProvider,
            },
            "ollama": {
                "default_model": os.getenv("OLLAMA_MODEL", "llama3.2"),
                "available_models": ["llama3.2", "llama3", "mistral"],
                "class": OllamaProvider,
            },
        }

        # Default provider from environment or fallback to ollama
        self.current_provider_name = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.current_model = self.providers[self.current_provider_name]["default_model"]

        # Initialize provider
        provider_class = self.providers[self.current_provider_name]["class"]
        self.provider = provider_class(self.current_model)

        # Initialize agent
        self.agent = CoTAgent(
            provider=self.provider,
            tools=self.tools,
            max_steps=10,
        )

        # User preferences
        self.show_thinking = False  # Default to hiding thinking steps

        # Commands available in the chat interface
        self.commands = {
            "/help": self._cmd_help,
            "/provider": self._cmd_provider,
            "/model": self._cmd_model,
            "/providers": self._cmd_list_providers,
            "/models": self._cmd_list_models,
            "/clear": self._cmd_clear,
            "/maxsteps": self._cmd_maxsteps,
            "/thinking": self._cmd_toggle_thinking,
            "/exit": self._cmd_exit,
            "/quit": self._cmd_exit,
        }

    def _setup_readline(self):
        """Set up readline for command history."""
        if not readline_available:
            return

        try:
            # Create history file if it doesn't exist
            if not os.path.exists(self.history_file):
                with open(self.history_file, "w") as f:
                    pass

            # Load history
            readline.read_history_file(self.history_file)
            # Set history length
            readline.set_history_length(1000)
        except Exception as e:
            print(f"Warning: Could not set up readline history: {e}")

    def _save_history(self):
        """Save readline history to file."""
        if not readline_available:
            return

        try:
            readline.write_history_file(self.history_file)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")

    def _cmd_help(self, args=None):
        """Display help information."""
        help_text = """
Available commands:
  /help              - Show this help message
  /provider <name>   - Change provider (openai, anthropic, ollama)
  /providers         - List available providers
  /model <name>      - Change model for current provider
  /models            - List available models for current provider
  /maxsteps <number> - Set maximum reasoning steps (default: 10)
  /thinking          - Toggle display of thinking steps (current: {thinking_state})
  /clear             - Clear conversation history
  /exit or /quit     - Exit the chat

Current configuration:
  Provider: {provider}
  Model: {model}
  Max Steps: {max_steps}
  Show Thinking: {thinking_state}
""".format(
            provider=self.current_provider_name,
            model=self.current_model,
            max_steps=self.agent.max_steps,
            thinking_state="enabled" if self.show_thinking else "disabled",
        )
        return help_text

    def _cmd_provider(self, args):
        """Change the current provider."""
        if not args:
            return "Please specify a provider. Available providers: " + ", ".join(
                self.providers.keys()
            )

        provider_name = args[0].lower()
        if provider_name not in self.providers:
            return (
                f"Unknown provider: {provider_name}. Available providers: "
                + ", ".join(self.providers.keys())
            )

        # Change provider
        self.current_provider_name = provider_name
        self.current_model = self.providers[provider_name]["default_model"]

        # Initialize new provider
        provider_class = self.providers[provider_name]["class"]
        self.provider = provider_class(self.current_model)

        # Update agent
        self.agent.provider = self.provider
        self.agent.clear_history()

        return f"Provider changed to {provider_name} with model {self.current_model}"

    def _cmd_model(self, args):
        """Change the model for the current provider."""
        if not args:
            return "Please specify a model. Available models: " + ", ".join(
                self.providers[self.current_provider_name]["available_models"]
            )

        model_name = args[0]
        # Allow any model name, not just the predefined ones
        self.current_model = model_name

        # Reinitialize provider with new model
        provider_class = self.providers[self.current_provider_name]["class"]
        self.provider = provider_class(self.current_model)

        # Update agent
        self.agent.provider = self.provider
        self.agent.clear_history()

        return (
            f"Model changed to {model_name} for provider {self.current_provider_name}"
        )

    def _cmd_list_providers(self, args=None):
        """List available providers."""
        providers_list = ", ".join(self.providers.keys())
        return f"Available providers: {providers_list}\nCurrent provider: {self.current_provider_name}"

    def _cmd_list_models(self, args=None):
        """List available models for the current provider."""
        models = self.providers[self.current_provider_name]["available_models"]
        models_list = ", ".join(models)
        return f"Available models for {self.current_provider_name}: {models_list}\nCurrent model: {self.current_model}"

    def _cmd_clear(self, args=None):
        """Clear conversation history."""
        self.agent.clear_history()
        return "Conversation history cleared."

    def _cmd_maxsteps(self, args):
        """Set the maximum number of reasoning steps."""
        if not args:
            return f"Current max steps: {self.agent.max_steps}. Please specify a number to change it."

        try:
            steps = int(args[0])
            if steps < 1:
                return "Max steps must be at least 1."

            self.agent.max_steps = steps
            return f"Maximum reasoning steps set to {steps}."
        except ValueError:
            return f"Invalid number: {args[0]}. Please provide a positive integer."

    def _cmd_toggle_thinking(self, args=None):
        """Toggle display of thinking steps."""
        self.show_thinking = not self.show_thinking
        return f"Thinking steps display is now {'enabled' if self.show_thinking else 'disabled'}."

    def _cmd_exit(self, args=None):
        """Exit the chat interface."""
        self._save_history()
        print("Goodbye!")
        sys.exit(0)

    def _handle_command(self, user_input):
        """Parse and handle commands."""
        parts = user_input.strip().split()
        if not parts:
            return None

        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command in self.commands:
            return self.commands[command](args)

        return None

    def start(self):
        """Start the interactive chat loop."""
        print(
            f"""
CoT Agent Chat Interface
========================
Type your questions or use commands (type /help for available commands)
Current configuration: Provider: {self.current_provider_name}, Model: {self.current_model}
"""
        )

        # For managing the animation thread
        animation_stop_event = threading.Event()
        animation_thread = None

        def progress_indicator():
            """Simple dot animation that adds one dot at a time."""
            while not animation_stop_event.is_set():
                print(".", end="", flush=True)
                time.sleep(0.8)

        while True:
            try:
                # Get user input with prompt
                user_input = input("\nYou: ").strip()

                # Skip empty inputs
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    result = self._handle_command(user_input)
                    if result:
                        print(f"\n{result}")
                    continue

                # Process normal input through the agent
                if self.show_thinking:
                    print("\nAgent is thinking...")
                else:
                    # Make sure any previous animation is stopped
                    if animation_thread and animation_thread.is_alive():
                        animation_stop_event.set()
                        animation_thread.join(timeout=1.0)
                        animation_stop_event.clear()
                        animation_thread = None

                    print("\nThinking", end="", flush=True)
                    animation_thread = threading.Thread(target=progress_indicator)
                    animation_thread.daemon = True
                    animation_thread.start()

                # Solve the problem
                for messages in self.agent.solve_problem(
                    user_input, show_thinking=self.show_thinking
                ):
                    for message in messages:
                        for content in message["content"]:
                            if "text" in content:
                                print(content["text"])
                            elif "tool_result" in content:
                                print(content["tool_result"])

                # Stop the animation if it was running
                if (
                    not self.show_thinking
                    and animation_thread
                    and animation_thread.is_alive()
                ):
                    animation_stop_event.set()
                    animation_thread.join(timeout=1.0)
                    # Clear the entire line once finished
                    print("\r" + " " * 50 + "\r", end="", flush=True)
                    # Reset animation state for next query
                    animation_stop_event.clear()
                    animation_thread = None

            except KeyboardInterrupt:
                # Also stop animation on keyboard interrupt
                if animation_thread and animation_thread.is_alive():
                    animation_stop_event.set()
                    animation_thread.join(timeout=1.0)
                    print("\r" + " " * 50 + "\r", end="", flush=True)
                    animation_stop_event.clear()
                    animation_thread = None
                print("\nOperation interrupted. Type /exit to quit.")
            except EOFError:
                if animation_thread and animation_thread.is_alive():
                    animation_stop_event.set()
                    animation_thread.join(timeout=1.0)
                    print("\r" + " " * 50 + "\r", end="", flush=True)
                    animation_stop_event.clear()
                    animation_thread = None
                self._cmd_exit()
            except Exception as e:
                if animation_thread and animation_thread.is_alive():
                    animation_stop_event.set()
                    animation_thread.join(timeout=1.0)
                    print("\r" + " " * 50 + "\r", end="", flush=True)
                    animation_stop_event.clear()
                    animation_thread = None
                print(f"\nError: {e}")
                traceback.print_exc()


# Example Usage
if __name__ == "__main__":
    """
    Main entry point for the CoT Agent when run as a script.

    This creates an interactive chat interface where you can:
    1. Ask questions to the AI agent
    2. Change between providers (OpenAI, Anthropic, Ollama)
    3. Change models for each provider
    4. Maintain command history between sessions

    Commands:
        /help              - Show help information
        /provider <name>   - Change provider (openai, anthropic, ollama)
        /providers         - List available providers
        /model <name>      - Change model for current provider
        /models            - List available models for current provider
        /maxsteps <number> - Set maximum reasoning steps (default: 10)
        /thinking          - Toggle display of thinking steps (off by default)
        /clear             - Clear conversation history
        /exit or /quit     - Exit the chat

    Environment variables:
        LLM_PROVIDER: The default LLM provider to use ("openai", "anthropic", or "ollama")
        OPENAI_MODEL: The default OpenAI model to use (default: "gpt-4o")
        ANTHROPIC_MODEL: The default Anthropic model to use (default: "claude-3-7-sonnet-20250219")
        OLLAMA_MODEL: The default Ollama model to use (default: "llama3.2")
    """
    chat = ChatInterface()
    chat.start()
