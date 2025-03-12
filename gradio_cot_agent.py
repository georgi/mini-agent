import traceback
import gradio as gr
from gradio import ChatMessage
import os
import sys
import json
import yaml
import requests
from dotenv import load_dotenv, find_dotenv, set_key
from cot_agent import (
    CoTAgent,
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    CalculatorTool,
    BrowserTool,
    ExecuteShellTool,
    FileReadTool,
    GoogleSearchTool,
    GoogleFinanceTool,
    GoogleFlightsTool,
    GoogleNewsTool,
)

# Lists of known models for each provider
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
]

ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-7-sonnet-20250219",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
    "claude-2.0",
    "claude-instant-1.2",
]

# Settings file path
SETTINGS_FILE = "cot_agent_settings.yaml"

# Default settings
DEFAULT_SETTINGS = {
    "provider": "ollama",
    "openai_model": "gpt-4o",
    "anthropic_model": "claude-3-7-sonnet-20250219",
    "ollama_model": "llama3.2",
    "max_steps": 10,
}


# Functions to save and load settings
def load_settings():
    """Load settings from YAML file or return defaults"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as file:
                settings = yaml.safe_load(file)
                # Ensure all required settings exist
                for key, value in DEFAULT_SETTINGS.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        return DEFAULT_SETTINGS.copy()
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    """Save settings to YAML file"""
    try:
        with open(SETTINGS_FILE, "w") as file:
            yaml.dump(settings, file)
        return "Settings saved successfully!"
    except Exception as e:
        return f"Error saving settings: {str(e)}"


# Load settings at startup
settings = load_settings()

# Add global variables for current settings (initialized from saved settings)
global_provider = settings["provider"]
global_model = settings[f"{global_provider}_model"]
global_max_steps = settings["max_steps"]

# CSS for better styling
custom_css = """
/* Add any custom CSS you need for the chatbot */
"""

# Load environment variables from .env file
dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    dotenv_path = os.path.join(os.getcwd(), ".env")
    with open(dotenv_path, "w") as f:
        f.write(
            "OPENAI_API_KEY=\nOPENAI_MODEL=gpt-4o\nANTHROPIC_API_KEY=\nANTHROPIC_MODEL=claude-3-7-sonnet-20250219\nSERPAPI_KEY=\nOLLAMA_MODEL=llama3.2\n"
        )
load_dotenv(dotenv_path)


# Functions to manage API credentials
def get_api_credentials():
    """Get current API credentials from environment variables"""
    credentials = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219"),
        "SERPAPI_KEY": os.getenv("SERPAPI_KEY", ""),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3.2"),
    }
    return credentials


def save_api_credentials(openai_key, anthropic_key, serpapi_key):
    """Save API credentials to .env file"""
    try:
        # Get current models from env vars (preserve them)
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

        # Update environment variables
        set_key(dotenv_path, "OPENAI_API_KEY", openai_key)
        set_key(dotenv_path, "ANTHROPIC_API_KEY", anthropic_key)
        set_key(dotenv_path, "SERPAPI_KEY", serpapi_key)

        # Reload environment variables
        load_dotenv(dotenv_path, override=True)

        return "API credentials saved successfully!"
    except Exception as e:
        return f"Error saving API credentials: {str(e)}"


# Function to update the selected model in environment variables
def update_selected_model(provider, model):
    """Save the selected model to environment variables"""
    try:
        if provider == "openai":
            set_key(dotenv_path, "OPENAI_MODEL", model)
        elif provider == "anthropic":
            set_key(dotenv_path, "ANTHROPIC_MODEL", model)
        elif provider == "ollama":
            set_key(dotenv_path, "OLLAMA_MODEL", model)

        # Reload environment variables
        load_dotenv(dotenv_path, override=True)
        return True
    except Exception as e:
        print(f"Error saving model selection: {str(e)}")
        return False


# Initialize tools
def initialize_tools():
    calculator = CalculatorTool()
    browser = BrowserTool()
    execute_shell = ExecuteShellTool()
    file_read = FileReadTool()
    google_search = GoogleSearchTool()
    google_finance = GoogleFinanceTool()
    google_flights = GoogleFlightsTool()
    google_news = GoogleNewsTool()

    return [
        calculator,
        browser,
        execute_shell,
        file_read,
        google_search,
        google_finance,
        google_flights,
        google_news,
    ]


# Initialize the agent based on provider selection
def initialize_agent(provider_type, model_name, max_steps):
    tools = initialize_tools()

    if provider_type == "openai":
        if not model_name:
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        else:
            update_selected_model("openai", model_name)
        provider = OpenAIProvider(model_name)
    elif provider_type == "anthropic":
        if not model_name:
            model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
        else:
            update_selected_model("anthropic", model_name)
        provider = AnthropicProvider(model_name)
    else:  # default to ollama
        if not model_name:
            model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        else:
            update_selected_model("ollama", model_name)
        provider = OllamaProvider(model_name)

    # We'll extend the CoTAgent to support Gradio output
    class GradioCoTAgent(CoTAgent):
        def solve_problem_with_gradio(self, problem, history, progress=None):
            """Version of solve_problem that updates a Gradio progress component and streams output"""
            messages = []

            # Add user message
            messages.append(ChatMessage(role="user", content=problem))
            yield messages

            thinking_content = ""
            current_tool = None

            for step_data in super().solve_problem(problem, history):
                for message in step_data:
                    for content in message["content"]:
                        if "text" in content:
                            messages.append(
                                ChatMessage(
                                    role="assistant",
                                    content=content["text"],
                                    metadata={
                                        "title": "🧠 Thinking",
                                        "status": "pending",
                                    },
                                )
                            )

                        elif "tool_call" in content:
                            # Tool is being called - create a new tool section
                            tool_name = content["tool_call"]["name"]
                            tool_input = content["tool_call"]["input"]

                            tool_call_msg = f"Tool: {tool_name}\nInput: {json.dumps(tool_input, indent=2)}"

                            messages.append(
                                ChatMessage(
                                    role="assistant",
                                    content=tool_call_msg,
                                    metadata={
                                        "title": f"🔧 Using {tool_name}",
                                        "status": "pending",
                                    },
                                )
                            )

                        # elif "tool_result" in content:
                        #     # Update the tool message with the result
                        #     tool_result = content["tool_result"]

                        #     # Find the last tool message to update
                        #     for i in range(len(messages) - 1, -1, -1):
                        #         if (
                        #             messages[i].role == "assistant"
                        #             and "title" in (messages[i].metadata or {})
                        #             and messages[i]
                        #             .metadata["title"]
                        #             .startswith("🔧 Using")
                        #         ):
                        #             tool_content = (
                        #                 messages[i].content
                        #                 + f"\n\nResult:\n{tool_result}"
                        #             )
                        #             messages[i] = ChatMessage(
                        #                 role="assistant",
                        #                 content=tool_content,
                        #                 metadata={
                        #                     "title": messages[i].metadata["title"],
                        #                     "status": "done",
                        #                 },
                        #             )
                        #             break

                # Mark previous thinking as done when moving to next step
                for i in range(len(messages) - 1, -1, -1):
                    if (
                        messages[i].role == "assistant"
                        and "title" in (messages[i].metadata or {})
                        and messages[i].metadata["title"] == "🧠 Thinking"
                        and messages[i].metadata["status"] == "pending"
                    ):
                        messages[i] = ChatMessage(
                            role="assistant",
                            content=messages[i].content,
                            metadata={"title": "🧠 Thinking", "status": "done"},
                        )
                        break

                yield messages

            # Final answer - add as a regular message without metadata
            if thinking_content:
                # Add a final answer that summarizes the findings
                final_answer = (
                    "Based on my analysis:\n\n"
                    + thinking_content.split("\nFinal Answer:")[-1].strip()
                )
                messages.append(ChatMessage(role="assistant", content=final_answer))
                yield messages

    return GradioCoTAgent(
        provider=provider,
        tools=tools,
        max_steps=max_steps,
    )


# Modify solve_problem to use global variables
def solve_problem(problem, history, progress=gr.Progress()):
    try:
        # Use global variables for provider and model
        global global_provider, global_model, global_max_steps

        # Initialize the agent using global values
        agent = initialize_agent(global_provider, global_model, int(global_max_steps))

        # Solve the problem with streaming
        messages = []
        for updated_messages in agent.solve_problem_with_gradio(
            problem, history, progress
        ):
            messages = updated_messages
            yield messages
    except Exception as e:
        print(e)
        traceback.print_exc()
        messages = []
        messages.append(ChatMessage(role="user", content=problem))
        messages.append(
            ChatMessage(role="assistant", content=f"I encountered an error: {str(e)}")
        )
        yield messages


# Create functions to update global variables
def update_global_provider(provider):
    global global_provider, global_model
    global_provider = provider

    # Update model based on provider
    if provider == "openai":
        global_model = settings["openai_model"]
    elif provider == "anthropic":
        global_model = settings["anthropic_model"]
    else:  # ollama
        global_model = settings["ollama_model"]

    # Update settings
    settings["provider"] = provider
    save_settings(settings)

    return provider


def update_global_model(model, provider_type):
    global global_model

    if provider_type == global_provider:  # Only update if this is the active provider
        global_model = model
        # Update settings
        settings[f"{provider_type}_model"] = model
        save_settings(settings)

    return model


def update_global_max_steps(steps):
    global global_max_steps
    global_max_steps = steps

    # Update settings
    settings["max_steps"] = steps
    save_settings(settings)

    return steps


# Function to fetch available Ollama models
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            # Extract model names from the response
            model_names = [model["name"] for model in models_data.get("models", [])]
            return model_names if model_names else ["llama3.2"]  # Default if empty list
        else:
            print(f"Failed to fetch Ollama models: {response.status_code}")
            return ["llama3.2"]  # Default model
    except Exception as e:
        print(f"Error fetching Ollama models: {str(e)}")
        return ["llama3.2"]  # Default model


# Create the Gradio interface
with gr.Blocks(css=custom_css) as demo:
    with gr.Tabs():
        with gr.Tab("CoT Agent"):
            gr.Markdown("# Chain of Thought (CoT) Agent")

            # Add examples
            gr.Examples(
                inputs=[],
                examples=[
                    ["What is the square root of 144?"],
                    [
                        "If I invest $10,000 at 5% annual interest compounded monthly, how much will I have after 10 years?"
                    ],
                    ["How many ways can 5 people be seated at a round table?"],
                    ["What is the current weather in New York?"],
                ],
            )

            # Chat interface only (settings moved to Settings tab)
            chat_interface = gr.ChatInterface(
                solve_problem,
                description="Ask a complex question and watch the agent think through it step by step.",
                chatbot=gr.Chatbot(
                    type="messages",  # Important for using ChatMessage
                    render_markdown=True,
                    bubble_full_width=False,
                    show_copy_button=True,
                ),
            )

        with gr.Tab("Settings"):
            gr.Markdown("# Agent Settings")
            gr.Markdown(
                "Configure the agent's behavior and model selection. Settings are saved automatically to a YAML file."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # Provider selection
                    provider_dropdown = gr.Dropdown(
                        choices=[
                            ("Ollama (Local)", "ollama"),
                            ("OpenAI", "openai"),
                            ("Anthropic", "anthropic"),
                        ],
                        label="LLM Provider",
                        value=settings["provider"],
                    )

                    # Model selection (dynamic based on provider)
                    openai_model_dropdown = gr.Dropdown(
                        choices=OPENAI_MODELS,
                        label="OpenAI Model",
                        value=settings["openai_model"],
                        visible=(settings["provider"] == "openai"),
                    )

                    anthropic_model_dropdown = gr.Dropdown(
                        choices=ANTHROPIC_MODELS,
                        label="Anthropic Model",
                        value=settings["anthropic_model"],
                        visible=(settings["provider"] == "anthropic"),
                    )

                    # Ollama model selection with refresh button
                    with gr.Row():
                        ollama_model_dropdown = gr.Dropdown(
                            choices=get_ollama_models(),
                            label="Ollama Model",
                            value=settings["ollama_model"],
                            scale=4,
                            visible=(settings["provider"] == "ollama"),
                        )
                        refresh_ollama_btn = gr.Button("Refresh Ollama Models", scale=1)

                    max_steps_slider = gr.Slider(
                        minimum=3,
                        maximum=20,
                        value=settings["max_steps"],
                        step=1,
                        label="Maximum Reasoning Steps",
                    )

            settings_status = gr.Markdown("")

            # Connect provider dropdown to update global variable and change visibility
            provider_dropdown.change(
                fn=lambda provider: [
                    update_global_provider(provider),
                    gr.update(visible=(provider == "openai")),
                    gr.update(visible=(provider == "anthropic")),
                    gr.update(visible=(provider == "ollama")),
                    "Provider updated and saved to settings file.",
                ],
                inputs=provider_dropdown,
                outputs=[
                    provider_dropdown,
                    openai_model_dropdown,
                    anthropic_model_dropdown,
                    ollama_model_dropdown,
                    settings_status,
                ],
            )

            # Connect model dropdowns to update global variable
            openai_model_dropdown.change(
                fn=lambda m: [
                    update_global_model(m, "openai"),
                    "OpenAI model updated and saved to settings file.",
                ],
                inputs=openai_model_dropdown,
                outputs=[openai_model_dropdown, settings_status],
            )

            anthropic_model_dropdown.change(
                fn=lambda m: [
                    update_global_model(m, "anthropic"),
                    "Anthropic model updated and saved to settings file.",
                ],
                inputs=anthropic_model_dropdown,
                outputs=[anthropic_model_dropdown, settings_status],
            )

            ollama_model_dropdown.change(
                fn=lambda m: [
                    update_global_model(m, "ollama"),
                    "Ollama model updated and saved to settings file.",
                ],
                inputs=ollama_model_dropdown,
                outputs=[ollama_model_dropdown, settings_status],
            )

            # Connect max steps slider to update global variable
            max_steps_slider.change(
                fn=lambda steps: [
                    update_global_max_steps(steps),
                    "Maximum steps updated and saved to settings file.",
                ],
                inputs=max_steps_slider,
                outputs=[max_steps_slider, settings_status],
            )

            # Refresh Ollama models
            refresh_ollama_btn.click(
                fn=get_ollama_models,
                inputs=[],
                outputs=ollama_model_dropdown,
            )

        with gr.Tab("API Credentials"):
            gr.Markdown("# API Credentials")
            gr.Markdown(
                "Configure your API credentials for different LLM providers. These will be saved to the .env file."
            )

            # Get current credentials
            credentials = get_api_credentials()

            with gr.Group():
                gr.Markdown("## OpenAI")
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    value=credentials["OPENAI_API_KEY"],
                    type="password",
                )

            with gr.Group():
                gr.Markdown("## Anthropic")
                anthropic_api_key = gr.Textbox(
                    label="Anthropic API Key",
                    value=credentials["ANTHROPIC_API_KEY"],
                    type="password",
                )

            with gr.Group():
                gr.Markdown("## Other Services")
                serpapi_key = gr.Textbox(
                    label="SerpAPI Key (for Google Search Tool)",
                    value=credentials["SERPAPI_KEY"],
                    type="password",
                )

            save_btn = gr.Button("Save Credentials", variant="primary")
            result_message = gr.Markdown("")

            # Add save button functionality
            save_btn.click(
                fn=save_api_credentials,
                inputs=[
                    openai_api_key,
                    anthropic_api_key,
                    serpapi_key,
                ],
                outputs=result_message,
            )

    # Add custom JavaScript to set the height of the output container
    gr.HTML(
        """
    <script>
    /* No longer needed as we're using the native Chatbot component */
    </script>
    """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
