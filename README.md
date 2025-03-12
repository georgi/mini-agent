# 🧠 Chain of Thought (CoT) Agent

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-awesome-orange)

> **A flexible, multi-provider Chain of Thought reasoning framework with tool-calling capabilities**

## 🚀 Overview

**CoT Agent** is a powerful framework that enables large language models (LLMs) to solve complex problems through step-by-step reasoning. By implementing Chain of Thought (CoT) methodology, models can break down problems, show their work, and leverage external tools to achieve more accurate and reliable results.

```
📝 Problem → 🤔 Step-by-Step Reasoning → 🔧 Tool Usage → 💡 Final Answer
```

This implementation supports multiple LLM providers (OpenAI, Anthropic, Ollama) and includes a range of tools that extend the capabilities of language models beyond simple text generation.

## ✨ Key Features

- **Multi-Provider Support**: Works with OpenAI (GPT-4), Anthropic (Claude), and Ollama (Llama, Mistral) models
- **Tool Integration**: Extends LLM capabilities with various tools:
  - 🔢 Calculator for mathematical operations
  - 🌐 Browser control for web interactions
  - 💻 Shell command execution
  - 📂 File system operations
  - 🔍 Google search integration
- **Step-by-Step Reasoning**: Encourages models to break down problems into logical steps
- **Modular Architecture**: Easily extend with custom tools and LLM providers
- **Interactive Problem Solving**: Watch the agent's reasoning process unfold in real-time

## 📋 Requirements

- Python 3.8+
- Required Python packages:
  - For OpenAI: `openai`
  - For Anthropic: `anthropic`
  - For Ollama: `ollama`
  - For browser functionality: `playwright`
  - Utility packages: `python-dotenv`, `requests`

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cot-agent.git
cd cot-agent

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (create a .env file with your API keys)
echo "OPENAI_API_KEY=your_openai_key" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_key" >> .env
echo "SERPAPI_KEY=your_serpapi_key" >> .env
```

## 🛠️ Usage

### Basic Example

```python
from cot_agent import CoTAgent, OpenAIProvider, CalculatorTool

# Initialize provider and agent
provider = OpenAIProvider(model_name="gpt-4o")
calculator = CalculatorTool()

# Create agent with tools
agent = CoTAgent(
    provider=provider,
    tools=[calculator],
    max_steps=5
)

# Solve a problem
answer = agent.solve_problem("What is the square root of 144 plus the cube root of 27?")
print(f"Final Answer: {answer}")
```

### Command-Line Interface

The module can be run directly as a script:

```bash
# Using environment variables for configuration
export LLM_PROVIDER=openai
export OPENAI_MODEL=gpt-4o

# Run the agent with a problem
python cot_agent.py "If a train travels at 120 km/h, how long will it take to travel 450 km?"
```

## 🧩 Architecture

The CoT Agent is built with a modular architecture consisting of:

1. **LLM Providers**: Abstract interface for different LLM services
   - `OpenAIProvider`: For GPT models
   - `AnthropicProvider`: For Claude models
   - `OllamaProvider`: For open-source models
2. **Tools**: Extensible set of external capabilities

   - `CalculatorTool`: Mathematical operations
   - `BrowserTool`: Web browsing and interaction
   - `ExecuteShellTool`: Shell command execution
   - `FileReadTool`: File system operations
   - `GoogleSearchTool`: Web search integration

3. **CoTAgent**: Core reasoning engine
   - Manages conversation context
   - Handles tool calling
   - Implements step-by-step reasoning flow

## 🌟 Advanced Examples

### Multi-Tool Problem Solving

```python
from cot_agent import (
    CoTAgent, AnthropicProvider, CalculatorTool,
    GoogleSearchTool, FileReadTool
)

# Initialize provider with Claude
provider = AnthropicProvider(model_name="claude-3-7-sonnet-20250219")

# Create agent with multiple tools
agent = CoTAgent(
    provider=provider,
    tools=[CalculatorTool(), GoogleSearchTool(), FileReadTool()],
    max_steps=10
)

# Solve a complex problem requiring multiple tools
problem = """
Analyze the growth rate of Tesla's stock price over the last year.
Calculate the average monthly growth rate and compare it to the S&P 500 index.
If the data shows Tesla outperforming the index, explain possible reasons.
"""

answer = agent.solve_problem(problem)
print(f"Analysis Results:\n{answer}")
```

### Custom Tool Creation

```python
from cot_agent import Tool, CoTAgent, OllamaProvider

# Create a custom weather tool
class WeatherTool(Tool):
    def __init__(self):
        super().__init__(
            name="weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    }
                },
                "required": ["location"]
            }
        )

    def execute(self, location):
        # Implement weather API call
        # This is a simplified example
        return f"Weather data for {location}: Sunny, 25°C"

# Use the custom tool with Ollama
provider = OllamaProvider(model_name="llama3.2")
agent = CoTAgent(provider=provider, tools=[WeatherTool()])
```

## 🔄 Supported LLM Providers

| Provider  | Models                       | Features                                     |
| --------- | ---------------------------- | -------------------------------------------- |
| OpenAI    | GPT-3.5-Turbo, GPT-4o        | Advanced tool use, function calling          |
| Anthropic | Claude 3 Opus, Sonnet, Haiku | High-quality reasoning, multi-content blocks |
| Ollama    | Llama 3, Mistral, Vicuna     | Local deployment, no API key needed          |

## 🙌 Contributing

Contributions are welcome! Feel free to:

- 🐛 Report bugs and issues
- ✨ Suggest new features or enhancements
- 🛠️ Submit pull requests with improvements
- 📖 Improve documentation

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Roadmap

- [ ] Support for multi-modal inputs (images, audio)
- [ ] Memory management for long-running sessions
- [ ] Agent self-improvement and meta-cognition
- [ ] Parallel tool execution for efficiency
- [ ] Web interface for interactive use

---

<p align="center">
  <b>Built with 💻 code and 🧠 intelligence</b>
</p>
