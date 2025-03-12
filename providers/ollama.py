"""
Ollama provider implementation.

This module provides an implementation of the LLMProvider interface for Ollama models
like Llama 3, Mistral, and similar open-source models.
"""

from providers.base import LLMProvider


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

    def is_finished(self, message):
        return "Final Answer" in message.get("content") or message.get("content") == ""

    def chat(self, messages, tools=[]):
        """
        Send a chat request to the Ollama API and return the response.

        If tools are provided and the model calls a tool, this method will
        execute the tool with the provided arguments and return both the
        model's message and the tool result.

        Args:
        messages: List of message dicts with "role" and "content" fields or message.get("content") == ""
            tools: List of Tool instances to be passed to the model

        Returns:
            List of message dicts including the model's response and any tool results
        """
        response = self.client.chat(
            model=self.model_name, messages=messages, tools=self.format_tools(tools)
        )

        # Check if the model wants to use a tool
        if "tool_calls" in response["message"]:
            for tool_call in response["message"]["tool_calls"]:
                print(f"Executing tool: {tool_call}")
                tool_result = self.execute_tool(tools, tool_call)
                print(f"Tool result: {tool_result}")

                tool_response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        *messages,
                        response["message"],
                        tool_result,
                    ],
                    tools=self.format_tools(tools),
                )
                print(f"Tool response: {tool_response}")

                return [response["message"], tool_result, tool_response["message"]]

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

    def execute_tool(self, tools: list, tool_call: dict) -> dict:
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
