"""
OpenAI provider implementation.

This module provides an implementation of the LLMProvider interface for OpenAI models
like GPT-3.5-Turbo and GPT-4.
"""

import json
import os
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from providers.base import LLMProvider


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

    def is_finished(self, message):
        return message.get("content") == "" or "Final answer" in message.get("content")

    def chat(self, messages, tools: list = []):
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
        message = response.choices[0].message.model_dump()

        # Check for tool calls - try both attribute and dictionary access
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = message["tool_calls"]

        # Handle tool calls if present
        if tool_calls:
            for tool_call in tool_calls:
                # Print tool call in real time
                print(f"Tool call: {tool_call}")
                print("-" * 50)

                # Execute the tool
                tool_result = self.execute_tool(tools, tool_call)
                print(tool_result)
                print("-" * 50)

                # Call the LLM again with the tool result
                tool_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        *messages,
                        message,
                        tool_result,
                    ],  # type: ignore
                )

                # Get the second response from OpenAI
                tool_response_message = tool_response.choices[0].message
                tool_response_dict = (
                    tool_response_message.model_dump()
                    if hasattr(tool_response_message, "model_dump")
                    else tool_response_message
                )

                return [message, tool_result, tool_response_dict]

        # If no tool calls, return the message as a list
        return [message]

    def format_tools(self, tools: list = []) -> list[ChatCompletionToolParam]:
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

    def execute_tool(self, tools: list, tool_call) -> dict:
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
