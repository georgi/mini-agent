"""
Anthropic provider implementation.

This module provides an implementation of the LLMProvider interface for Anthropic Claude models.
"""

import os
from anthropic.types.tool_use_block import ToolUseBlock
from anthropic.types.tool_param import ToolParam
from providers.base import LLMProvider


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

    def is_finished(self, message):
        return message.get("content") == [] or any(
            "Final Answer" in block.get("text", "") for block in message.get("content")
        )

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

    def format_tools(self, tools: list) -> list[ToolParam]:
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
