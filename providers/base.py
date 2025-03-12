"""
Base provider class for language model services.

This module provides the foundation for all LLM provider implementations in the CoT agent system.
It defines a common interface that all providers must implement.
"""

from abc import ABC, abstractmethod


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
