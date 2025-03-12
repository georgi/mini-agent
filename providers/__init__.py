"""
Providers module for the CoT Agent system.

This module provides a collection of language model providers that can be used
to interact with various LLM APIs including OpenAI, Anthropic, and Ollama.
"""

# Base provider class
from providers.base import LLMProvider

# OpenAI provider
from providers.openai import OpenAIProvider

# Anthropic provider
from providers.anthropic import AnthropicProvider

# Ollama provider
from providers.ollama import OllamaProvider

# Export all providers
__all__ = [
    # Base class
    "LLMProvider",
    # Provider implementations
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
]
