"""
Base Tool class for defining external tools.

This module provides the foundation for all tool implementations in the CoT agent system.
"""


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
