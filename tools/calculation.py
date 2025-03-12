"""
Mathematical calculation tools.

This module provides tools for performing mathematical calculations.
"""

from tools.base import Tool


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
