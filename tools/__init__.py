"""
Tools module for the CoT Agent system.

This module provides a collection of tools that can be used by language models
to perform various tasks including calculations, web browsing, file operations,
and accessing external services like Google Search.
"""

# Base tool class
from tools.base import Tool

# Calculation tools
from tools.calculation import CalculatorTool

# Browser tools
from tools.browser import BrowserTool

# System tools
from tools.system import ExecuteShellTool, FileReadTool

# Google service tools
from tools.google import (
    GoogleSearchTool,
    GoogleFinanceTool,
    GoogleFlightsTool,
    GoogleNewsTool,
)

# Export all tools
__all__ = [
    # Base class
    "Tool",
    # Calculation tools
    "CalculatorTool",
    # Browser tools
    "BrowserTool",
    # System tools
    "ExecuteShellTool",
    "FileReadTool",
    # Google tools
    "GoogleSearchTool",
    "GoogleFinanceTool",
    "GoogleFlightsTool",
    "GoogleNewsTool",
]
