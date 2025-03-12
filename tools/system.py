"""
System interaction tools.

This module provides tools for interacting with the operating system, such as
executing shell commands and reading files.
"""

import json
import subprocess
from tools.base import Tool


class ExecuteShellTool(Tool):
    """
    A tool that executes shell commands and returns their output.

    This tool enables language models to interact with the operating system
    by executing shell commands. It returns the stdout, stderr, and return code
    of the executed command, with a timeout to prevent hanging.

    Note: This tool should be used with caution as it can potentially execute
    dangerous commands with system-wide effects.
    """

    def __init__(self):
        """
        Initialize the ExecuteShellTool with its name, description, and parameter schema.

        Sets up the tool with a parameter schema that requires a command string
        to be executed in the shell.
        """
        super().__init__(
            name="execute_shell",
            description="Execute a shell command and return its output (use with caution)",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                },
                "required": ["command"],
            },
        )

    def execute(self, command, timeout=10):
        """
        Executes a shell command and returns the result.

        Runs the specified command in a shell environment with a timeout
        to prevent hanging processes. Returns a JSON object containing
        the command's success status, return code, stdout, and stderr.

        Args:
            command (str): Shell command to execute
            timeout (int, optional): Maximum execution time in seconds. Defaults to 10.

        Returns:
            str: JSON string containing command execution results or error information
        """
        try:
            import subprocess
            from pathlib import Path

            # Create a process with timeout
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return json.dumps(
                {
                    "success": process.returncode == 0,
                    "return_code": process.returncode,
                    "stdout": process.stdout.strip(),
                    "stderr": process.stderr.strip(),
                }
            )

        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout} seconds"}
        except Exception as e:
            return {"error": str(e)}


class FileReadTool(Tool):
    """
    A tool that reads files from the filesystem and returns their contents.

    This tool allows language models to read files from the local filesystem,
    with options to limit file size, specify encoding, and list directory contents.
    It includes safety checks to prevent reading sensitive system files.
    """

    def __init__(self):
        """
        Initialize the FileReadTool with its name, description, and parameter schema.

        Sets up the tool with a parameter schema that defines file reading options
        including the file path, maximum size to read, encoding, and whether to list
        directory contents instead of reading a file.
        """
        super().__init__(
            name="file_read",
            description="Read files from the filesystem",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "max_size": {
                        "type": "integer",
                        "description": "Maximum number of bytes to read (default: 100KB)",
                        "default": 102400,
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding (default: utf-8)",
                        "default": "utf-8",
                    },
                    "list_dir": {
                        "type": "boolean",
                        "description": "If true and path is a directory, list its contents instead of reading",
                        "default": False,
                    },
                },
                "required": ["path"],
            },
        )

    def execute(self, path, max_size=102400, encoding="utf-8", list_dir=False):
        """
        Reads a file or lists directory contents.

        Safely reads a file from the filesystem, with size limits and encoding options.
        Can alternatively list the contents of a directory. Includes safety checks
        to prevent reading sensitive system files.

        Args:
            path (str): Path to the file or directory to read
            max_size (int, optional): Maximum file size in bytes to read. Defaults to 102400 (100KB).
            encoding (str, optional): File encoding to use. Defaults to "utf-8".
            list_dir (bool, optional): If True and path is a directory, list its contents. Defaults to False.

        Returns:
            str: JSON string containing file contents or directory listing or error information
        """
        try:
            from pathlib import Path

            file_path = Path(path).expanduser().resolve()

            # Security check - prevent reading sensitive system files
            sensitive_paths = ["/etc/passwd", "/etc/shadow", "/etc/sudoers", "/etc/ssh"]
            for sensitive in sensitive_paths:
                if str(file_path).startswith(sensitive):
                    return json.dumps(
                        {
                            "error": f"Access denied: Cannot read sensitive system file {file_path}"
                        }
                    )

            # Check if path exists
            if not file_path.exists():
                return json.dumps({"error": f"Path does not exist: {file_path}"})

            # Handle directory listing
            if file_path.is_dir():
                if list_dir:
                    dir_contents = []
                    for item in file_path.iterdir():
                        dir_contents.append(
                            {
                                "name": item.name,
                                "type": "directory" if item.is_dir() else "file",
                                "size": item.stat().st_size if item.is_file() else None,
                                "modified": item.stat().st_mtime,
                            }
                        )
                    return json.dumps(
                        {
                            "is_directory": True,
                            "path": str(file_path),
                            "contents": dir_contents,
                        }
                    )
                else:
                    return json.dumps(
                        {
                            "error": f"Path is a directory: {file_path}. Set list_dir=true to list contents."
                        }
                    )

            # Handle file reading
            if not file_path.is_file():
                return json.dumps({"error": f"Path is not a regular file: {file_path}"})

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > max_size:
                return json.dumps(
                    {
                        "error": f"File too large: {file_size} bytes (max: {max_size} bytes)"
                    }
                )

            # Read file content
            content = file_path.read_text(encoding=encoding)

            return json.dumps(
                {"path": str(file_path), "size": file_size, "content": content}
            )

        except UnicodeDecodeError:
            return json.dumps(
                {
                    "error": f"Cannot decode file with encoding {encoding}. The file might be binary."
                }
            )
        except PermissionError:
            return json.dumps({"error": f"Permission denied: Cannot read {path}"})
        except Exception as e:
            return json.dumps({"error": f"Error reading file: {str(e)}"})
