"""Built-in tools for multi-tool agent.

This module auto-imports all builtin tools to trigger @register_tool decorators.

Available tools:
- answer: Submit final answer (triggers completion)
- bash: Execute bash commands
- browse: Fetch and extract web page content
- file_read: Read file contents
- python: Execute Python code (sandbox)
- search: Web search via Serper API
"""

# Import all tools to trigger registration
from miniagenticrouter.tools.builtin.answer import AnswerTool
from miniagenticrouter.tools.builtin.bash import BashTool
from miniagenticrouter.tools.builtin.browse import WebBrowseTool
from miniagenticrouter.tools.builtin.file_read import FileReadTool
from miniagenticrouter.tools.builtin.python import PythonTool
from miniagenticrouter.tools.builtin.search import WebSearchTool

__all__ = [
    "AnswerTool",
    "BashTool",
    "WebBrowseTool",
    "FileReadTool",
    "PythonTool",
    "WebSearchTool",
]
