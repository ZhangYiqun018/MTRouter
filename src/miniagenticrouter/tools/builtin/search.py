"""Web search tool using Serper API.

This tool performs web searches and returns formatted results.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

from miniagenticrouter.tools.base import BaseTool, ToolResult, ToolSchema
from miniagenticrouter.tools.registry import register_tool

logger = logging.getLogger(__name__)


@dataclass
class WebSearchConfig:
    """Configuration for WebSearchTool."""

    api_key: str = ""
    api_key_env: str = "SERPER_API_KEY"
    max_results: int = 10
    timeout: int = 30


@register_tool("search")
class WebSearchTool(BaseTool):
    """Search the web using Serper API.

    Supports batch queries for efficiency.
    Results are cached by default to avoid redundant API calls.

    Example:
        >>> tool = WebSearchTool()  # Uses SERPER_API_KEY env var
        >>> result = tool.execute(query="capital of France")
        >>> "Paris" in result.output
        True
    """

    name = "search"
    description = "Search the web for information using Google"

    # Enable caching - same query returns cached result
    cacheable = True
    cache_key_fields = ["query", "num_results"]

    def __init__(
        self,
        api_key: str = "",
        api_key_env: str = "SERPER_API_KEY",
        max_results: int = 10,
        timeout: int = 30,
        **kwargs,
    ) -> None:
        """Initialize the search tool.

        Args:
            api_key: Serper API key (optional if using env var).
            api_key_env: Environment variable name for API key.
            max_results: Maximum results per query.
            timeout: Request timeout in seconds.
        """
        self.config = WebSearchConfig(
            api_key=api_key,
            api_key_env=api_key_env,
            max_results=max_results,
            timeout=timeout,
        )
        self._api_key = api_key or os.getenv(api_key_env, "")

    def get_schema(self) -> ToolSchema:
        """Return the tool schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": "Search query or list of queries for batch search",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 10)",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
            },
            required=["query"],
            examples=[
                {"query": "latest AI developments 2024"},
                {"query": ["Python best practices", "asyncio tutorial"]},
                {"query": "machine learning", "num_results": 5},
            ],
        )

    def execute(
        self,
        query: str | list[str],
        num_results: int | None = None,
        **kwargs,
    ) -> ToolResult:
        """Execute web search.

        Args:
            query: Search query or list of queries.
            num_results: Number of results to return.

        Returns:
            ToolResult with formatted search results.
        """
        if not self._api_key:
            return ToolResult(
                output=f"Error: No API key configured. Set {self.config.api_key_env} environment variable.",
                returncode=1,
                error="no_api_key",
            )

        # Normalize to list
        queries = [query] if isinstance(query, str) else query
        max_results = num_results or self.config.max_results

        # Execute searches
        all_results = []
        for q in queries:
            result = self._search_serper(q, max_results)
            all_results.append(f"## Results for: {q}\n\n{result}")

        output = "\n\n---\n\n".join(all_results)

        return ToolResult(
            output=output,
            returncode=0,
            metadata={"queries": queries, "num_queries": len(queries)},
        )

    def _search_serper(self, query: str, num_results: int) -> str:
        """Perform a search using Serper API.

        Args:
            query: Search query.
            num_results: Number of results to return.

        Returns:
            Formatted search results as string.
        """
        import requests

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        data = {
            "q": query,
            "num": num_results,
        }

        try:
            response = requests.post(
                url,
                json=data,
                headers=headers,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            results = response.json()

            return self._format_results(results)

        except requests.RequestException as e:
            logger.error(f"Search request failed: {e}")
            return f"Search failed: {e}"

    def _format_results(self, results: dict[str, Any]) -> str:
        """Format search results as markdown.

        Args:
            results: Raw API response.

        Returns:
            Formatted markdown string.
        """
        lines = []

        # Knowledge Graph (if available)
        if "knowledgeGraph" in results:
            kg = results["knowledgeGraph"]
            lines.append("### Knowledge Graph")
            if "title" in kg:
                lines.append(f"**{kg['title']}**")
            if "description" in kg:
                lines.append(kg["description"])
            lines.append("")

        # Answer Box (if available)
        if "answerBox" in results:
            ab = results["answerBox"]
            lines.append("### Answer")
            if "answer" in ab:
                lines.append(ab["answer"])
            elif "snippet" in ab:
                lines.append(ab["snippet"])
            lines.append("")

        # Organic results
        organic = results.get("organic", [])
        if organic:
            lines.append("### Web Results")
            lines.append("")

            for i, item in enumerate(organic, 1):
                title = item.get("title", "No title")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                date = item.get("date", "")

                lines.append(f"{i}. **{title}**")
                if link:
                    lines.append(f"   URL: {link}")
                if date:
                    lines.append(f"   Date: {date}")
                if snippet:
                    lines.append(f"   {snippet}")
                lines.append("")

        # Related searches
        related = results.get("relatedSearches", [])
        if related:
            lines.append("### Related Searches")
            for item in related[:5]:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('query', item)}")
                else:
                    lines.append(f"- {item}")

        if not lines:
            return "No results found. Try a different or broader search query."

        return "\n".join(lines)

    def get_config(self) -> dict[str, Any]:
        """Return tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "max_results": self.config.max_results,
            "has_api_key": bool(self._api_key),
        }
