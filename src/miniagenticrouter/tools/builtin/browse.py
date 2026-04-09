"""Web browsing tool for fetching and extracting page content.

This tool fetches web pages and extracts their content in various formats.
Supports LLM-based summarization aligned with DeepResearch's tool_visit.py.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from miniagenticrouter.tools.base import BaseTool, ToolResult, ToolSchema
from miniagenticrouter.tools.registry import register_tool

logger = logging.getLogger(__name__)

# Prompt template aligned with DeepResearch
EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**"""


@dataclass
class WebBrowseConfig:
    """Configuration for WebBrowseTool."""

    # Basic settings
    max_content_length: int = 50000
    timeout: int = 30
    use_jina: bool = False
    jina_api_key: str | None = None
    user_agent: str = "Mozilla/5.0 (compatible; mini-agentic-router/1.0)"

    # LLM summarization settings (aligned with DeepResearch)
    use_llm_summary: bool = False
    summary_model: str | None = None
    summary_model_kwargs: dict[str, Any] = field(default_factory=dict)
    max_summary_tokens: int = 95000


@register_tool("browse")
class WebBrowseTool(BaseTool):
    """Fetch and extract content from web pages.

    Supports multiple extraction modes:
    - text: Plain text extraction
    - markdown: Convert HTML to markdown
    - html: Raw HTML (truncated)

    When use_llm_summary is enabled, uses an LLM to extract relevant
    information based on the provided goal (aligned with DeepResearch).

    Results are cached by URL to avoid redundant fetches.

    Example:
        >>> tool = WebBrowseTool()
        >>> result = tool.execute(url="https://example.com", goal="Find the main topic")
        >>> "Example Domain" in result.output
        True
    """

    name = "browse"
    description = "Visit webpage(s) and extract information based on a goal"

    # Enable caching - same URL + goal returns cached result
    cacheable = True
    cache_key_fields = ["url", "goal"]

    def should_cache_result(self, result: ToolResult, **kwargs) -> bool:
        """Only cache successful results with meaningful content."""
        if result.returncode != 0:
            return False
        # Don't cache empty or very short responses
        if not result.output or len(result.output) < 100:
            return False
        return True

    def __init__(
        self,
        max_content_length: int = 50000,
        timeout: int = 30,
        use_jina: bool = False,
        jina_api_key_env: str = "JINA_API_KEY",
        use_llm_summary: bool = False,
        summary_model: str | None = None,
        summary_model_kwargs: dict[str, Any] | None = None,
        max_summary_tokens: int = 95000,
        **kwargs,
    ) -> None:
        """Initialize the browse tool.

        Args:
            max_content_length: Maximum content length to return.
            timeout: Request timeout in seconds.
            use_jina: Whether to use Jina Reader API for content extraction.
            jina_api_key_env: Environment variable name for Jina API key.
            use_llm_summary: Whether to use LLM for summarization.
            summary_model: Model name for summarization.
            summary_model_kwargs: Additional kwargs for the summary model.
            max_summary_tokens: Maximum tokens for content before summarization.
        """
        import os

        jina_api_key = os.environ.get(jina_api_key_env)

        self.config = WebBrowseConfig(
            max_content_length=max_content_length,
            timeout=timeout,
            use_jina=use_jina,
            jina_api_key=jina_api_key,
            use_llm_summary=use_llm_summary,
            summary_model=summary_model,
            summary_model_kwargs=summary_model_kwargs or {},
            max_summary_tokens=max_summary_tokens,
        )

    def get_schema(self) -> ToolSchema:
        """Return the tool schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to visit. Can be a single URL or array of URLs.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "The goal of the visit - what information to extract from the webpage(s).",
                    },
                    "extract_mode": {
                        "type": "string",
                        "enum": ["text", "markdown", "html"],
                        "description": "Content extraction mode (default: text). Only used when LLM summary is disabled.",
                    },
                },
            },
            required=["url", "goal"],
            examples=[
                {
                    "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                    "goal": "Find the history and main features of Python",
                },
                {
                    "url": ["https://example.com", "https://example.org"],
                    "goal": "Compare the content of these two pages",
                },
            ],
        )

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Override to allow url to be string or list.

        The base class would convert list to string, which is incorrect for URLs.
        """
        # Save original url value before base class validation
        url_value = args.get("url")

        # Call parent validation (may convert url to string)
        validated = super().validate_args(args)

        # Restore url to its original type (string or list)
        if url_value is not None:
            validated["url"] = url_value

        return validated

    def execute(
        self,
        url: str | list[str],
        goal: str | None = None,
        extract_mode: str = "text",
        **kwargs,
    ) -> ToolResult:
        """Fetch and extract content from URL(s).

        Args:
            url: URL(s) to fetch. Can be a single URL string or list of URLs.
            goal: The goal of the visit - what information to extract.
            extract_mode: Extraction mode (text/markdown/html) when LLM summary is disabled.

        Returns:
            ToolResult with extracted content or LLM summary.
        """
        # Handle single URL or list of URLs
        urls = [url] if isinstance(url, str) else list(url)

        # Normalize URLs
        urls = [u if u.startswith(("http://", "https://")) else f"https://{u}" for u in urls]

        # Fetch all URLs
        all_contents = []
        all_metadata = []

        for u in urls:
            if self.config.use_jina:
                result = self._fetch_via_jina(u)
            else:
                result = self._fetch_direct(u, extract_mode)

            if result.returncode == 0:
                all_contents.append(f"=== Content from {u} ===\n{result.output}")
                all_metadata.append(result.metadata)
            else:
                all_contents.append(f"=== Error fetching {u} ===\n{result.output}")

        # Combine all content
        combined_content = "\n\n".join(all_contents)

        # If LLM summary is enabled and goal is provided, summarize with LLM
        if self.config.use_llm_summary and goal:
            return self._summarize_with_llm(combined_content, goal, all_metadata)

        # Otherwise return raw content
        return ToolResult(
            output=combined_content[: self.config.max_content_length],
            returncode=0,
            metadata={
                "urls": urls,
                "num_urls": len(urls),
                "total_length": len(combined_content),
            },
        )

    def _truncate_tokens(self, content: str, max_tokens: int) -> str:
        """Truncate content to max_tokens using tiktoken.

        Args:
            content: Content to truncate.
            max_tokens: Maximum number of tokens.

        Returns:
            Truncated content string.
        """
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(content)

            if len(tokens) <= max_tokens:
                return content

            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)

        except ImportError:
            # Fallback: rough estimate of 4 chars per token
            char_limit = max_tokens * 4
            if len(content) <= char_limit:
                return content
            return content[:char_limit]

    def _summarize_with_llm(
        self,
        content: str,
        goal: str,
        metadata: list[dict[str, Any]],
    ) -> ToolResult:
        """Summarize content using LLM based on goal.

        Aligned with DeepResearch's tool_visit.py implementation.

        Args:
            content: Combined webpage content.
            goal: The goal of extraction.
            metadata: Metadata from fetched pages.

        Returns:
            ToolResult with extracted evidence and summary.
        """
        from miniagenticrouter.models import get_model

        # Truncate content to max tokens
        content = self._truncate_tokens(content, self.config.max_summary_tokens)

        # Build prompt
        prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)

        # Get or create model (same pattern as LLMJudge)
        model_name = self.config.summary_model or "deepseek/deepseek-v3.2"
        model_kwargs = {"temperature": 0.7, **self.config.summary_model_kwargs}
        config = {"model_kwargs": model_kwargs} if model_kwargs else {}
        model = get_model(model_name, config)

        # Retry logic same as DeepResearch: re-call LLM on parse failure
        max_retries = 3
        last_error = None
        response = ""

        for attempt in range(max_retries):
            try:
                # Query the model (same pattern as LLMJudge)
                result = model.query([{"role": "user", "content": prompt}])
                response = result.get("content", "")

                # Clean up markdown code blocks if present
                if response.startswith("```json"):
                    response = response.replace("```json", "").replace("```", "").strip()

                # Parse JSON response (same approach as DeepResearch)
                # Find first '{' and last '}' to extract JSON
                left = response.find('{')
                right = response.rfind('}')
                if left != -1 and right != -1 and right > left:
                    json_str = response[left:right + 1]
                    parsed_result = json.loads(json_str)
                else:
                    parsed_result = json.loads(response)

                # Only return summary (same as DeepResearch)
                summary = parsed_result.get("summary", "")
                if not summary:
                    # Fallback: use evidence or full response
                    summary = parsed_result.get("evidence", response)

                return ToolResult(
                    output=summary,
                    returncode=0,
                    metadata={
                        "goal": goal,
                        "summary_model": model_name,
                        "parsed_result": parsed_result,
                        "page_metadata": metadata,
                    },
                )

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"JSON parse attempt {attempt + 1}/{max_retries} failed: {e}")
                # DeepResearch: retry LLM call on parse failure
                continue

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt + 1}/{max_retries} failed: {e}")
                continue

        # All retries failed - return formatted error (same as DeepResearch)
        error_msg = (
            f"The useful information for user goal '{goal}' as follows:\n\n"
            f"Evidence in page:\n"
            f"The provided webpage content could not be processed. "
            f"Error: {last_error}"
        )
        return ToolResult(
            output=error_msg,
            returncode=1,
            metadata={
                "goal": goal,
                "summary_model": model_name,
                "parse_error": str(last_error),
                "page_metadata": metadata,
            },
        )

    def _fetch_via_jina(self, url: str) -> ToolResult:
        """Fetch content using Jina Reader API.

        Args:
            url: URL to fetch.

        Returns:
            ToolResult with markdown content.
        """
        import requests

        jina_url = f"https://r.jina.ai/{url}"

        # Build headers
        headers = {"User-Agent": self.config.user_agent}
        if self.config.jina_api_key:
            headers["Authorization"] = f"Bearer {self.config.jina_api_key}"

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    jina_url,
                    timeout=self.config.timeout,
                    headers=headers,
                )
                response.raise_for_status()

                content = response.text[: self.config.max_content_length]

                return ToolResult(
                    output=content,
                    returncode=0,
                    metadata={"url": url, "via_jina": True, "length": len(content)},
                )

            except requests.RequestException as e:
                last_error = e
                logger.warning(f"Jina fetch attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    import time

                    time.sleep(1)  # Brief delay before retry

        logger.error(f"Jina fetch failed after {max_retries} attempts: {last_error}")
        # Fall back to direct fetch
        return self._fetch_direct(url, "text")

    def _fetch_direct(self, url: str, extract_mode: str) -> ToolResult:
        """Fetch content directly.

        Args:
            url: URL to fetch.
            extract_mode: Extraction mode.

        Returns:
            ToolResult with extracted content.
        """
        import requests

        try:
            response = requests.get(
                url,
                timeout=self.config.timeout,
                headers={
                    "User-Agent": self.config.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                allow_redirects=True,
            )
            response.raise_for_status()

            html = response.text
            content = self._extract_content(html, extract_mode)

            return ToolResult(
                output=content[: self.config.max_content_length],
                returncode=0,
                metadata={
                    "url": url,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "length": len(content),
                },
            )

        except requests.RequestException as e:
            return ToolResult(
                output=f"Failed to fetch {url}: {e}",
                returncode=1,
                error=str(e),
            )

    def _extract_content(self, html: str, mode: str) -> str:
        """Extract content from HTML.

        Args:
            html: Raw HTML content.
            mode: Extraction mode.

        Returns:
            Extracted content string.
        """
        if mode == "html":
            return html

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            if mode == "markdown":
                return self._html_to_markdown(soup)
            else:
                return soup.get_text(separator="\n", strip=True)

        except ImportError:
            # Fallback without BeautifulSoup
            return self._extract_text_fallback(html)

    def _html_to_markdown(self, soup) -> str:
        """Convert BeautifulSoup object to markdown.

        Args:
            soup: BeautifulSoup object.

        Returns:
            Markdown string.
        """
        lines = []

        # Get title
        title = soup.find("title")
        if title:
            lines.append(f"# {title.get_text(strip=True)}")
            lines.append("")

        # Process main content
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code"]):
            if tag.name.startswith("h"):
                level = int(tag.name[1])
                text = tag.get_text(strip=True)
                if text:
                    lines.append(f"{'#' * level} {text}")
                    lines.append("")
            elif tag.name == "p":
                text = tag.get_text(strip=True)
                if text:
                    lines.append(text)
                    lines.append("")
            elif tag.name == "li":
                text = tag.get_text(strip=True)
                if text:
                    lines.append(f"- {text}")
            elif tag.name in ("pre", "code"):
                text = tag.get_text(strip=True)
                if text:
                    lines.append("```")
                    lines.append(text)
                    lines.append("```")
                    lines.append("")

        return "\n".join(lines)

    def _extract_text_fallback(self, html: str) -> str:
        """Extract text from HTML without BeautifulSoup.

        Args:
            html: Raw HTML content.

        Returns:
            Extracted text.
        """
        # Remove scripts and styles
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Remove tags
        text = re.sub(r"<[^>]+>", " ", html)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

        return text

    def get_config(self) -> dict[str, Any]:
        """Return tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "max_content_length": self.config.max_content_length,
            "use_jina": self.config.use_jina,
            "use_llm_summary": self.config.use_llm_summary,
            "summary_model": self.config.summary_model,
        }
