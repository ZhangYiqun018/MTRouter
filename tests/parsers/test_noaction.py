"""Tests for NoActionParser."""

import pytest

from miniagenticrouter.parsers.noaction import NoActionParser, NoActionParserConfig


class TestNoActionParserBasic:
    """Test basic parsing functionality."""

    def test_parse_returns_none_action(self):
        """Test that parse always returns None for action."""
        parser = NoActionParser()
        response = {"content": "Hello, how can I help you?"}
        result = parser.parse(response)

        assert result["action"] is None
        assert result["action_type"] == "none"
        assert result["content"] == response["content"]

    def test_parse_with_code_block_still_returns_none(self):
        """Test that even content with code blocks returns None action."""
        parser = NoActionParser()
        response = {"content": "Here's some code\n```bash\nls -la\n```"}
        result = parser.parse(response)

        assert result["action"] is None
        assert result["action_type"] == "none"

    def test_parse_empty_content(self):
        """Test parsing empty content."""
        parser = NoActionParser()
        response = {"content": ""}
        result = parser.parse(response)

        assert result["action"] is None
        assert result["action_type"] == "none"
        assert result["content"] == ""

    def test_parse_preserves_response_fields(self):
        """Test that original response fields are preserved."""
        parser = NoActionParser()
        response = {
            "content": "Test response",
            "model": "test-model",
            "tokens": 50,
            "metadata": {"key": "value"},
        }
        result = parser.parse(response)

        assert result["action"] is None
        assert result["action_type"] == "none"
        assert result["content"] == "Test response"
        assert result["model"] == "test-model"
        assert result["tokens"] == 50
        assert result["metadata"] == {"key": "value"}

    def test_parse_missing_content_key(self):
        """Test parsing response without content key."""
        parser = NoActionParser()
        response = {"other_field": "value"}
        result = parser.parse(response)

        assert result["action"] is None
        assert result["action_type"] == "none"
        assert result["other_field"] == "value"

    def test_parse_empty_response(self):
        """Test parsing empty response dict."""
        parser = NoActionParser()
        response = {}
        result = parser.parse(response)

        assert result["action"] is None
        assert result["action_type"] == "none"


class TestNoActionParserConfig:
    """Test configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        parser = NoActionParser()
        config = parser.get_config()

        assert config["action_type"] == "none"

    def test_custom_action_type(self):
        """Test custom action_type configuration."""
        parser = NoActionParser(action_type="custom_none")
        config = parser.get_config()
        result = parser.parse({"content": "test"})

        assert config["action_type"] == "custom_none"
        assert result["action_type"] == "custom_none"

    def test_custom_config_class(self):
        """Test using a custom config class."""
        from dataclasses import dataclass

        @dataclass
        class CustomConfig(NoActionParserConfig):
            custom_field: str = "custom_value"

        parser = NoActionParser(config_class=CustomConfig)
        config = parser.get_config()

        assert config["custom_field"] == "custom_value"
        assert config["action_type"] == "none"  # inherited default


class TestNoActionParserConversation:
    """Test NoActionParser in conversation scenarios."""

    def test_multi_turn_conversation(self):
        """Test parsing multiple turns of conversation."""
        parser = NoActionParser()

        turns = [
            {"content": "What is Python?"},
            {"content": "Python is a programming language."},
            {"content": "Can you give an example?"},
            {"content": "Here's an example:\n```python\nprint('Hello')\n```"},
        ]

        for turn in turns:
            result = parser.parse(turn)
            assert result["action"] is None
            assert result["action_type"] == "none"
            assert result["content"] == turn["content"]

    def test_various_content_types(self):
        """Test parsing various types of content."""
        parser = NoActionParser()

        contents = [
            "Simple text",
            "Text with\nnewlines\n",
            "Text with special chars: @#$%^&*()",
            "Unicode: 你好世界 🌍",
            "Very " + "long " * 1000 + "text",
        ]

        for content in contents:
            result = parser.parse({"content": content})
            assert result["action"] is None
            assert result["action_type"] == "none"
            assert result["content"] == content
