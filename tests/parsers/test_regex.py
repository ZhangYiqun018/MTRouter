"""Tests for RegexActionParser."""

import pytest

from miniagenticrouter.parsers.regex import FormatError, RegexActionParser, RegexActionParserConfig


class TestRegexActionParserBasic:
    """Test basic parsing functionality."""

    def test_parse_single_bash_block(self):
        """Test parsing a single bash code block."""
        parser = RegexActionParser()
        response = {"content": "Let me list files\n\n```bash\nls -la\n```"}
        result = parser.parse(response)

        assert result["action"] == "ls -la"
        assert result["action_type"] == "bash"
        assert result["content"] == response["content"]

    def test_parse_bash_block_with_surrounding_text(self):
        """Test parsing bash block with text before and after."""
        parser = RegexActionParser()
        response = {"content": "Some explanation\n```bash\necho 'hello'\n```\nMore text"}
        result = parser.parse(response)

        assert result["action"] == "echo 'hello'"
        assert result["action_type"] == "bash"

    def test_parse_multiline_command(self):
        """Test parsing multiline bash commands."""
        parser = RegexActionParser()
        response = {"content": "```bash\necho 'line1'\necho 'line2'\nls\n```"}
        result = parser.parse(response)

        assert result["action"] == "echo 'line1'\necho 'line2'\nls"
        assert result["action_type"] == "bash"

    def test_parse_strips_whitespace(self):
        """Test that action content is stripped of leading/trailing whitespace."""
        parser = RegexActionParser()
        response = {"content": "```bash\n  echo 'test'  \n```"}
        result = parser.parse(response)

        assert result["action"] == "echo 'test'"

    def test_preserves_original_response_fields(self):
        """Test that original response fields are preserved."""
        parser = RegexActionParser()
        response = {
            "content": "```bash\nls\n```",
            "model": "test-model",
            "tokens": 100,
            "custom_field": "value",
        }
        result = parser.parse(response)

        assert result["model"] == "test-model"
        assert result["tokens"] == 100
        assert result["custom_field"] == "value"


class TestRegexActionParserErrors:
    """Test error handling."""

    def test_no_action_raises_format_error(self):
        """Test that missing action raises FormatError."""
        parser = RegexActionParser()
        response = {"content": "No code blocks here"}

        with pytest.raises(FormatError) as exc_info:
            parser.parse(response)
        assert "found 0 actions" in str(exc_info.value)

    def test_multiple_actions_raises_format_error(self):
        """Test that multiple actions raise FormatError."""
        parser = RegexActionParser()
        response = {"content": "```bash\necho 'first'\n```\n```bash\necho 'second'\n```"}

        with pytest.raises(FormatError) as exc_info:
            parser.parse(response)
        assert "found 2 actions" in str(exc_info.value)

    def test_wrong_language_raises_format_error(self):
        """Test that non-bash code blocks raise FormatError."""
        parser = RegexActionParser()
        response = {"content": "```python\nprint('hello')\n```"}

        with pytest.raises(FormatError):
            parser.parse(response)

    def test_empty_content_raises_format_error(self):
        """Test that empty content raises FormatError."""
        parser = RegexActionParser()
        response = {"content": ""}

        with pytest.raises(FormatError):
            parser.parse(response)

    def test_missing_content_key_raises_format_error(self):
        """Test that missing content key raises FormatError."""
        parser = RegexActionParser()
        response = {}

        with pytest.raises(FormatError):
            parser.parse(response)


class TestRegexActionParserAllowNoAction:
    """Test allow_no_action configuration."""

    def test_allow_no_action_returns_none(self):
        """Test that allow_no_action=True returns None for missing action."""
        parser = RegexActionParser(allow_no_action=True)
        response = {"content": "Just a response without code"}
        result = parser.parse(response)

        assert result["action"] is None
        assert result["action_type"] == "none"
        assert result["content"] == response["content"]

    def test_allow_no_action_still_parses_valid_action(self):
        """Test that allow_no_action=True still parses valid actions."""
        parser = RegexActionParser(allow_no_action=True)
        response = {"content": "```bash\nls\n```"}
        result = parser.parse(response)

        assert result["action"] == "ls"
        assert result["action_type"] == "bash"


class TestRegexActionParserRequireSingleAction:
    """Test require_single_action configuration."""

    def test_require_single_action_false_returns_first(self):
        """Test that require_single_action=False returns first action."""
        parser = RegexActionParser(require_single_action=False)
        response = {"content": "```bash\necho 'first'\n```\n```bash\necho 'second'\n```"}
        result = parser.parse(response)

        assert result["action"] == "echo 'first'"
        assert result["action_type"] == "bash"
        assert result["all_actions"] == ["echo 'first'", "echo 'second'"]

    def test_require_single_action_false_no_action(self):
        """Test that require_single_action=False with no action returns None."""
        parser = RegexActionParser(require_single_action=False)
        response = {"content": "No code blocks"}
        result = parser.parse(response)

        assert result["action"] is None
        assert result["action_type"] == "none"
        assert result["all_actions"] == []


class TestRegexActionParserCustomRegex:
    """Test custom regex patterns."""

    def test_custom_regex_for_python(self):
        """Test custom regex for Python code blocks."""
        parser = RegexActionParser(
            action_regex=r"```python\s*\n(.*?)\n```",
            action_type="python",
        )
        response = {"content": "```python\nprint('hello')\n```"}
        result = parser.parse(response)

        assert result["action"] == "print('hello')"
        assert result["action_type"] == "python"

    def test_custom_regex_for_text_commands(self):
        """Test custom regex for text commands (ScienceWorld style)."""
        parser = RegexActionParser(
            action_regex=r"```text\s*\n(.*?)\n```",
            action_type="text",
        )
        response = {"content": "I'll look around\n```text\nlook around\n```"}
        result = parser.parse(response)

        assert result["action"] == "look around"
        assert result["action_type"] == "text"

    def test_custom_regex_with_capture_groups(self):
        """Test custom regex with different capture patterns."""
        parser = RegexActionParser(
            action_regex=r"ACTION:\s*(.+?)(?:\n|$)",
            action_type="custom",
        )
        response = {"content": "I will do this:\nACTION: do something"}
        result = parser.parse(response)

        assert result["action"] == "do something"
        assert result["action_type"] == "custom"


class TestRegexActionParserConfig:
    """Test configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        parser = RegexActionParser()
        config = parser.get_config()

        assert config["action_regex"] == r"```bash\s*\n(.*?)\n```"
        assert config["action_type"] == "bash"
        assert config["require_single_action"] is True
        assert config["allow_no_action"] is False

    def test_custom_config(self):
        """Test custom configuration values."""
        parser = RegexActionParser(
            action_regex=r"```python\s*\n(.*?)\n```",
            action_type="python",
            require_single_action=False,
            allow_no_action=True,
        )
        config = parser.get_config()

        assert config["action_regex"] == r"```python\s*\n(.*?)\n```"
        assert config["action_type"] == "python"
        assert config["require_single_action"] is False
        assert config["allow_no_action"] is True

    def test_custom_error_template(self):
        """Test custom error message template."""
        parser = RegexActionParser(
            format_error_template="Expected one action, got {count}",
        )
        response = {"content": "No actions"}

        with pytest.raises(FormatError) as exc_info:
            parser.parse(response)
        assert str(exc_info.value) == "Expected one action, got 0"

    def test_custom_config_class(self):
        """Test using a custom config class."""
        from dataclasses import dataclass

        @dataclass
        class CustomConfig(RegexActionParserConfig):
            custom_field: str = "custom_value"

        parser = RegexActionParser(config_class=CustomConfig)
        config = parser.get_config()

        assert config["custom_field"] == "custom_value"
        assert config["action_type"] == "bash"  # inherited default


class TestFormatError:
    """Test FormatError exception."""

    def test_format_error_is_exception(self):
        """Test that FormatError is an Exception."""
        assert issubclass(FormatError, Exception)

    def test_format_error_message(self):
        """Test FormatError message handling."""
        error = FormatError("Test error message")
        assert str(error) == "Test error message"
