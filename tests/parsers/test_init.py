"""Tests for parser factory functions."""

import pytest

from miniagenticrouter.parsers import get_parser, get_parser_class
from miniagenticrouter.parsers.noaction import NoActionParser
from miniagenticrouter.parsers.regex import RegexActionParser


class TestGetParserClass:
    """Test get_parser_class function."""

    def test_get_regex_parser_by_short_name(self):
        """Test getting RegexActionParser by short name."""
        cls = get_parser_class("regex")
        assert cls is RegexActionParser

    def test_get_noaction_parser_by_short_name(self):
        """Test getting NoActionParser by short name."""
        cls = get_parser_class("noaction")
        assert cls is NoActionParser

    def test_get_parser_by_full_path(self):
        """Test getting parser by full import path."""
        cls = get_parser_class("miniagenticrouter.parsers.regex.RegexActionParser")
        assert cls is RegexActionParser

    def test_get_parser_by_full_path_noaction(self):
        """Test getting NoActionParser by full import path."""
        cls = get_parser_class("miniagenticrouter.parsers.noaction.NoActionParser")
        assert cls is NoActionParser

    def test_unknown_parser_raises_value_error(self):
        """Test that unknown parser raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_parser_class("unknown_parser")
        assert "Unknown parser type" in str(exc_info.value)
        assert "unknown_parser" in str(exc_info.value)

    def test_invalid_path_raises_value_error(self):
        """Test that invalid import path raises ValueError."""
        with pytest.raises(ValueError):
            get_parser_class("invalid.module.path.Parser")


class TestGetParser:
    """Test get_parser factory function."""

    def test_get_parser_default(self):
        """Test getting default parser (regex)."""
        parser = get_parser()
        assert isinstance(parser, RegexActionParser)

    def test_get_parser_with_explicit_type(self):
        """Test getting parser with explicit type."""
        parser = get_parser({"parser_class": "regex"})
        assert isinstance(parser, RegexActionParser)

    def test_get_parser_noaction(self):
        """Test getting NoActionParser."""
        parser = get_parser({"parser_class": "noaction"})
        assert isinstance(parser, NoActionParser)

    def test_get_parser_with_custom_default(self):
        """Test getting parser with custom default type."""
        parser = get_parser(default_type="noaction")
        assert isinstance(parser, NoActionParser)

    def test_get_parser_with_config(self):
        """Test getting parser with configuration."""
        parser = get_parser({
            "parser_class": "regex",
            "action_regex": r"```python\s*\n(.*?)\n```",
            "action_type": "python",
        })
        assert isinstance(parser, RegexActionParser)
        config = parser.get_config()
        assert config["action_regex"] == r"```python\s*\n(.*?)\n```"
        assert config["action_type"] == "python"

    def test_get_parser_none_config(self):
        """Test getting parser with None config."""
        parser = get_parser(None)
        assert isinstance(parser, RegexActionParser)

    def test_get_parser_empty_config(self):
        """Test getting parser with empty config."""
        parser = get_parser({})
        assert isinstance(parser, RegexActionParser)

    def test_get_parser_config_not_mutated(self):
        """Test that original config dict is not mutated."""
        config = {
            "parser_class": "regex",
            "action_type": "custom",
        }
        original_config = config.copy()
        get_parser(config)
        assert config == original_config

    def test_get_parser_by_full_path(self):
        """Test getting parser by full import path."""
        parser = get_parser({
            "parser_class": "miniagenticrouter.parsers.noaction.NoActionParser",
        })
        assert isinstance(parser, NoActionParser)


class TestParserProtocol:
    """Test that parsers conform to ActionParser protocol."""

    def test_regex_parser_has_parse_method(self):
        """Test RegexActionParser has parse method."""
        parser = RegexActionParser()
        assert hasattr(parser, "parse")
        assert callable(parser.parse)

    def test_regex_parser_has_get_config_method(self):
        """Test RegexActionParser has get_config method."""
        parser = RegexActionParser()
        assert hasattr(parser, "get_config")
        assert callable(parser.get_config)

    def test_noaction_parser_has_parse_method(self):
        """Test NoActionParser has parse method."""
        parser = NoActionParser()
        assert hasattr(parser, "parse")
        assert callable(parser.parse)

    def test_noaction_parser_has_get_config_method(self):
        """Test NoActionParser has get_config method."""
        parser = NoActionParser()
        assert hasattr(parser, "get_config")
        assert callable(parser.get_config)

    def test_parse_returns_dict(self):
        """Test that parse returns a dict."""
        regex_parser = RegexActionParser()
        noaction_parser = NoActionParser()

        result1 = regex_parser.parse({"content": "```bash\nls\n```"})
        result2 = noaction_parser.parse({"content": "test"})

        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

    def test_get_config_returns_dict(self):
        """Test that get_config returns a dict."""
        regex_parser = RegexActionParser()
        noaction_parser = NoActionParser()

        assert isinstance(regex_parser.get_config(), dict)
        assert isinstance(noaction_parser.get_config(), dict)
