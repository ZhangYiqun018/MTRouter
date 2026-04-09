"""Tests for NoOpTool."""

import pytest

from miniagenticrouter.tools.noop import NoOpTool, NoOpToolConfig


class TestNoOpToolBasic:
    """Test basic functionality."""

    def test_execute_returns_empty_result(self):
        """Test that execute returns empty result."""
        tool = NoOpTool()
        result = tool.execute({"action": "anything", "action_type": "any"})

        assert result["output"] == ""
        assert result["returncode"] == 0
        assert result["done"] is False

    def test_execute_ignores_action(self):
        """Test that execute ignores the action content."""
        tool = NoOpTool()

        # All these should return the same result
        results = [
            tool.execute({"action": "echo 'hello'", "action_type": "bash"}),
            tool.execute({"action": None, "action_type": "none"}),
            tool.execute({"action": "rm -rf /", "action_type": "dangerous"}),
            tool.execute({}),
        ]

        for result in results:
            assert result["output"] == ""
            assert result["returncode"] == 0
            assert result["done"] is False

    def test_tool_name(self):
        """Test tool name property."""
        tool = NoOpTool()
        assert tool.name == "noop"

    def test_custom_tool_name(self):
        """Test custom tool name."""
        tool = NoOpTool(name="custom_noop")
        assert tool.name == "custom_noop"


class TestNoOpToolConfig:
    """Test configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        tool = NoOpTool()
        config = tool.get_config()

        assert config["name"] == "noop"

    def test_custom_config_class(self):
        """Test using a custom config class."""
        from dataclasses import dataclass

        @dataclass
        class CustomConfig(NoOpToolConfig):
            custom_field: str = "custom_value"

        tool = NoOpTool(config_class=CustomConfig)
        config = tool.get_config()

        assert config["custom_field"] == "custom_value"
        assert config["name"] == "noop"


class TestNoOpToolProtocol:
    """Test that NoOpTool conforms to Tool protocol."""

    def test_has_name_attribute(self):
        """Test NoOpTool has name attribute."""
        tool = NoOpTool()
        assert hasattr(tool, "name")
        assert isinstance(tool.name, str)

    def test_has_execute_method(self):
        """Test NoOpTool has execute method."""
        tool = NoOpTool()
        assert hasattr(tool, "execute")
        assert callable(tool.execute)

    def test_has_get_config_method(self):
        """Test NoOpTool has get_config method."""
        tool = NoOpTool()
        assert hasattr(tool, "get_config")
        assert callable(tool.get_config)

    def test_execute_returns_dict(self):
        """Test that execute returns a dict."""
        tool = NoOpTool()
        result = tool.execute({"action": "test", "action_type": "test"})

        assert isinstance(result, dict)
        assert "output" in result
        assert "returncode" in result

    def test_get_config_returns_dict(self):
        """Test that get_config returns a dict."""
        tool = NoOpTool()
        config = tool.get_config()

        assert isinstance(config, dict)


class TestNoOpToolConversationMode:
    """Test NoOpTool in conversation scenarios."""

    def test_multiple_executions(self):
        """Test multiple executions return consistent results."""
        tool = NoOpTool()

        for i in range(10):
            result = tool.execute({"action": f"action_{i}", "action_type": "none"})
            assert result["output"] == ""
            assert result["returncode"] == 0
            assert result["done"] is False
