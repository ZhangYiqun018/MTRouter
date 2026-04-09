"""Tests for tool factory functions."""

import pytest

from miniagenticrouter.environments.local import LocalEnvironment
from miniagenticrouter.tools import get_tool, get_tool_class
from miniagenticrouter.tools.bash import BashTool
from miniagenticrouter.tools.noop import NoOpTool


class TestGetToolClass:
    """Test get_tool_class function."""

    def test_get_bash_tool_by_short_name(self):
        """Test getting BashTool by short name."""
        cls = get_tool_class("bash")
        assert cls is BashTool

    def test_get_noop_tool_by_short_name(self):
        """Test getting NoOpTool by short name."""
        cls = get_tool_class("noop")
        assert cls is NoOpTool

    def test_get_tool_by_full_path(self):
        """Test getting tool by full import path."""
        cls = get_tool_class("miniagenticrouter.tools.bash.BashTool")
        assert cls is BashTool

    def test_get_tool_by_full_path_noop(self):
        """Test getting NoOpTool by full import path."""
        cls = get_tool_class("miniagenticrouter.tools.noop.NoOpTool")
        assert cls is NoOpTool

    def test_unknown_tool_raises_value_error(self):
        """Test that unknown tool raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_tool_class("unknown_tool")
        assert "Unknown tool type" in str(exc_info.value)
        assert "unknown_tool" in str(exc_info.value)

    def test_invalid_path_raises_value_error(self):
        """Test that invalid import path raises ValueError."""
        with pytest.raises(ValueError):
            get_tool_class("invalid.module.path.Tool")


class TestGetTool:
    """Test get_tool factory function."""

    def test_get_tool_default_requires_env(self):
        """Test that default (bash) tool requires env."""
        # BashTool requires env argument
        with pytest.raises(TypeError):
            get_tool()

    def test_get_tool_bash_with_env(self):
        """Test getting BashTool with environment."""
        env = LocalEnvironment()
        tool = get_tool(env=env)
        assert isinstance(tool, BashTool)

    def test_get_tool_noop_no_env(self):
        """Test getting NoOpTool without environment."""
        tool = get_tool({"tool_class": "noop"})
        assert isinstance(tool, NoOpTool)

    def test_get_tool_with_explicit_type(self):
        """Test getting tool with explicit type."""
        env = LocalEnvironment()
        tool = get_tool({"tool_class": "bash"}, env=env)
        assert isinstance(tool, BashTool)

    def test_get_tool_with_custom_default(self):
        """Test getting tool with custom default type."""
        tool = get_tool(default_type="noop")
        assert isinstance(tool, NoOpTool)

    def test_get_tool_with_config(self):
        """Test getting tool with configuration."""
        env = LocalEnvironment()
        tool = get_tool({
            "tool_class": "bash",
            "name": "custom_bash",
        }, env=env)
        assert isinstance(tool, BashTool)
        assert tool.name == "custom_bash"

    def test_get_tool_none_config(self):
        """Test getting tool with None config and explicit kwargs."""
        env = LocalEnvironment()
        tool = get_tool(None, env=env)
        assert isinstance(tool, BashTool)

    def test_get_tool_empty_config(self):
        """Test getting tool with empty config and explicit kwargs."""
        env = LocalEnvironment()
        tool = get_tool({}, env=env)
        assert isinstance(tool, BashTool)

    def test_get_tool_config_not_mutated(self):
        """Test that original config dict is not mutated."""
        config = {
            "tool_class": "noop",
            "name": "custom",
        }
        original_config = config.copy()
        get_tool(config)
        assert config == original_config

    def test_get_tool_by_full_path(self):
        """Test getting tool by full import path."""
        tool = get_tool({
            "tool_class": "miniagenticrouter.tools.noop.NoOpTool",
        })
        assert isinstance(tool, NoOpTool)


class TestToolProtocol:
    """Test that tools conform to Tool protocol."""

    def test_bash_tool_has_name(self):
        """Test BashTool has name attribute."""
        tool = BashTool(env=LocalEnvironment())
        assert hasattr(tool, "name")
        assert isinstance(tool.name, str)

    def test_noop_tool_has_name(self):
        """Test NoOpTool has name attribute."""
        tool = NoOpTool()
        assert hasattr(tool, "name")
        assert isinstance(tool.name, str)

    def test_bash_tool_has_execute(self):
        """Test BashTool has execute method."""
        tool = BashTool(env=LocalEnvironment())
        assert hasattr(tool, "execute")
        assert callable(tool.execute)

    def test_noop_tool_has_execute(self):
        """Test NoOpTool has execute method."""
        tool = NoOpTool()
        assert hasattr(tool, "execute")
        assert callable(tool.execute)

    def test_bash_tool_has_get_config(self):
        """Test BashTool has get_config method."""
        tool = BashTool(env=LocalEnvironment())
        assert hasattr(tool, "get_config")
        assert callable(tool.get_config)

    def test_noop_tool_has_get_config(self):
        """Test NoOpTool has get_config method."""
        tool = NoOpTool()
        assert hasattr(tool, "get_config")
        assert callable(tool.get_config)

    def test_execute_returns_dict(self):
        """Test that execute returns a dict for all tools."""
        bash_tool = BashTool(env=LocalEnvironment())
        noop_tool = NoOpTool()

        result1 = bash_tool.execute({"action": "echo 'test'", "action_type": "bash"})
        result2 = noop_tool.execute({"action": "test", "action_type": "none"})

        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

    def test_get_config_returns_dict(self):
        """Test that get_config returns a dict for all tools."""
        bash_tool = BashTool(env=LocalEnvironment())
        noop_tool = NoOpTool()

        assert isinstance(bash_tool.get_config(), dict)
        assert isinstance(noop_tool.get_config(), dict)
