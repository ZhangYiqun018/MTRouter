"""Tests for BashTool."""

import pytest

from miniagenticrouter.environments.local import LocalEnvironment
from miniagenticrouter.tools.bash import BashTool, BashToolConfig


class TestBashToolBasic:
    """Test basic execution functionality."""

    def test_execute_simple_command(self):
        """Test executing a simple echo command."""
        tool = BashTool(env=LocalEnvironment())
        result = tool.execute({"action": "echo 'hello'", "action_type": "bash"})

        assert result["output"] == "hello\n"
        assert result["returncode"] == 0
        assert result["done"] is False

    def test_execute_command_with_exit_code(self):
        """Test command that returns non-zero exit code."""
        tool = BashTool(env=LocalEnvironment())
        result = tool.execute({"action": "exit 1", "action_type": "bash"})

        assert result["returncode"] == 1

    def test_execute_multiline_command(self):
        """Test executing multiline bash commands."""
        tool = BashTool(env=LocalEnvironment())
        result = tool.execute({
            "action": "echo 'line1'\necho 'line2'",
            "action_type": "bash",
        })

        assert "line1" in result["output"]
        assert "line2" in result["output"]
        assert result["returncode"] == 0

    def test_execute_none_action(self):
        """Test executing None action returns empty result."""
        tool = BashTool(env=LocalEnvironment())
        result = tool.execute({"action": None, "action_type": "none"})

        assert result["output"] == ""
        assert result["returncode"] == 0
        assert result["done"] is False

    def test_execute_missing_action_key(self):
        """Test executing dict without action key returns empty result."""
        tool = BashTool(env=LocalEnvironment())
        result = tool.execute({"action_type": "bash"})

        assert result["output"] == ""
        assert result["returncode"] == 0

    def test_tool_name(self):
        """Test tool name property."""
        tool = BashTool(env=LocalEnvironment())
        assert tool.name == "bash"

    def test_custom_tool_name(self):
        """Test custom tool name."""
        tool = BashTool(env=LocalEnvironment(), name="custom_bash")
        assert tool.name == "custom_bash"


class TestBashToolConfig:
    """Test configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        tool = BashTool(env=LocalEnvironment())
        config = tool.get_config()

        assert config["name"] == "bash"
        assert config["cwd"] == ""

    def test_custom_cwd(self):
        """Test custom working directory configuration."""
        tool = BashTool(env=LocalEnvironment(), cwd="/tmp")
        config = tool.get_config()

        assert config["cwd"] == "/tmp"

    def test_custom_config_class(self):
        """Test using a custom config class."""
        from dataclasses import dataclass

        @dataclass
        class CustomConfig(BashToolConfig):
            custom_field: str = "custom_value"

        tool = BashTool(env=LocalEnvironment(), config_class=CustomConfig)
        config = tool.get_config()

        assert config["custom_field"] == "custom_value"
        assert config["name"] == "bash"


class TestBashToolEnvironmentIntegration:
    """Test integration with different environments."""

    def test_with_local_environment(self):
        """Test BashTool with LocalEnvironment."""
        env = LocalEnvironment()
        tool = BashTool(env=env)
        result = tool.execute({"action": "pwd", "action_type": "bash"})

        assert result["returncode"] == 0
        assert len(result["output"]) > 0

    def test_with_timeout_environment(self):
        """Test BashTool propagates environment timeout.

        Note: LocalEnvironment raises TimeoutExpired exception on timeout.
        The agent layer is responsible for handling this exception.
        """
        import subprocess

        env = LocalEnvironment(timeout=1)
        tool = BashTool(env=env)

        # LocalEnvironment raises TimeoutExpired on timeout
        with pytest.raises(subprocess.TimeoutExpired):
            tool.execute({"action": "sleep 5", "action_type": "bash"})


class TestBashToolProtocol:
    """Test that BashTool conforms to Tool protocol."""

    def test_has_name_attribute(self):
        """Test BashTool has name attribute."""
        tool = BashTool(env=LocalEnvironment())
        assert hasattr(tool, "name")
        assert isinstance(tool.name, str)

    def test_has_execute_method(self):
        """Test BashTool has execute method."""
        tool = BashTool(env=LocalEnvironment())
        assert hasattr(tool, "execute")
        assert callable(tool.execute)

    def test_has_get_config_method(self):
        """Test BashTool has get_config method."""
        tool = BashTool(env=LocalEnvironment())
        assert hasattr(tool, "get_config")
        assert callable(tool.get_config)

    def test_execute_returns_dict(self):
        """Test that execute returns a dict."""
        tool = BashTool(env=LocalEnvironment())
        result = tool.execute({"action": "echo 'test'", "action_type": "bash"})

        assert isinstance(result, dict)
        assert "output" in result
        assert "returncode" in result

    def test_get_config_returns_dict(self):
        """Test that get_config returns a dict."""
        tool = BashTool(env=LocalEnvironment())
        config = tool.get_config()

        assert isinstance(config, dict)
