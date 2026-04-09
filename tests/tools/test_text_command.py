"""Tests for TextCommandTool."""

import pytest

from miniagenticrouter.tools.text_command import TextCommandTool, TextCommandToolConfig


class MockTextEnvironment:
    """Mock text environment for testing."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.last_command = None

    def step(self, command: str) -> dict:
        self.last_command = command
        if self.call_count < len(self.responses):
            result = self.responses[self.call_count]
            self.call_count += 1
            return result
        return {"observation": "Default response", "done": False, "reward": 0}


class TestTextCommandToolBasic:
    """Test basic execution functionality."""

    def test_execute_returns_observation(self):
        """Test that execute returns observation from environment."""
        env = MockTextEnvironment(responses=[
            {"observation": "You are in a kitchen.", "done": False, "reward": 0}
        ])
        tool = TextCommandTool(env=env)
        result = tool.execute({"action": "look around", "action_type": "text"})

        assert result["output"] == "You are in a kitchen."
        assert result["returncode"] == 0
        assert result["done"] is False
        assert env.last_command == "look around"

    def test_execute_with_reward(self):
        """Test that execute includes reward from environment."""
        env = MockTextEnvironment(responses=[
            {"observation": "Task complete!", "done": True, "reward": 1.0}
        ])
        tool = TextCommandTool(env=env)
        result = tool.execute({"action": "finish task", "action_type": "text"})

        assert result["reward"] == 1.0
        assert result["done"] is True

    def test_execute_none_action(self):
        """Test executing None action returns empty result."""
        env = MockTextEnvironment()
        tool = TextCommandTool(env=env)
        result = tool.execute({"action": None, "action_type": "none"})

        assert result["output"] == ""
        assert result["returncode"] == 0
        assert result["done"] is False
        assert env.call_count == 0

    def test_execute_missing_action_key(self):
        """Test executing dict without action key returns empty result."""
        env = MockTextEnvironment()
        tool = TextCommandTool(env=env)
        result = tool.execute({"action_type": "text"})

        assert result["output"] == ""
        assert result["returncode"] == 0

    def test_tool_name(self):
        """Test tool name property."""
        env = MockTextEnvironment()
        tool = TextCommandTool(env=env)
        assert tool.name == "text_command"

    def test_custom_tool_name(self):
        """Test custom tool name."""
        env = MockTextEnvironment()
        tool = TextCommandTool(env=env, name="custom_text")
        assert tool.name == "custom_text"


class TestTextCommandToolConfig:
    """Test configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        env = MockTextEnvironment()
        tool = TextCommandTool(env=env)
        config = tool.get_config()

        assert config["name"] == "text_command"

    def test_custom_config_class(self):
        """Test using a custom config class."""
        from dataclasses import dataclass

        @dataclass
        class CustomConfig(TextCommandToolConfig):
            custom_field: str = "custom_value"

        env = MockTextEnvironment()
        tool = TextCommandTool(env=env, config_class=CustomConfig)
        config = tool.get_config()

        assert config["custom_field"] == "custom_value"
        assert config["name"] == "text_command"


class TestTextCommandToolProtocol:
    """Test that TextCommandTool conforms to Tool protocol."""

    def test_has_name_attribute(self):
        """Test TextCommandTool has name attribute."""
        env = MockTextEnvironment()
        tool = TextCommandTool(env=env)
        assert hasattr(tool, "name")
        assert isinstance(tool.name, str)

    def test_has_execute_method(self):
        """Test TextCommandTool has execute method."""
        env = MockTextEnvironment()
        tool = TextCommandTool(env=env)
        assert hasattr(tool, "execute")
        assert callable(tool.execute)

    def test_has_get_config_method(self):
        """Test TextCommandTool has get_config method."""
        env = MockTextEnvironment()
        tool = TextCommandTool(env=env)
        assert hasattr(tool, "get_config")
        assert callable(tool.get_config)

    def test_execute_returns_dict(self):
        """Test that execute returns a dict."""
        env = MockTextEnvironment(responses=[
            {"observation": "test", "done": False, "reward": 0}
        ])
        tool = TextCommandTool(env=env)
        result = tool.execute({"action": "test", "action_type": "text"})

        assert isinstance(result, dict)
        assert "output" in result
        assert "returncode" in result

    def test_get_config_returns_dict(self):
        """Test that get_config returns a dict."""
        env = MockTextEnvironment()
        tool = TextCommandTool(env=env)
        config = tool.get_config()

        assert isinstance(config, dict)


class TestTextCommandToolMultiStep:
    """Test multi-step interactions."""

    def test_multiple_steps(self):
        """Test executing multiple commands in sequence."""
        env = MockTextEnvironment(responses=[
            {"observation": "Room 1", "done": False, "reward": 0},
            {"observation": "Room 2", "done": False, "reward": 0.5},
            {"observation": "Goal!", "done": True, "reward": 1.0},
        ])
        tool = TextCommandTool(env=env)

        r1 = tool.execute({"action": "go north", "action_type": "text"})
        assert r1["output"] == "Room 1"
        assert r1["done"] is False

        r2 = tool.execute({"action": "go east", "action_type": "text"})
        assert r2["output"] == "Room 2"
        assert r2["reward"] == 0.5

        r3 = tool.execute({"action": "finish", "action_type": "text"})
        assert r3["output"] == "Goal!"
        assert r3["done"] is True
        assert r3["reward"] == 1.0
