"""Tests for ScienceWorld environment adapter."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from miniagenticrouter.environments.extra.scienceworld import (
    ScienceWorldEnvironment,
    ScienceWorldEnvironmentConfig,
)


class MockScienceWorldEnv:
    """Mock ScienceWorld environment for testing."""

    def __init__(self, *args, **kwargs):
        self.task_name = None
        self.variation_idx = None
        self.taskDescription = "Test task description"
        self._score = 0.0
        self._valid_actions = ["look around", "go north", "pick up apple"]

    def load(self, task_name, variation_idx, simplificationStr=""):
        self.task_name = task_name
        self.variation_idx = variation_idx

    def reset(self):
        return "Initial observation", {}

    def step(self, action):
        obs = f"You executed: {action}"
        reward = 0.0
        done = action == "task completed"
        info = {"score": self._score}
        return obs, reward, done, info

    def getValidActionObjectCombinations(self):
        return self._valid_actions

    def getTaskNames(self):
        return ["boil", "melt-ice", "freeze"]

    def getScore(self):
        return self._score

    def close(self):
        pass


@pytest.fixture
def mock_scienceworld():
    """Fixture to mock the scienceworld import."""
    mock_module = MagicMock()
    mock_module.ScienceWorldEnv = MockScienceWorldEnv
    with patch.dict(sys.modules, {"scienceworld": mock_module}):
        yield mock_module


class TestScienceWorldEnvironmentConfig:
    """Test configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ScienceWorldEnvironmentConfig()
        assert config.task_name == ""
        assert config.variation_idx == 0
        assert config.simplification_str == ""

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ScienceWorldEnvironmentConfig(
            task_name="boil",
            variation_idx=5,
            simplification_str="noHints",
        )
        assert config.task_name == "boil"
        assert config.variation_idx == 5
        assert config.simplification_str == "noHints"


class TestScienceWorldEnvironmentWithMock:
    """Test ScienceWorld environment with mock."""

    def test_initialization(self, mock_scienceworld):
        """Test environment initialization."""
        env = ScienceWorldEnvironment()
        assert env._env is None  # Lazy initialization

    def test_load_task(self, mock_scienceworld):
        """Test loading a task."""
        env = ScienceWorldEnvironment()
        result = env.load_task("boil", 0)

        assert env.config.task_name == "boil"
        assert env.config.variation_idx == 0
        assert "observation" in result
        assert "task_description" in result

    def test_step(self, mock_scienceworld):
        """Test stepping through the environment."""
        env = ScienceWorldEnvironment(task_name="boil")
        env._ensure_env()  # Force initialization

        result = env.step("look around")

        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert result["done"] is False

    def test_step_done(self, mock_scienceworld):
        """Test step that completes the task."""
        env = ScienceWorldEnvironment(task_name="boil")
        env._ensure_env()

        result = env.step("task completed")

        assert result["done"] is True

    def test_execute_for_environment_protocol(self, mock_scienceworld):
        """Test execute method for Environment protocol compatibility."""
        env = ScienceWorldEnvironment(task_name="boil")
        env._ensure_env()

        result = env.execute("look around")

        assert "output" in result
        assert "returncode" in result
        assert result["returncode"] == 0

    def test_get_task_names(self, mock_scienceworld):
        """Test getting available task names."""
        env = ScienceWorldEnvironment()
        task_names = env.get_task_names()

        assert isinstance(task_names, list)
        assert "boil" in task_names

    def test_get_task_description(self, mock_scienceworld):
        """Test getting task description."""
        env = ScienceWorldEnvironment(task_name="boil")
        env._ensure_env()

        description = env.get_task_description()
        assert description == "Test task description"

    def test_get_valid_actions(self, mock_scienceworld):
        """Test getting valid actions."""
        env = ScienceWorldEnvironment(task_name="boil")
        env._ensure_env()

        actions = env.get_valid_actions()
        assert isinstance(actions, list)
        assert "look around" in actions

    def test_get_template_vars(self, mock_scienceworld):
        """Test getting template variables."""
        env = ScienceWorldEnvironment(task_name="boil")
        env._ensure_env()

        vars = env.get_template_vars()
        assert "task_name" in vars
        assert "task_description" in vars
        assert vars["task_name"] == "boil"

    def test_reset(self, mock_scienceworld):
        """Test resetting the environment."""
        env = ScienceWorldEnvironment(task_name="boil")
        env._ensure_env()

        result = env.reset()
        assert "observation" in result

    def test_close(self, mock_scienceworld):
        """Test closing the environment."""
        env = ScienceWorldEnvironment(task_name="boil")
        env._ensure_env()

        env.close()
        assert env._env is None


class TestScienceWorldEnvironmentImportError:
    """Test import error handling."""

    def test_import_error_message(self):
        """Test that ImportError has helpful message."""
        # Remove scienceworld from sys.modules if present
        with patch.dict(sys.modules, {"scienceworld": None}):
            env = ScienceWorldEnvironment()
            with pytest.raises(ImportError) as exc_info:
                env._ensure_env()

            assert "scienceworld package is required" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)


class TestScienceWorldEnvironmentLazyInit:
    """Test lazy initialization behavior."""

    def test_lazy_init_on_load_task(self, mock_scienceworld):
        """Test that environment is lazily initialized on load_task."""
        env = ScienceWorldEnvironment()
        assert env._env is None

        env.load_task("boil", 0)
        assert env._env is not None

    def test_lazy_init_on_step(self, mock_scienceworld):
        """Test that environment is lazily initialized on step."""
        env = ScienceWorldEnvironment(task_name="boil")
        assert env._env is None

        env.step("look")
        assert env._env is not None

    def test_lazy_init_on_execute(self, mock_scienceworld):
        """Test that environment is lazily initialized on execute."""
        env = ScienceWorldEnvironment(task_name="boil")
        assert env._env is None

        env.execute("look")
        assert env._env is not None
