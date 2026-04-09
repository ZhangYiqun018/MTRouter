"""Tests for FlexibleAgent."""

import pytest

from miniagenticrouter.agents.flexible import (
    FlexibleAgent,
    FlexibleAgentConfig,
    LimitsExceeded,
    NonTerminatingException,
    Submitted,
)
from miniagenticrouter.environments.local import LocalEnvironment
from miniagenticrouter.models.test_models import DeterministicModel
from miniagenticrouter.parsers.noaction import NoActionParser
from miniagenticrouter.parsers.regex import RegexActionParser
from miniagenticrouter.tools.bash import BashTool
from miniagenticrouter.tools.noop import NoOpTool


class TestFlexibleAgentBashMode:
    """Test FlexibleAgent in bash mode (default)."""

    def test_successful_completion(self):
        """Test agent completes successfully with bash commands."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=[
                    "Let me echo\n```bash\necho 'hello'\n```",
                    "Now finish\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
                ]
            ),
            env=LocalEnvironment(),
        )

        exit_status, result = agent.run("Echo hello and finish")
        assert exit_status == "Submitted"
        assert result == "done\n"
        assert agent.model.n_calls == 2

    def test_step_limit(self):
        """Test agent respects step limit."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=["```bash\necho 'test'\n```", "```bash\necho 'test2'\n```"]
            ),
            env=LocalEnvironment(),
            step_limit=1,
        )

        exit_status, _ = agent.run("Test")
        assert exit_status == "LimitsExceeded"
        assert agent.model.n_calls == 1

    def test_cost_limit(self):
        """Test agent respects cost limit."""
        agent = FlexibleAgent(
            model=DeterministicModel(outputs=["```bash\necho 'test'\n```"]),
            env=LocalEnvironment(),
            cost_limit=0.5,
        )

        exit_status, _ = agent.run("Test")
        assert exit_status == "LimitsExceeded"

    def test_format_error_recovery(self):
        """Test agent recovers from format errors."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=[
                    "No code block here",
                    "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'\n```",
                ]
            ),
            env=LocalEnvironment(),
        )

        exit_status, result = agent.run("Test format error")
        assert exit_status == "Submitted"
        assert result == "recovered\n"
        assert agent.model.n_calls == 2


class TestFlexibleAgentTextMode:
    """Test FlexibleAgent in text mode."""

    def test_text_mode_parsing(self):
        """Test text mode uses correct regex."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=[
                    "I'll look around\n```text\nlook around\n```",
                    "```text\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
                ]
            ),
            env=LocalEnvironment(),
            action_mode="text",
        )

        # Should parse text blocks instead of bash blocks
        exit_status, result = agent.run("Look around")
        assert exit_status == "Submitted"

    def test_text_mode_custom_regex(self):
        """Test text mode with custom regex in parser_config."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=[
                    "ACTION: do something",
                    "ACTION: echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'",
                ]
            ),
            env=LocalEnvironment(),
            action_mode="text",
            parser_config={
                "action_regex": r"ACTION:\s*(.+?)(?:\n|$)",
            },
        )

        exit_status, _ = agent.run("Do something")
        assert exit_status == "Submitted"


class TestFlexibleAgentNoneMode:
    """Test FlexibleAgent in none (conversation) mode."""

    def test_none_mode_no_env_required(self):
        """Test none mode doesn't require environment."""
        agent = FlexibleAgent(
            model=DeterministicModel(outputs=["Hello!"]),
            env=None,
            action_mode="none",
            step_limit=1,
        )

        # Should not raise, but will hit step limit
        exit_status, _ = agent.run("Say hello")
        assert exit_status == "LimitsExceeded"

    def test_none_mode_no_action_extraction(self):
        """Test none mode doesn't extract actions."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=[
                    "Here's some code: ```bash\nls -la\n``` but I won't execute it",
                ]
            ),
            env=None,
            action_mode="none",
            step_limit=1,
        )

        exit_status, _ = agent.run("Show code without executing")
        # Should complete without trying to execute
        assert exit_status == "LimitsExceeded"

    def test_none_mode_with_noop_tool(self):
        """Test none mode uses NoOpTool."""
        agent = FlexibleAgent(
            model=DeterministicModel(outputs=["Test"]),
            env=None,
            action_mode="none",
            step_limit=1,
        )

        assert isinstance(agent.tool, NoOpTool)
        assert isinstance(agent.parser, NoActionParser)


class TestFlexibleAgentCustomComponents:
    """Test FlexibleAgent with custom parsers and tools."""

    def test_custom_parser(self):
        """Test using a custom parser instance."""
        custom_parser = RegexActionParser(
            action_regex=r"\[CMD\](.*?)\[/CMD\]",
            action_type="custom",
        )
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=[
                    "[CMD]echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'custom'[/CMD]",
                ]
            ),
            env=LocalEnvironment(),
            parser=custom_parser,
        )

        exit_status, result = agent.run("Test custom parser")
        assert exit_status == "Submitted"
        assert result == "custom\n"

    def test_custom_tool(self):
        """Test using a custom tool instance."""
        custom_tool = NoOpTool(name="custom_noop")
        agent = FlexibleAgent(
            model=DeterministicModel(outputs=["```bash\necho 'test'\n```"]),
            env=LocalEnvironment(),
            tool=custom_tool,
            step_limit=1,
        )

        assert agent.tool.name == "custom_noop"


class TestFlexibleAgentFinishMarkers:
    """Test FlexibleAgent finish marker functionality."""

    def test_default_finish_markers(self):
        """Test default finish markers work."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=["```bash\necho 'MINI_SWE_AGENT_FINAL_OUTPUT'\necho 'done'\n```"]
            ),
            env=LocalEnvironment(),
        )

        exit_status, result = agent.run("Test finish")
        assert exit_status == "Submitted"
        assert result == "done\n"

    def test_custom_finish_markers(self):
        """Test custom finish markers."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=["```bash\necho 'TASK_COMPLETE'\necho 'custom done'\n```"]
            ),
            env=LocalEnvironment(),
            finish_markers=["TASK_COMPLETE"],
        )

        exit_status, result = agent.run("Test custom finish")
        assert exit_status == "Submitted"
        assert result == "custom done\n"

    def test_finish_on_action(self):
        """Test finish_on_action configuration."""
        agent = FlexibleAgent(
            model=DeterministicModel(outputs=["```bash\nsubmit\n```"]),
            env=LocalEnvironment(),
            finish_on_action="submit",
        )

        exit_status, result = agent.run("Submit the task")
        assert exit_status == "Submitted"
        assert result == "submit"


class TestFlexibleAgentConfig:
    """Test FlexibleAgent configuration."""

    def test_default_config(self):
        """Test default configuration."""
        agent = FlexibleAgent(
            model=DeterministicModel(outputs=[]),
            env=LocalEnvironment(),
        )

        assert agent.config.action_mode == "bash"
        assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in agent.config.finish_markers
        assert agent.config.step_limit == 0
        assert agent.config.cost_limit == 3.0

    def test_custom_templates(self):
        """Test custom templates."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=["```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```"]
            ),
            env=LocalEnvironment(),
            system_template="Custom system: {{task}}",
            instance_template="Do this: {{task}}",
        )

        agent.run("Test task")

        assert agent.messages[0]["content"] == "Custom system: Test task"
        assert agent.messages[1]["content"] == "Do this: Test task"

    def test_parser_config_override(self):
        """Test parser_config overrides mode preset."""
        agent = FlexibleAgent(
            model=DeterministicModel(outputs=[]),
            env=LocalEnvironment(),
            action_mode="bash",
            parser_config={
                "allow_no_action": True,
            },
        )

        # Parser should have allow_no_action from override
        config = agent.parser.get_config()
        assert config["allow_no_action"] is True
        # But should still have bash regex from preset
        assert "bash" in config["action_regex"]


class TestFlexibleAgentMessageHistory:
    """Test message history tracking."""

    def test_messages_have_timestamps(self):
        """Test all messages have timestamps."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=["```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```"]
            ),
            env=LocalEnvironment(),
        )

        agent.run("Test timestamps")

        assert all("timestamp" in msg for msg in agent.messages)
        assert all(isinstance(msg["timestamp"], float) for msg in agent.messages)

    def test_message_roles(self):
        """Test message roles are correct."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=[
                    "```bash\necho 'test'\n```",
                    "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
                ]
            ),
            env=LocalEnvironment(),
        )

        agent.run("Test")

        roles = [msg["role"] for msg in agent.messages]
        # system, user (task), assistant, user (observation), assistant, user (final)
        assert roles[:3] == ["system", "user", "assistant"]


class TestFlexibleAgentRenderTemplate:
    """Test template rendering."""

    def test_render_includes_parser_config(self):
        """Test render_template includes parser config."""
        agent = FlexibleAgent(
            model=DeterministicModel(outputs=[]),
            env=LocalEnvironment(),
        )
        agent.extra_template_vars["task"] = "test"

        # action_regex should be available from parser config
        result = agent.render_template("{{action_regex}}")
        assert "bash" in result

    def test_render_includes_tool_config(self):
        """Test render_template includes tool config."""
        agent = FlexibleAgent(
            model=DeterministicModel(outputs=[]),
            env=LocalEnvironment(),
        )
        agent.extra_template_vars["task"] = "test"

        # name should be available from tool config
        result = agent.render_template("{{name}}")
        assert result == "bash"


class TestFlexibleAgentBackwardCompatibility:
    """Test backward compatibility with DefaultAgent behavior."""

    def test_similar_to_default_agent(self):
        """Test FlexibleAgent behaves like DefaultAgent in bash mode."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=[
                    "I'll echo a message\n```bash\necho 'hello world'\n```",
                    "Now finishing\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'Task completed'\n```",
                ]
            ),
            env=LocalEnvironment(),
        )

        exit_status, result = agent.run("Echo hello world then finish")
        assert exit_status == "Submitted"
        assert result == "Task completed\n"
        assert agent.model.n_calls == 2

    def test_handles_multiple_code_blocks(self):
        """Test agent handles multiple code blocks with error."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=[
                    "```bash\necho 'first'\n```\n```bash\necho 'second'\n```",
                    "```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'\n```",
                ]
            ),
            env=LocalEnvironment(),
        )

        exit_status, result = agent.run("Test multiple blocks")
        assert exit_status == "Submitted"
        assert result == "recovered\n"


class TestFlexibleAgentConversationMode:
    """Test FlexibleAgent in pure conversation mode."""

    def test_conversation_no_action_extraction(self):
        """Test that conversation mode doesn't extract actions."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=[
                    "Hello! I'm happy to help.",
                    "Here's some code: ```python\nprint('hi')\n``` but I won't execute it.",
                    "Let me know if you have more questions!",
                ]
            ),
            env=None,
            action_mode="none",
            step_limit=3,
        )

        exit_status, _ = agent.run("Hello, can you help?")
        assert exit_status == "LimitsExceeded"
        assert agent.model.n_calls == 3

    def test_conversation_messages_preserved(self):
        """Test that conversation history is preserved."""
        agent = FlexibleAgent(
            model=DeterministicModel(
                outputs=[
                    "Hello!",
                    "Sure, I can help with that.",
                ]
            ),
            env=None,
            action_mode="none",
            step_limit=2,
        )

        agent.run("Hi there")

        # Should have system, user (task), assistant, assistant
        assert len(agent.messages) >= 3
        roles = [msg["role"] for msg in agent.messages]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_conversation_no_observations(self):
        """Test that conversation mode doesn't add observations."""
        agent = FlexibleAgent(
            model=DeterministicModel(outputs=["Response 1"]),
            env=None,
            action_mode="none",
            step_limit=1,
        )

        agent.run("Test")

        # Should not have any "Observation:" messages
        for msg in agent.messages:
            if msg["role"] == "user":
                assert "Observation:" not in msg.get("content", "")

    def test_conversation_with_noop_tool(self):
        """Test that conversation mode uses NoOpTool."""
        agent = FlexibleAgent(
            model=DeterministicModel(outputs=[]),
            env=None,
            action_mode="none",
        )

        assert isinstance(agent.tool, NoOpTool)
        assert isinstance(agent.parser, NoActionParser)
