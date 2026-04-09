"""Answer tool for submitting final answers.

This tool signals task completion when invoked.
"""

from dataclasses import dataclass

from miniagenticrouter.tools.base import BaseTool, ToolResult, ToolSchema
from miniagenticrouter.tools.registry import register_tool


@dataclass
class AnswerToolConfig:
    """Configuration for AnswerTool."""

    pass  # No config needed


@register_tool("answer")
class AnswerTool(BaseTool):
    """Submit the final answer to complete the task.

    When this tool is called, the agent will stop and return the answer.

    Example:
        >>> tool = AnswerTool()
        >>> result = tool.execute(answer="42", reasoning="The answer to everything")
        >>> result.done
        True
        >>> result.metadata["answer"]
        '42'
    """

    name = "answer"
    description = "Submit your final answer to complete the task"

    def __init__(self, **kwargs) -> None:
        """Initialize the answer tool."""
        pass

    def get_schema(self) -> ToolSchema:
        """Return the tool schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to submit",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence level from 0 to 1 (optional)",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of how you arrived at the answer (optional)",
                    },
                },
            },
            required=["answer"],
            examples=[
                {"answer": "Paris"},
                {"answer": "42", "confidence": 0.95, "reasoning": "Calculated using the formula"},
            ],
        )

    def execute(
        self,
        answer: str,
        confidence: float = 1.0,
        reasoning: str = "",
        **kwargs,
    ) -> ToolResult:
        """Submit the final answer.

        Args:
            answer: The final answer to submit.
            confidence: Confidence level (0-1).
            reasoning: Explanation of the answer.

        Returns:
            ToolResult with done=True to signal completion.
        """
        output_parts = [f"Final Answer: {answer}"]

        if reasoning:
            output_parts.append(f"Reasoning: {reasoning}")

        if confidence < 1.0:
            output_parts.append(f"Confidence: {confidence:.0%}")

        return ToolResult(
            output="\n".join(output_parts),
            returncode=0,
            done=True,  # Signal task completion
            metadata={
                "answer": answer,
                "confidence": confidence,
                "reasoning": reasoning,
            },
        )
