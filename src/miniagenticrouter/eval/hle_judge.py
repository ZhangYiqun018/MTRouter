"""HLE (Humanity's Last Exam) answer evaluation.

This module provides evaluation utilities for HLE benchmark answers,
following the official evaluation methodology from centerforaisafety/hle.

Official HLE uses LLM-as-a-Judge for answer evaluation. This module provides:
1. Rule-based evaluation (fast, deterministic) - HLEJudge
2. LLM-based evaluation (semantic matching, higher accuracy) - LLMJudge
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Regex patterns for answer extraction
ANSWER_PATTERNS = [
    # Tool call format: <tool_call>{"name": "answer", "arguments": {"answer": "..."}}
    r'<tool_call>\s*\{[^}]*"name"\s*:\s*"answer"[^}]*"arguments"\s*:\s*\{[^}]*"answer"\s*:\s*"([^"]+)"',
    # Answer: format
    r"(?:^|\n)\s*(?:Final\s+)?Answer\s*:\s*(.+?)(?:\n|$)",
    # <answer>...</answer> format
    r"<answer>\s*(.+?)\s*</answer>",
    # **Answer:** format (markdown)
    r"\*\*(?:Final\s+)?Answer\*\*\s*:\s*(.+?)(?:\n|$)",
]


@dataclass
class JudgeResult:
    """Result from evaluating a single answer."""

    task_id: str
    prediction: str | None
    ground_truth: str
    correct: bool
    confidence: float = 1.0
    reasoning: str = ""
    method: str = "rule_based"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "correct": self.correct,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "method": self.method,
        }


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""

    total: int = 0
    correct: int = 0
    incorrect: int = 0
    no_answer: int = 0
    results: list[JudgeResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy (correct / total)."""
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate ((total - no_answer) / total)."""
        return (self.total - self.no_answer) / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "no_answer": self.no_answer,
            "accuracy": self.accuracy,
            "completion_rate": self.completion_rate,
        }


class HLEJudge:
    """HLE answer evaluator.

    Provides both rule-based and LLM-based evaluation methods.

    Example:
        >>> judge = HLEJudge()
        >>> result = judge.judge("Paris", "Paris", "task_001")
        >>> result.correct
        True

        >>> # Evaluate a batch from results file
        >>> summary = judge.evaluate_results(Path("results.json"))
        >>> print(f"Accuracy: {summary.accuracy:.2%}")
    """

    def __init__(
        self,
        normalize: bool = True,
        case_sensitive: bool = False,
    ):
        """Initialize the judge.

        Args:
            normalize: Whether to normalize answers before comparison.
            case_sensitive: Whether comparison is case-sensitive.
        """
        self.normalize = normalize
        self.case_sensitive = case_sensitive

    def extract_answer(self, response: str) -> str | None:
        """Extract answer from model response.

        Tries multiple patterns to extract the final answer.

        Args:
            response: Full model response text.

        Returns:
            Extracted answer string, or None if not found.
        """
        if not response:
            return None

        # Try each pattern in order
        for pattern in ANSWER_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # Clean up common artifacts
                answer = re.sub(r"^\s*['\"]|['\"]\s*$", "", answer)
                return answer

        # If no pattern matched, try to get the last non-empty line
        # (often the answer in short responses)
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        if lines:
            last_line = lines[-1]
            # Only use if it's reasonably short (likely an answer, not explanation)
            if len(last_line) < 200:
                return last_line

        return None

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison.

        Args:
            answer: Raw answer string.

        Returns:
            Normalized answer string.
        """
        if not answer:
            return ""

        normalized = answer.strip()

        # Case normalization
        if not self.case_sensitive:
            normalized = normalized.lower()

        # Remove common punctuation at the end
        normalized = re.sub(r"[.,;:!?]+$", "", normalized)

        # Normalize whitespace
        normalized = " ".join(normalized.split())

        # Remove quotes
        normalized = re.sub(r'^["\']|["\']$', "", normalized)

        # Normalize numbers (remove trailing zeros, handle scientific notation)
        try:
            # Try to parse as float for numeric comparison
            num = float(normalized.replace(",", ""))
            normalized = f"{num:g}"  # Remove trailing zeros
        except (ValueError, TypeError):
            pass

        return normalized

    def judge(
        self,
        prediction: str | None,
        ground_truth: str,
        task_id: str = "",
    ) -> JudgeResult:
        """Judge whether prediction matches ground truth.

        Args:
            prediction: Model's predicted answer (or None if no answer).
            ground_truth: Ground truth answer.
            task_id: Task identifier for result tracking.

        Returns:
            JudgeResult with correctness and metadata.
        """
        if prediction is None:
            return JudgeResult(
                task_id=task_id,
                prediction=None,
                ground_truth=ground_truth,
                correct=False,
                confidence=1.0,
                reasoning="No answer extracted from response",
            )

        # Normalize if enabled
        pred_normalized = self.normalize_answer(prediction) if self.normalize else prediction
        gt_normalized = self.normalize_answer(ground_truth) if self.normalize else ground_truth

        # Direct match
        if pred_normalized == gt_normalized:
            return JudgeResult(
                task_id=task_id,
                prediction=prediction,
                ground_truth=ground_truth,
                correct=True,
                confidence=1.0,
                reasoning="Exact match after normalization",
            )

        # Check if prediction contains ground truth (for longer answers)
        if gt_normalized in pred_normalized or pred_normalized in gt_normalized:
            return JudgeResult(
                task_id=task_id,
                prediction=prediction,
                ground_truth=ground_truth,
                correct=True,
                confidence=0.9,
                reasoning="Substring match",
            )

        # Multiple choice: check if single letter matches
        if len(gt_normalized) == 1 and gt_normalized.isalpha():
            pred_letter = re.search(r"\b([A-Ha-h])\b", prediction)
            if pred_letter and pred_letter.group(1).lower() == gt_normalized.lower():
                return JudgeResult(
                    task_id=task_id,
                    prediction=prediction,
                    ground_truth=ground_truth,
                    correct=True,
                    confidence=0.95,
                    reasoning="Multiple choice letter match",
                )

        # Numeric comparison with tolerance
        try:
            pred_num = float(pred_normalized.replace(",", ""))
            gt_num = float(gt_normalized.replace(",", ""))
            if abs(pred_num - gt_num) < 1e-6 * max(abs(gt_num), 1):
                return JudgeResult(
                    task_id=task_id,
                    prediction=prediction,
                    ground_truth=ground_truth,
                    correct=True,
                    confidence=0.95,
                    reasoning="Numeric match within tolerance",
                )
        except (ValueError, TypeError):
            pass

        # No match found
        return JudgeResult(
            task_id=task_id,
            prediction=prediction,
            ground_truth=ground_truth,
            correct=False,
            confidence=1.0,
            reasoning="No match found",
        )

    def judge_from_response(
        self,
        response: str,
        ground_truth: str,
        task_id: str = "",
    ) -> JudgeResult:
        """Judge a full response by first extracting the answer.

        Args:
            response: Full model response text.
            ground_truth: Ground truth answer.
            task_id: Task identifier.

        Returns:
            JudgeResult with correctness and metadata.
        """
        prediction = self.extract_answer(response)
        result = self.judge(prediction, ground_truth, task_id)
        if prediction is None:
            result.reasoning = f"Could not extract answer from response: {response[:100]}..."
        return result

    def evaluate_results(self, results_path: Path) -> EvaluationSummary:
        """Evaluate results from a results.json file.

        Args:
            results_path: Path to results.json file.

        Returns:
            EvaluationSummary with accuracy and per-task results.
        """
        with open(results_path) as f:
            data = json.load(f)

        tasks = data.get("tasks", data)  # Handle both old and new format
        summary = EvaluationSummary()

        for task_id, task_data in tasks.items():
            prediction = task_data.get("prediction")
            ground_truth = task_data.get("ground_truth", "")

            result = self.judge(prediction, ground_truth, task_id)
            summary.results.append(result)
            summary.total += 1

            if prediction is None:
                summary.no_answer += 1
            elif result.correct:
                summary.correct += 1
            else:
                summary.incorrect += 1

        return summary

    def evaluate_trajectories(self, output_dir: Path) -> EvaluationSummary:
        """Evaluate all trajectories in an output directory.

        Args:
            output_dir: Directory containing task subdirectories with .traj.json files.

        Returns:
            EvaluationSummary with accuracy and per-task results.
        """
        summary = EvaluationSummary()

        for traj_file in output_dir.rglob("*.traj.json"):
            with open(traj_file) as f:
                traj_data = json.load(f)

            info = traj_data.get("info", {})
            benchmark_data = info.get("benchmark_data", info.get("extra_info", {}).get("benchmark_data", {}))

            task_id = info.get("task_id", traj_file.stem)
            ground_truth = benchmark_data.get("ground_truth", "")
            prediction = benchmark_data.get("prediction") or info.get("submission")

            result = self.judge(prediction, ground_truth, task_id)
            summary.results.append(result)
            summary.total += 1

            if prediction is None:
                summary.no_answer += 1
            elif result.correct:
                summary.correct += 1
            else:
                summary.incorrect += 1

        return summary


def evaluate_hle_results(
    results_path: Path | str,
    output_path: Path | str | None = None,
) -> EvaluationSummary:
    """Convenience function to evaluate HLE results.

    Args:
        results_path: Path to results.json file.
        output_path: Optional path to save evaluation results.

    Returns:
        EvaluationSummary with accuracy statistics.
    """
    judge = HLEJudge()
    summary = judge.evaluate_results(Path(results_path))

    if output_path:
        output_data = {
            "summary": summary.to_dict(),
            "results": [r.to_dict() for r in summary.results],
        }
        Path(output_path).write_text(json.dumps(output_data, indent=2))

    return summary


# =============================================================================
# LLM-as-a-Judge (Official HLE Implementation)
# =============================================================================

# Official HLE Judge Prompt from centerforaisafety/hle
LLM_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.

Respond with a JSON object containing these four fields: extracted_final_answer, reasoning, correct, confidence."""


@dataclass
class LLMJudgeResult:
    """Structured output from LLM Judge (matches official HLE format)."""

    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int

    @classmethod
    def from_dict(cls, data: dict) -> "LLMJudgeResult":
        return cls(
            extracted_final_answer=data.get("extracted_final_answer", "None"),
            reasoning=data.get("reasoning", ""),
            correct=data.get("correct", "no"),
            confidence=data.get("confidence", 100),
        )


class LLMJudge:
    """LLM-as-a-Judge evaluator aligned with official HLE implementation.

    Uses the official HLE judge prompt to evaluate answers using an LLM.
    The judge extracts the final answer from the response and compares it
    with the ground truth, accounting for equivalent formats and small
    numerical differences.

    Reference: https://github.com/centerforaisafety/hle

    Example:
        >>> from miniagenticrouter.eval.hle_judge import LLMJudge
        >>> judge = LLMJudge(model_name="openai/gpt-4o")
        >>> result = judge.judge(
        ...     question="What is the capital of France?",
        ...     response="The capital of France is Paris.",
        ...     correct_answer="Paris",
        ...     task_id="task_001",
        ... )
        >>> result.correct
        True
    """

    def __init__(
        self,
        model_name: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the LLM Judge.

        Args:
            model_name: Model to use for judging (e.g., "openai/gpt-4o").
            model_kwargs: Additional kwargs for the model (e.g., temperature).
        """
        from miniagenticrouter.models import get_model

        # Wrap model_kwargs properly for get_model config format
        config = {"model_kwargs": model_kwargs} if model_kwargs else {}
        self.model = get_model(model_name, config)
        self.model_name = model_name

    def _parse_response(self, content: str) -> LLMJudgeResult:
        """Parse the LLM response into structured format.

        Tries to extract JSON from the response, handling various formats.

        Args:
            content: Raw response content from the LLM.

        Returns:
            Parsed LLMJudgeResult.
        """
        # Try to find JSON in the response
        json_patterns = [
            r"```json\s*(.*?)\s*```",  # Markdown code block
            r"```\s*(.*?)\s*```",  # Generic code block
            r"\{[^{}]*\}",  # Raw JSON object
        ]

        for pattern in json_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    data = json.loads(json_str)
                    return LLMJudgeResult.from_dict(data)
                except json.JSONDecodeError:
                    continue

        # Fallback: try to parse field by field from text
        result_data = {
            "extracted_final_answer": "None",
            "reasoning": "",
            "correct": "no",
            "confidence": 100,
        }

        # Extract each field using regex
        patterns = {
            "extracted_final_answer": r"extracted_final_answer[:\s]+(.+?)(?:\n|reasoning:|correct:|confidence:|$)",
            "reasoning": r"reasoning[:\s]+(.+?)(?:\n\n|correct:|confidence:|$)",
            "correct": r"correct[:\s]+(yes|no)",
            "confidence": r"confidence[:\s]+(\d+)",
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                if field == "confidence":
                    try:
                        result_data[field] = int(value)
                    except ValueError:
                        pass
                elif field == "correct":
                    result_data[field] = value.lower()
                else:
                    result_data[field] = value

        return LLMJudgeResult.from_dict(result_data)

    def judge(
        self,
        question: str,
        response: str,
        correct_answer: str,
        task_id: str = "",
    ) -> JudgeResult:
        """Judge whether the response correctly answers the question.

        Args:
            question: The original question.
            response: The model's full response.
            correct_answer: The ground truth answer.
            task_id: Task identifier for tracking.

        Returns:
            JudgeResult with correctness and detailed reasoning.
        """
        if not response:
            return JudgeResult(
                task_id=task_id,
                prediction=None,
                ground_truth=correct_answer,
                correct=False,
                confidence=1.0,
                reasoning="Empty response",
                method="llm_judge",
            )

        # Format the judge prompt
        prompt = LLM_JUDGE_PROMPT.format(
            question=question,
            response=response,
            correct_answer=correct_answer,
        )

        try:
            # Query the judge model
            result = self.model.query([{"role": "user", "content": prompt}])
            content = result.get("content", "")

            # Parse the structured response
            parsed = self._parse_response(content)

            return JudgeResult(
                task_id=task_id,
                prediction=parsed.extracted_final_answer,
                ground_truth=correct_answer,
                correct=(parsed.correct == "yes"),
                confidence=parsed.confidence / 100.0,
                reasoning=parsed.reasoning,
                method="llm_judge",
            )

        except Exception as e:
            logger.error(f"LLM Judge error for task {task_id}: {e}")
            return JudgeResult(
                task_id=task_id,
                prediction=None,
                ground_truth=correct_answer,
                correct=False,
                confidence=0.0,
                reasoning=f"LLM Judge error: {e}",
                method="llm_judge",
            )

    def judge_batch(
        self,
        items: list[dict[str, str]],
    ) -> list[JudgeResult]:
        """Judge multiple items.

        Args:
            items: List of dicts with keys: question, response, correct_answer, task_id.

        Returns:
            List of JudgeResult objects.
        """
        results = []
        for item in items:
            result = self.judge(
                question=item.get("question", ""),
                response=item.get("response", ""),
                correct_answer=item.get("correct_answer", ""),
                task_id=item.get("task_id", ""),
            )
            results.append(result)
        return results
