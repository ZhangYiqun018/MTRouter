"""Tests for RouletteRouter."""

import random
from unittest.mock import patch

import pytest

from miniagenticrouter.routers.roulette import RouletteRouter, RouletteRouterConfig


class TestRouletteRouterConfig:
    """Tests for RouletteRouterConfig dataclass."""

    def test_default_model_name(self):
        """Test default model_name is 'roulette'."""
        config = RouletteRouterConfig(model_kwargs=[])
        assert config.model_name == "roulette"

    def test_custom_model_name(self):
        """Test custom model_name can be set."""
        config = RouletteRouterConfig(model_kwargs=[], model_name="custom-roulette")
        assert config.model_name == "custom-roulette"

    def test_model_kwargs_stored(self):
        """Test model_kwargs are stored correctly."""
        model_kwargs = [{"model_name": "test"}]
        config = RouletteRouterConfig(model_kwargs=model_kwargs)
        assert config.model_kwargs == model_kwargs


class TestRouletteRouter:
    """Tests for RouletteRouter."""

    @pytest.fixture
    def model_kwargs(self):
        """Model configurations for testing."""
        return [
            {"model_name": "model-a", "model_class": "deterministic", "outputs": ["a"]},
            {"model_name": "model-b", "model_class": "deterministic", "outputs": ["b"]},
            {"model_name": "model-c", "model_class": "deterministic", "outputs": ["c"]},
        ]

    def test_instantiation(self, model_kwargs):
        """Test basic router instantiation."""
        router = RouletteRouter(model_kwargs=model_kwargs)
        assert len(router.models) == 3
        assert router.config.model_name == "roulette"

    def test_select_model_random_choice(self, model_kwargs):
        """Test that select_model uses random.choice."""
        router = RouletteRouter(model_kwargs=model_kwargs)

        with patch.object(random, "choice") as mock_choice:
            mock_choice.return_value = router.models[1]
            selected = router.select_model()
            mock_choice.assert_called_once_with(router.models)
            assert selected == router.models[1]

    def test_random_distribution(self, model_kwargs):
        """Test that models are selected with roughly equal probability."""
        router = RouletteRouter(model_kwargs=model_kwargs)

        # Set seed for reproducibility
        random.seed(42)

        selections = {"model-a": 0, "model-b": 0, "model-c": 0}
        n_samples = 300

        for _ in range(n_samples):
            model = router.select_model()
            selections[model.config.model_name] += 1

        # Each model should be selected roughly 1/3 of the time
        # Allow for some variance (20% tolerance)
        expected = n_samples / 3
        tolerance = expected * 0.4

        for count in selections.values():
            assert abs(count - expected) < tolerance, f"Selection counts: {selections}"

    def test_query_returns_selected_model_name(self, model_kwargs):
        """Test that query response includes selected model name."""
        router = RouletteRouter(model_kwargs=model_kwargs)

        # Fix selection to first model
        with patch.object(random, "choice", return_value=router.models[0]):
            response = router.query([{"role": "user", "content": "test"}])
            assert response["model_name"] == "model-a"

    def test_query_returns_router_type(self, model_kwargs):
        """Test that query response includes router type."""
        router = RouletteRouter(model_kwargs=model_kwargs)
        response = router.query([{"role": "user", "content": "test"}])
        assert response["router_type"] == "RouletteRouter"

    def test_get_template_vars_includes_config(self, model_kwargs):
        """Test that get_template_vars includes config fields."""
        router = RouletteRouter(model_kwargs=model_kwargs)
        vars = router.get_template_vars()

        assert "model_kwargs" in vars
        assert "model_name" in vars
        assert vars["model_name"] == "roulette"

    def test_cost_tracking(self, model_kwargs):
        """Test cost is tracked across queries."""
        router = RouletteRouter(model_kwargs=model_kwargs)
        assert router.cost == 0.0

        router.query([{"role": "user", "content": "test"}])
        # DeterministicModel has default cost_per_call
        assert router.cost > 0

    def test_n_calls_tracking(self, model_kwargs):
        """Test n_calls is tracked across queries."""
        router = RouletteRouter(model_kwargs=model_kwargs)
        assert router.n_calls == 0

        router.query([{"role": "user", "content": "test1"}])
        assert router.n_calls == 1

        router.query([{"role": "user", "content": "test2"}])
        assert router.n_calls == 2

    def test_backward_compatibility_alias(self):
        """Test RouletteModel alias is available."""
        from miniagenticrouter.routers.roulette import RouletteModel, RouletteModelConfig

        assert RouletteModel is RouletteRouter
        assert RouletteModelConfig is RouletteRouterConfig

    def test_single_model(self):
        """Test router with single model always returns that model."""
        model_kwargs = [{"model_name": "only-model", "model_class": "deterministic", "outputs": ["x"]}]
        router = RouletteRouter(model_kwargs=model_kwargs)

        for _ in range(10):
            model = router.select_model()
            assert model.config.model_name == "only-model"

    def test_custom_config_class(self, model_kwargs):
        """Test using a custom config class."""
        from dataclasses import dataclass

        @dataclass
        class CustomConfig(RouletteRouterConfig):
            custom_field: str = "default"

        router = RouletteRouter(
            config_class=CustomConfig, model_kwargs=model_kwargs, custom_field="custom_value"
        )
        assert router.config.custom_field == "custom_value"
