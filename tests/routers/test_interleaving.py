"""Tests for InterleavingRouter."""

import pytest

from miniagenticrouter.routers.interleaving import InterleavingRouter, InterleavingRouterConfig


class TestInterleavingRouterConfig:
    """Tests for InterleavingRouterConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = InterleavingRouterConfig(model_kwargs=[])
        assert config.model_name == "interleaving"
        assert config.sequence is None

    def test_custom_sequence(self):
        """Test custom sequence can be set."""
        config = InterleavingRouterConfig(model_kwargs=[], sequence=[0, 0, 1])
        assert config.sequence == [0, 0, 1]

    def test_model_kwargs_stored(self):
        """Test model_kwargs are stored correctly."""
        model_kwargs = [{"model_name": "test"}]
        config = InterleavingRouterConfig(model_kwargs=model_kwargs)
        assert config.model_kwargs == model_kwargs


class TestInterleavingRouter:
    """Tests for InterleavingRouter."""

    @pytest.fixture
    def model_kwargs(self):
        """Model configurations for testing."""
        return [
            {"model_name": "model-a", "model_class": "deterministic", "outputs": ["a"] * 10},
            {"model_name": "model-b", "model_class": "deterministic", "outputs": ["b"] * 10},
            {"model_name": "model-c", "model_class": "deterministic", "outputs": ["c"] * 10},
        ]

    def test_instantiation(self, model_kwargs):
        """Test basic router instantiation."""
        router = InterleavingRouter(model_kwargs=model_kwargs)
        assert len(router.models) == 3
        assert router.config.model_name == "interleaving"
        assert router.config.sequence is None

    def test_round_robin_selection(self, model_kwargs):
        """Test round-robin selection when sequence is None."""
        router = InterleavingRouter(model_kwargs=model_kwargs)

        # Should cycle through models in order
        expected_order = ["model-a", "model-b", "model-c", "model-a", "model-b", "model-c"]

        for expected_name in expected_order:
            # Query to increment n_calls
            response = router.query([{"role": "user", "content": "test"}])
            assert response["model_name"] == expected_name

    def test_custom_sequence_selection(self, model_kwargs):
        """Test selection with custom sequence."""
        # Sequence: model 0, model 0, model 2, repeat
        router = InterleavingRouter(model_kwargs=model_kwargs, sequence=[0, 0, 2])

        expected_order = ["model-a", "model-a", "model-c", "model-a", "model-a", "model-c"]

        for expected_name in expected_order:
            response = router.query([{"role": "user", "content": "test"}])
            assert response["model_name"] == expected_name

    def test_sequence_with_single_index(self, model_kwargs):
        """Test sequence with repeated single index."""
        router = InterleavingRouter(model_kwargs=model_kwargs, sequence=[1])

        # Should always return model-b
        for _ in range(5):
            response = router.query([{"role": "user", "content": "test"}])
            assert response["model_name"] == "model-b"

    def test_select_model_uses_n_calls(self, model_kwargs):
        """Test that select_model uses n_calls for index calculation."""
        router = InterleavingRouter(model_kwargs=model_kwargs)

        # First call should return model-a (index 0)
        model = router.select_model()
        assert model.config.model_name == "model-a"

        # n_calls is still 0 until query is made
        # After query, n_calls becomes 1
        router.query([{"role": "user", "content": "test"}])

        # Now select_model should return model-b (index 1)
        model = router.select_model()
        assert model.config.model_name == "model-b"

    def test_query_returns_router_type(self, model_kwargs):
        """Test that query response includes router type."""
        router = InterleavingRouter(model_kwargs=model_kwargs)
        response = router.query([{"role": "user", "content": "test"}])
        assert response["router_type"] == "InterleavingRouter"

    def test_get_template_vars_includes_config(self, model_kwargs):
        """Test that get_template_vars includes config fields."""
        router = InterleavingRouter(model_kwargs=model_kwargs, sequence=[0, 1])
        vars = router.get_template_vars()

        assert "model_kwargs" in vars
        assert "model_name" in vars
        assert "sequence" in vars
        assert vars["model_name"] == "interleaving"
        assert vars["sequence"] == [0, 1]

    def test_cost_tracking(self, model_kwargs):
        """Test cost is tracked across queries."""
        router = InterleavingRouter(model_kwargs=model_kwargs)
        assert router.cost == 0.0

        router.query([{"role": "user", "content": "test"}])
        assert router.cost > 0

    def test_n_calls_tracking(self, model_kwargs):
        """Test n_calls is tracked across queries."""
        router = InterleavingRouter(model_kwargs=model_kwargs)
        assert router.n_calls == 0

        router.query([{"role": "user", "content": "test1"}])
        assert router.n_calls == 1

        router.query([{"role": "user", "content": "test2"}])
        assert router.n_calls == 2

    def test_backward_compatibility_alias(self):
        """Test InterleavingModel alias is available."""
        from miniagenticrouter.routers.interleaving import InterleavingModel, InterleavingModelConfig

        assert InterleavingModel is InterleavingRouter
        assert InterleavingModelConfig is InterleavingRouterConfig

    def test_two_models_round_robin(self):
        """Test round-robin with two models."""
        model_kwargs = [
            {"model_name": "model-x", "model_class": "deterministic", "outputs": ["x"] * 5},
            {"model_name": "model-y", "model_class": "deterministic", "outputs": ["y"] * 5},
        ]
        router = InterleavingRouter(model_kwargs=model_kwargs)

        expected_order = ["model-x", "model-y", "model-x", "model-y"]

        for expected_name in expected_order:
            response = router.query([{"role": "user", "content": "test"}])
            assert response["model_name"] == expected_name

    def test_single_model(self):
        """Test router with single model."""
        model_kwargs = [{"model_name": "only-model", "model_class": "deterministic", "outputs": ["x"] * 10}]
        router = InterleavingRouter(model_kwargs=model_kwargs)

        for _ in range(5):
            response = router.query([{"role": "user", "content": "test"}])
            assert response["model_name"] == "only-model"

    def test_custom_config_class(self, model_kwargs):
        """Test using a custom config class."""
        from dataclasses import dataclass

        @dataclass
        class CustomConfig(InterleavingRouterConfig):
            custom_field: str = "default"

        router = InterleavingRouter(
            config_class=CustomConfig, model_kwargs=model_kwargs, custom_field="custom_value"
        )
        assert router.config.custom_field == "custom_value"

    def test_sequence_wraps_around(self, model_kwargs):
        """Test that sequence properly wraps around."""
        # Short sequence that will wrap multiple times
        router = InterleavingRouter(model_kwargs=model_kwargs, sequence=[2, 0])

        expected_order = ["model-c", "model-a", "model-c", "model-a", "model-c", "model-a"]

        for expected_name in expected_order:
            response = router.query([{"role": "user", "content": "test"}])
            assert response["model_name"] == expected_name
