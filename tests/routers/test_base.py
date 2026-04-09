"""Tests for BaseRouter abstract base class."""

import pytest

from miniagenticrouter.routers.base import BaseRouter


class ConcreteRouter(BaseRouter):
    """Concrete implementation of BaseRouter for testing."""

    def __init__(self, *, model_kwargs: list[dict], selection_index: int = 0, **kwargs):
        super().__init__(model_kwargs=model_kwargs, **kwargs)
        self.selection_index = selection_index

    def select_model(self):
        return self.models[self.selection_index % len(self.models)]


class TestBaseRouter:
    """Tests for BaseRouter abstract base class."""

    @pytest.fixture
    def model_kwargs(self):
        """Model configurations for testing."""
        return [
            {"model_name": "test-model-1", "model_class": "deterministic", "outputs": ["response1"] * 10},
            {"model_name": "test-model-2", "model_class": "deterministic", "outputs": ["response2"] * 10},
        ]

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BaseRouter cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseRouter(model_kwargs=[])

    def test_concrete_router_instantiation(self, model_kwargs):
        """Test that concrete subclass can be instantiated."""
        router = ConcreteRouter(model_kwargs=model_kwargs)
        assert len(router.models) == 2

    def test_models_created_from_kwargs(self, model_kwargs):
        """Test that models are created from model_kwargs."""
        router = ConcreteRouter(model_kwargs=model_kwargs)

        assert router.models[0].config.model_name == "test-model-1"
        assert router.models[1].config.model_name == "test-model-2"

    def test_cost_aggregation(self, model_kwargs):
        """Test that cost is aggregated across all models."""
        router = ConcreteRouter(model_kwargs=model_kwargs)

        # Initially cost should be 0
        assert router.cost == 0.0

        # Query to increment costs
        router.query([{"role": "user", "content": "test"}])
        assert router.cost > 0

    def test_n_calls_aggregation(self, model_kwargs):
        """Test that n_calls is aggregated across all models."""
        router = ConcreteRouter(model_kwargs=model_kwargs)

        # Initially n_calls should be 0
        assert router.n_calls == 0

        # Query to increment call count
        router.query([{"role": "user", "content": "test"}])
        assert router.n_calls == 1

        # Query again
        router.query([{"role": "user", "content": "test2"}])
        assert router.n_calls == 2

    def test_query_returns_response_with_metadata(self, model_kwargs):
        """Test that query adds router metadata to response."""
        router = ConcreteRouter(model_kwargs=model_kwargs)
        response = router.query([{"role": "user", "content": "test"}])

        assert "model_name" in response
        assert "router_type" in response
        assert response["router_type"] == "ConcreteRouter"

    def test_query_uses_selected_model(self, model_kwargs):
        """Test that query uses the model returned by select_model."""
        # Select first model
        router = ConcreteRouter(model_kwargs=model_kwargs, selection_index=0)
        response = router.query([{"role": "user", "content": "test"}])
        assert response["model_name"] == "test-model-1"

        # Select second model
        router2 = ConcreteRouter(model_kwargs=model_kwargs, selection_index=1)
        response2 = router2.query([{"role": "user", "content": "test"}])
        assert response2["model_name"] == "test-model-2"

    def test_get_template_vars(self, model_kwargs):
        """Test get_template_vars returns expected fields."""
        router = ConcreteRouter(model_kwargs=model_kwargs)
        vars = router.get_template_vars()

        assert "n_model_calls" in vars
        assert "model_cost" in vars
        assert "router_type" in vars
        assert "num_models" in vars
        assert vars["router_type"] == "ConcreteRouter"
        assert vars["num_models"] == 2

    def test_extra_config_stored(self, model_kwargs):
        """Test that extra kwargs are stored in _extra_config."""
        router = ConcreteRouter(model_kwargs=model_kwargs, custom_param="value")
        assert router._extra_config["custom_param"] == "value"

    def test_empty_model_kwargs(self):
        """Test router with empty model_kwargs."""
        router = ConcreteRouter(model_kwargs=[])
        assert len(router.models) == 0
        assert router.cost == 0.0
        assert router.n_calls == 0
