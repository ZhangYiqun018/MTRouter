"""Tests for router module initialization and factory functions."""

import pytest

from miniagenticrouter.routers import Router, get_router, get_router_class


class TestRouterProtocol:
    """Tests for Router Protocol."""

    def test_protocol_defined(self):
        """Test that Router Protocol is properly defined."""
        from typing import get_type_hints

        # Protocol should have the expected attributes/methods
        assert hasattr(Router, "__protocol_attrs__") or hasattr(Router, "_is_protocol")

    def test_protocol_attributes(self):
        """Test that Router Protocol specifies expected attributes."""
        # Protocol should specify these attributes
        annotations = getattr(Router, "__annotations__", {})
        assert "config" in annotations or hasattr(Router, "config")
        assert "models" in annotations or hasattr(Router, "models")


class TestGetRouterClass:
    """Tests for get_router_class factory function."""

    def test_get_roulette_router_class(self):
        """Test getting RouletteRouter class by name."""
        from miniagenticrouter.routers.roulette import RouletteRouter

        cls = get_router_class("roulette")
        assert cls == RouletteRouter

    def test_get_interleaving_router_class(self):
        """Test getting InterleavingRouter class by name."""
        from miniagenticrouter.routers.interleaving import InterleavingRouter

        cls = get_router_class("interleaving")
        assert cls == InterleavingRouter

    def test_get_router_class_by_full_path(self):
        """Test getting router class by full import path."""
        from miniagenticrouter.routers.roulette import RouletteRouter

        cls = get_router_class("miniagenticrouter.routers.roulette.RouletteRouter")
        assert cls == RouletteRouter

    def test_unknown_router_class_raises_error(self):
        """Test that unknown router class name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown router class"):
            get_router_class("nonexistent_router")

    def test_invalid_import_path_raises_error(self):
        """Test that invalid import path raises ValueError."""
        with pytest.raises(ValueError, match="Unknown router class"):
            get_router_class("invalid.path.to.NonExistentRouter")


class TestGetRouter:
    """Tests for get_router factory function."""

    def test_get_roulette_router(self):
        """Test creating RouletteRouter via factory."""
        from miniagenticrouter.routers.roulette import RouletteRouter

        config = {
            "router_class": "roulette",
            "model_kwargs": [
                {"model_name": "test-model-1", "model_class": "deterministic", "outputs": ["a"]},
                {"model_name": "test-model-2", "model_class": "deterministic", "outputs": ["b"]},
            ],
        }
        router = get_router(config)
        assert isinstance(router, RouletteRouter)
        assert len(router.models) == 2

    def test_get_interleaving_router(self):
        """Test creating InterleavingRouter via factory."""
        from miniagenticrouter.routers.interleaving import InterleavingRouter

        config = {
            "router_class": "interleaving",
            "model_kwargs": [
                {"model_name": "test-model-1", "model_class": "deterministic", "outputs": ["a"]},
                {"model_name": "test-model-2", "model_class": "deterministic", "outputs": ["b"]},
            ],
        }
        router = get_router(config)
        assert isinstance(router, InterleavingRouter)
        assert len(router.models) == 2

    def test_default_router_class_is_roulette(self):
        """Test that default router class is roulette when not specified."""
        from miniagenticrouter.routers.roulette import RouletteRouter

        config = {
            "model_kwargs": [
                {"model_name": "test-model", "model_class": "deterministic", "outputs": ["a"]},
            ],
        }
        router = get_router(config)
        assert isinstance(router, RouletteRouter)

    def test_config_not_mutated(self):
        """Test that original config is not mutated."""
        original_config = {
            "router_class": "roulette",
            "model_kwargs": [
                {"model_name": "test-model", "model_class": "deterministic", "outputs": ["a"]},
            ],
        }
        config_copy = original_config.copy()
        get_router(original_config)
        assert original_config == config_copy


class TestBackwardCompatibility:
    """Tests for backward compatibility with old import paths."""

    def test_import_from_old_location(self):
        """Test that old import paths still work."""
        from miniagenticrouter.models.extra.roulette import (
            InterleavingModel,
            InterleavingModelConfig,
            RouletteModel,
            RouletteModelConfig,
        )

        # Verify they are the same classes
        from miniagenticrouter.routers.interleaving import InterleavingRouter, InterleavingRouterConfig
        from miniagenticrouter.routers.roulette import RouletteRouter, RouletteRouterConfig

        assert RouletteModel is RouletteRouter
        assert RouletteModelConfig is RouletteRouterConfig
        assert InterleavingModel is InterleavingRouter
        assert InterleavingModelConfig is InterleavingRouterConfig

    def test_old_model_class_mapping_works(self):
        """Test that using old model_class names still works."""
        from miniagenticrouter.models import get_model_class
        from miniagenticrouter.routers.interleaving import InterleavingRouter
        from miniagenticrouter.routers.roulette import RouletteRouter

        # Using shortcut names
        assert get_model_class("any", "roulette") == RouletteRouter
        assert get_model_class("any", "interleaving") == InterleavingRouter
