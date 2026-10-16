"""
Engine factory module for creating training engines for different models.
This module distinguishes between models that require specialized training engines
and those that can use the standard BaseEngine.
"""

import importlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Type, Optional, Set
import warnings
from enum import Enum


class TrainingEngineFactory:
    """Factory for creating training engine instances.

    This factory distinguishes between two types of models:
    1. Specialized models: Have custom training logic, loss functions, or regularization
       that require specialized engine implementations.
    2. Standard models: Use the standard BaseEngine with common training pipeline.
    """

    # Class-level registry for specialized models
    _SPECIALIZED_MODELS: Set[str] = {
        "d2stgnn",  # Dilated Dense Spatio-Temporal Graph Neural Network
        "dcrnn",  # Diffusion Convolutional Recurrent Neural Network
        "dgcrn",  # Dynamic Graph Convolutional Recurrent Network
        "hl",  # History Learner
    }

    # Module and class name patterns for specialized engines
    _SPECIALIZED_MODULE_PATTERN = "src.engines.{}_engine"
    _SPECIALIZED_CLASS_PATTERN = "{}_Engine"

    @classmethod
    def register_specialized_model(cls, model_name: str) -> None:
        """Register a model as requiring specialized training engine.

        Args:
            model_name: Name of the model (will be normalized to lowercase)

        Raises:
            ValueError: If model is already registered
        """
        normalized_name = model_name.lower()
        if normalized_name in cls._SPECIALIZED_MODELS:
            raise ValueError(
                f"Model '{model_name}' is already registered as specialized"
            )

        cls._SPECIALIZED_MODELS.add(normalized_name)

    @classmethod
    def unregister_specialized_model(cls, model_name: str) -> bool:
        """Unregister a model from specialized training engines.

        Args:
            model_name: Name of the model (case-insensitive)

        Returns:
            True if model was removed, False if it wasn't registered
        """
        normalized_name = model_name.lower()
        if normalized_name in cls._SPECIALIZED_MODELS:
            cls._SPECIALIZED_MODELS.remove(normalized_name)
            return True
        return False

    @classmethod
    def is_specialized_model(cls, model_name: str) -> bool:
        """Check if a model requires specialized training engine.

        Args:
            model_name: Name of the model (case-insensitive)

        Returns:
            True if model requires specialized engine, False otherwise
        """
        return model_name.lower() in cls._SPECIALIZED_MODELS

    @classmethod
    def get_specialized_models(cls) -> Set[str]:
        """Get all registered specialized models.

        Returns:
            Set of model names
        """
        return cls._SPECIALIZED_MODELS.copy()

    @classmethod
    def _load_specialized_engine_class(cls, model_name: str) -> Any:
        """Load specialized engine class by dynamically importing module.

        Args:
            model_name: Name of the model

        Returns:
            Engine class

        Raises:
            ImportError: If module cannot be imported
            AttributeError: If class cannot be found in module
        """
        # Convert to appropriate case for import
        model_lower = model_name.lower()
        class_name_upper = cls._SPECIALIZED_CLASS_PATTERN.format(model_name.upper())

        module_path = cls._SPECIALIZED_MODULE_PATTERN.format(model_lower)
        module = importlib.import_module(module_path)
        engine_class = getattr(module, class_name_upper)

        return engine_class

    @classmethod
    def _get_engine_class(cls, model_name: str) -> Any:
        """Get the engine class for a model.

        Args:
            model_name: Name of the model (case-insensitive)

        Returns:
            Engine class

        Raises:
            ImportError: If specialized engine module cannot be imported
            AttributeError: If specialized engine class cannot be found
        """
        if cls.is_specialized_model(model_name):
            return cls._load_specialized_engine_class(model_name)
        else:
            from src.engines.expert import ExpertEngine

            return ExpertEngine

    @classmethod
    def create_engine(cls, model_name: str, **kwargs) -> Any:
        """Create and return an instantiated engine for the specified model.

        This is the main factory method that should be used to create engine instances.

        Args:
            model_name: Name of the model
            **kwargs: Arguments to pass to engine constructor

        Returns:
            Instantiated engine

        Raises:
            ImportError: If specialized engine module cannot be imported
            AttributeError: If specialized engine class cannot be found
            ValueError: If model_name is empty or invalid

        Example:
            >>> engine = TrainingEngineFactory.create_engine(
            ...     model_name="STGCN",
            ...     device=device,
            ...     model=model,
            ...     dataloader=loaders,
            ...     scaler=scaler,
            ...     log_dir=log_folder,
            ...     logger=logger,
            ...     loss_func=loss_func,
            ...     exp_kwargs=exp_kwargs,
            ... )
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")

        # Get the appropriate engine class
        engine_class = cls._get_engine_class(model_name)

        # Instantiate and return the engine
        return engine_class(**kwargs)
