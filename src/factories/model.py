"""
Model factory module for creating various spatial-temporal graph neural network models.
This module provides a flexible factory pattern implementation for model instantiation
with support for automatic graph structure processing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import numpy as np
import torch
from hydra.utils import instantiate

from src.utils.graph_algo import calculate_cheb_poly, normalize_adj_mx


class GraphProcessor(ABC):
    """Abstract base class for graph structure processors.

    This class defines the interface for processing adjacency matrices
    into formats suitable for different types of graph neural networks.
    """

    @abstractmethod
    def process(self, adj_mx: np.ndarray, device: torch.device) -> Any:
        """Process adjacency matrix into required format.

        Args:
            adj_mx: Input adjacency matrix of shape (num_nodes, num_nodes)
            device: Target device for torch tensors

        Returns:
            Processed graph structure in appropriate format
        """
        pass


class SimpleGraphProcessor(GraphProcessor):

    def process(self, adj_mx: np.ndarray, device: torch.device) -> torch.Tensor:
        np.fill_diagonal(adj_mx, 1)
        return torch.tensor(adj_mx != 0).float().to(device)


class ScalapGraphProcessor(GraphProcessor):
    """Processor for SCALAP normalized adjacency matrices."""

    def process(self, adj_mx: np.ndarray, device: torch.device) -> torch.Tensor:
        """Process adjacency matrix to SCALAP normalized GSO tensor.

        Args:
            adj_mx: Input adjacency matrix
            device: Target device

        Returns:
            torch.Tensor: Normalized graph shift operator
        """
        gso = normalize_adj_mx(adj_mx, "scalap")[0]
        return torch.tensor(gso).to(device)


class ChebyshevGraphProcessor(GraphProcessor):
    """Processor for Chebyshev polynomial graph filters."""

    def __init__(self, order: int):
        """Initialize processor with Chebyshev polynomial order.

        Args:
            order: Order of Chebyshev polynomials
        """
        self.order = order

    def process(self, adj_mx: np.ndarray, device: torch.device) -> list:
        """Process adjacency matrix to Chebyshev polynomial list.

        Args:
            adj_mx: Input adjacency matrix
            device: Target device

        Returns:
            list: List of Chebyshev polynomial tensors
        """
        node_num = adj_mx.shape[0]
        adj = np.zeros((node_num, node_num), dtype=np.float32)

        # Create binary adjacency matrix
        np.fill_diagonal(adj, 0)  # Ensure no self-loops
        for n in range(node_num):
            idx = np.nonzero(adj_mx[n])[0]
            adj[n, idx] = 1

        L_tilde = normalize_adj_mx(adj, "scalap")[0]
        cheb_poly = calculate_cheb_poly(L_tilde, self.order)

        return [
            torch.from_numpy(poly).type(torch.FloatTensor).to(device)
            for poly in cheb_poly
        ]


class DoubleTransitionGraphProcessor(GraphProcessor):
    """Processor for double transition normalized adjacency matrices."""

    def process(self, adj_mx: np.ndarray, device: torch.device) -> list:
        """Process adjacency matrix to double transition normalized tensors.

        Args:
            adj_mx: Input adjacency matrix
            device: Target device

        Returns:
            list: List of normalized adjacency matrices as torch tensors
        """
        adj_mx_list = normalize_adj_mx(adj_mx, "doubletransition")
        return [torch.tensor(adj).to(device) for adj in adj_mx_list]


class DSTAGNNGraphProcessor(GraphProcessor):
    """Processor for DSTAGNN"""

    def __init__(self, order: int):
        self.cheb_poly = ChebyshevGraphProcessor(order)
        self.simple = SimpleGraphProcessor()

    def process(self, adj_mx: np.ndarray, device: torch.device) -> list:
        """Process graph data for DSTAGNN.

        Args:
            adj_mx: Input adjacency matrix
            device: Target device

        Returns:
            ...
        """
        cheb_poly = self.cheb_poly.process(adj_mx, device)
        adj_pa = self.simple.process(adj_mx, device)
        return {"cheb_poly": cheb_poly, "adj_pa": adj_pa}


class NullGraphProcessor(GraphProcessor):
    """Processor for models that don't require graph structure."""

    def process(self, adj_mx: np.ndarray, device: torch.device) -> None:
        """Return None as graph structure is not required.

        Args:
            adj_mx: Input adjacency matrix (unused)
            device: Target device (unused)

        Returns:
            None
        """
        return None


class ModelBuilder(ABC):
    """Abstract base class for model builders.

    Each concrete builder implements the specific construction logic
    for a particular model architecture.
    """

    def __init__(self, graph_processor: Optional[GraphProcessor] = None):
        """Initialize builder with optional graph processor.

        Args:
            graph_processor: Processor for graph structure, None for models
                           that don't require graph processing
        """
        self.graph_processor = graph_processor

    def build(
        self,
        ext_kwargs: Dict[str, Any],
        adj_mx: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        """Build and return the model instance.

        Args:
            ext_kwargs: Extended keyword arguments for model configuration
            adj_mx: Adjacency matrix for graph-based models
            device: Target device for the model

        Returns:
            Instantiated model
        """
        # Process graph structure if needed
        graph_data = self._process_graph(adj_mx, device) if adj_mx is not None else None

        # Prepare model arguments
        model_args = self._prepare_model_args(ext_kwargs, graph_data, device)

        # Instantiate and return model
        return instantiate(model_args)

    def _process_graph(self, adj_mx: np.ndarray, device: torch.device) -> Any:
        """Process adjacency matrix using the configured graph processor.

        Args:
            adj_mx: Input adjacency matrix
            device: Target device

        Returns:
            Processed graph data

        Raises:
            ValueError: If graph processor is None but graph data is provided
        """
        if self.graph_processor is None:
            raise ValueError("Graph processor is not configured for this model")
        return self.graph_processor.process(adj_mx, device)

    @abstractmethod
    def _prepare_model_args(
        self,
        ext_kwargs: Dict[str, Any],
        graph_data: Any,
        device: Optional[torch.device],
    ) -> Dict[str, Any]:
        """Prepare arguments for model instantiation.

        Args:
            ext_kwargs: Original configuration arguments
            graph_data: Processed graph data
            device: Target device

        Returns:
            Dictionary of arguments for model instantiation
        """
        pass


class STGCNBuilder(ModelBuilder):
    """Builder for STGCN model."""

    def __init__(self):
        super().__init__(ScalapGraphProcessor())

    def _prepare_model_args(
        self, ext_kwargs: Dict[str, Any], graph_data: torch.Tensor, device: torch.device
    ) -> Dict[str, Any]:
        """Prepare arguments for STGCN instantiation.

        The STGCN architecture uses temporal convolutional blocks followed by
        graph convolutional layers. The block structure is dynamically calculated
        based on input sequence length and kernel parameters.

        Args:
            ext_kwargs: Original configuration
            graph_data: Processed GSO tensor
            device: Target device

        Returns:
            Dictionary of instantiation arguments
        """
        # Calculate output sequence length after temporal convolutions
        Kt = ext_kwargs["Kt"]
        block_num = ext_kwargs["block_num"]
        his_len = ext_kwargs["his_len"]

        # Ko represents the length after temporal convolutions
        Ko = his_len - (Kt - 1) * 2 * block_num

        # Define block architecture
        # Each block: [temporal_conv_in, graph_conv, temporal_conv_out]
        blocks = [[ext_kwargs["input_dim"]]]
        for _ in range(block_num):
            blocks.append([64, 16, 64])

        # Add output layers based on sequence length
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([ext_kwargs["pred_len"]])

        return {**ext_kwargs, "gso": graph_data, "blocks": blocks}


class ASTGCNBuilder(ModelBuilder):
    """Builder for ASTGCN model."""

    def __init__(self):
        super().__init__(ChebyshevGraphProcessor(order=3))  # Default order=3

    def _prepare_model_args(
        self, ext_kwargs: Dict[str, Any], graph_data: list, device: torch.device
    ) -> Dict[str, Any]:
        """Prepare arguments for ASTGCN instantiation.

        ASTGCN uses Chebyshev polynomials for graph convolution with
        attention mechanisms for spatial and temporal dependencies.

        Args:
            ext_kwargs: Original configuration
            graph_data: List of Chebyshev polynomial tensors
            device: Target device

        Returns:
            Dictionary of instantiation arguments
        """
        return {**ext_kwargs, "cheb_poly": graph_data}


class GWNetBuilder(ModelBuilder):
    """Builder for GWNet model."""

    def __init__(self):
        super().__init__(DoubleTransitionGraphProcessor())

    def _prepare_model_args(
        self, ext_kwargs: Dict[str, Any], graph_data: list, device: torch.device
    ) -> Dict[str, Any]:
        """Prepare arguments for GWNet instantiation.

        GWNet uses diffusion convolutional layers with multiple adjacency
        matrices for capturing complex spatial dependencies.

        Args:
            ext_kwargs: Original configuration
            graph_data: List of normalized adjacency matrices
            device: Target device

        Returns:
            Dictionary of instantiation arguments
        """
        return {
            **ext_kwargs,
            "supports": graph_data,
            "residual_channels": ext_kwargs["init_dim"],
            "dilation_channels": ext_kwargs["init_dim"],
        }


class STTNBuilder(ModelBuilder):
    """Builder for STTN model."""

    def __init__(self):
        super().__init__(DoubleTransitionGraphProcessor())

    def _prepare_model_args(
        self, ext_kwargs: Dict[str, Any], graph_data: list, device: torch.device
    ) -> Dict[str, Any]:
        """Prepare arguments for STTN instantiation.

        STTN combines transformer architecture with graph convolutions
        for spatial-temporal modeling.

        Args:
            ext_kwargs: Original configuration
            graph_data: List of normalized adjacency matrices
            device: Target device

        Returns:
            Dictionary of instantiation arguments
        """
        return {**ext_kwargs, "supports": graph_data}


class DSTAGNNBuilder(ModelBuilder):
    """Builder for DSTAGNN model."""

    def __init__(self):
        super().__init__(DSTAGNNGraphProcessor(order=3))

    def _prepare_model_args(
        self, ext_kwargs: Dict[str, Any], graph_data: list, device: torch.device
    ) -> Dict[str, Any]:
        """Prepare arguments for DSTAGNN instantiation.

        Args:
            ext_kwargs: Original configuration
            graph_data: List of normalized adjacency matrices
            device: Target device

        Returns:
            Dictionary of instantiation arguments
        """
        return {**ext_kwargs, **graph_data}


class NoGraphModelBuilder(ModelBuilder):
    """Builder for models that don't require graph structure."""

    def __init__(self):
        super().__init__(NullGraphProcessor())

    def _prepare_model_args(
        self, ext_kwargs: Dict[str, Any], graph_data: None, device: torch.device
    ) -> Dict[str, Any]:
        """Prepare arguments for models without graph dependency.

        Args:
            ext_kwargs: Original configuration
            graph_data: Always None
            device: Target device

        Returns:
            Dictionary of instantiation arguments
        """
        return ext_kwargs


class ModelFactory:
    """Factory class for creating model instances.

    This factory uses the builder pattern to create different types of
    spatial-temporal models. It maintains a registry of available models
    and their corresponding builders.
    """

    # Registry mapping model names to builder classes
    _BUILDER_REGISTRY: Dict[str, Callable[[], ModelBuilder]] = {
        "STGCN": STGCNBuilder,
        "STGCN_JEPA": STGCNBuilder, # JEPA形式的stgcn
        "ASTGCN": ASTGCNBuilder,
        "GWNET": GWNetBuilder,
        "DSTAGNN": DSTAGNNBuilder,
        "STTN": STTNBuilder,
        "STAEformer": lambda: NoGraphModelBuilder(),
        "AGCRN": lambda: NoGraphModelBuilder(),
        "STED": lambda: NoGraphModelBuilder(),
        "Autoformer": lambda: NoGraphModelBuilder(),
        "PatchTST": lambda: NoGraphModelBuilder(),
        "LSTM": lambda: NoGraphModelBuilder(),
        "DLinear": lambda: NoGraphModelBuilder(),
        "iTransformer": lambda: NoGraphModelBuilder(),
        "FEDformer": lambda: NoGraphModelBuilder(),
    }

    @classmethod
    def register_builder(
        cls, model_name: str, builder_class: Callable[[], ModelBuilder]
    ):
        """Register a new model builder.

        Args:
            model_name: Name of the model
            builder_class: Builder class or factory function

        Example:
            >>> ModelFactory.register_builder("MyModel", MyModelBuilder)
        """
        cls._BUILDER_REGISTRY[model_name] = builder_class

    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names.

        Returns:
            List of registered model names
        """
        return list(cls._BUILDER_REGISTRY.keys())

    @classmethod
    def create_model(
        cls,
        model_name: str,
        ext_kwargs: Dict[str, Any],
        adj_mx: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        """Create a model instance by name.

        Args:
            model_name: Name of the model to create
            ext_kwargs: Extended keyword arguments for model configuration
            adj_mx: Adjacency matrix for graph-based models
            device: Target device for the model

        Returns:
            Instantiated model

        Raises:
            ValueError: If model_name is not registered
        """
        if model_name not in cls._BUILDER_REGISTRY:
            available_models = ", ".join(cls.get_available_models())
            raise ValueError(
                f"Model '{model_name}' is not implemented. "
                f"Available models: {available_models}"
            )

        # Get builder instance
        builder_factory = cls._BUILDER_REGISTRY[model_name]
        builder = builder_factory()

        # Build and return model
        return builder.build(ext_kwargs, adj_mx, device)
