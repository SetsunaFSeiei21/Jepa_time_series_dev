"""
Spatio-temporal adjacency matrix building system.
Using Factory pattern for creation and Strategy pattern for different representations.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, List, Tuple, Dict, Any
import numpy as np
import torch
from scipy.sparse import csr_matrix, coo_matrix
from scipy.spatial.distance import cdist
from einops import rearrange, repeat
import warnings


class AdjacencyBuilder(ABC):
    """Abstract base class for adjacency matrix builders (Strategy pattern)."""
    
    @abstractmethod
    def build(self) -> torch.Tensor:
        """Build adjacency matrix with shape (N, N)."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the graph structure."""
        pass


class CoordinateAdjacencyBuilder(AdjacencyBuilder):
    """
    Build adjacency matrix from node coordinates using distance metrics.
    Strategy: Distance-based adjacency with Gaussian kernel.
    """
    
    def __init__(
        self,
        coordinates: Union[np.ndarray, torch.Tensor],
        threshold: Optional[float] = None,
        sigma: float = 1.0,
        normalized: bool = True,
        distance_metric: str = 'euclidean',
        sparse_threshold: Optional[float] = 0.1
    ):
        """
        Args:
            coordinates: Node coordinates with shape (N, D) where D is dimension
            threshold: Distance threshold for connectivity (None for fully connected)
            sigma: Bandwidth parameter for Gaussian kernel
            normalized: Whether to normalize the adjacency matrix
            distance_metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            sparse_threshold: Threshold for sparsification (set to 0 for dense)
        """
        # Convert to numpy array for compatibility
        if isinstance(coordinates, torch.Tensor):
            self.coordinates = coordinates.cpu().numpy()
        else:
            self.coordinates = np.array(coordinates)
        
        self.threshold = threshold
        self.sigma = sigma
        self.normalized = normalized
        self.distance_metric = distance_metric
        self.sparse_threshold = sparse_threshold
        
        # Validate inputs
        if self.coordinates.ndim != 2:
            raise ValueError(f"Coordinates must be 2D array, got {self.coordinates.ndim}D")
        
        self.n_nodes = self.coordinates.shape[0]
        self._adjacency = None
        self._metadata = {
            'type': 'coordinate_based',
            'n_nodes': self.n_nodes,
            'threshold': threshold,
            'sigma': sigma,
            'normalized': normalized
        }
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute pairwise distance matrix using vectorized operations."""
        # Using scipy's cdist for efficient distance computation
        # Shape: (N, N) -> distances between all pairs
        return cdist(self.coordinates, self.coordinates, metric=self.distance_metric)
    
    def _apply_gaussian_kernel(self, distances: np.ndarray) -> np.ndarray:
        """Apply Gaussian kernel to distance matrix."""
        # Vectorized Gaussian kernel: exp(-dist^2 / (2 * sigma^2))
        gaussian_weights = np.exp(-np.square(distances) / (2 * self.sigma ** 2))
        
        # Apply threshold if specified
        if self.threshold is not None:
            mask = distances <= self.threshold
            gaussian_weights = gaussian_weights * mask
        
        # Set diagonal to 0 (no self-loops)
        np.fill_diagonal(gaussian_weights, 0)
        
        return gaussian_weights
    
    def _normalize_adjacency(self, adjacency: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix using symmetric normalization."""
        if not self.normalized:
            return adjacency
        
        # Add self-loops
        adj_with_self_loops = adjacency + np.eye(self.n_nodes)
        
        # Compute degree matrix
        degree = np.sum(adj_with_self_loops, axis=1)
        
        # Avoid division by zero
        degree_sqrt_inv = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        # Using broadcasting for efficient computation
        degree_matrix_sqrt_inv = np.diag(degree_sqrt_inv)
        normalized_adj = degree_matrix_sqrt_inv @ adj_with_self_loops @ degree_matrix_sqrt_inv
        
        return normalized_adj
    
    def _sparsify(self, adjacency: np.ndarray) -> np.ndarray:
        """Sparsify adjacency matrix based on threshold."""
        if self.sparse_threshold is None or self.sparse_threshold <= 0:
            return adjacency
        
        # Create sparse matrix by thresholding
        sparse_adj = adjacency.copy()
        sparse_adj[sparse_adj < self.sparse_threshold] = 0
        
        return sparse_adj
    
    def build(self) -> torch.Tensor:
        """Build adjacency matrix from coordinates."""
        if self._adjacency is not None:
            return self._adjacency
        
        # Compute distance matrix (vectorized)
        distances = self._compute_distance_matrix()  # (N, N)
        
        # Apply Gaussian kernel
        adjacency = self._apply_gaussian_kernel(distances)  # (N, N)
        
        # Sparsify if needed
        adjacency = self._sparsify(adjacency)  # (N, N)
        
        # Normalize
        adjacency = self._normalize_adjacency(adjacency)  # (N, N)
        
        # Convert to torch tensor
        self._adjacency = torch.FloatTensor(adjacency)
        
        return self._adjacency
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the graph."""
        if self._adjacency is None:
            self.build()
        
        density = np.count_nonzero(self._adjacency.numpy()) / (self.n_nodes ** 2)
        self._metadata.update({
            'density': density,
            'shape': (self.n_nodes, self.n_nodes)
        })
        
        return self._metadata


class MatrixAdjacencyBuilder(AdjacencyBuilder):
    """
    Directly use provided adjacency matrix.
    Strategy: Direct matrix usage with validation.
    """
    
    def __init__(
        self,
        adjacency_matrix: Union[np.ndarray, torch.Tensor],
        normalized: bool = False
    ):
        """
        Args:
            adjacency_matrix: Precomputed adjacency matrix with shape (N, N)
            normalized: Whether to normalize the matrix
        """
        if isinstance(adjacency_matrix, torch.Tensor):
            self.adjacency = adjacency_matrix.cpu().numpy()
        else:
            self.adjacency = np.array(adjacency_matrix)
        
        self.normalized = normalized
        
        # Validate matrix
        if self.adjacency.ndim != 2:
            raise ValueError(f"Adjacency matrix must be 2D, got {self.adjacency.ndim}D")
        
        if self.adjacency.shape[0] != self.adjacency.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got {self.adjacency.shape}")
        
        self.n_nodes = self.adjacency.shape[0]
        self._metadata = {
            'type': 'matrix_based',
            'n_nodes': self.n_nodes,
            'normalized': normalized
        }
    
    def build(self) -> torch.Tensor:
        """Return the adjacency matrix, optionally normalized."""
        adjacency = self.adjacency.copy()
        
        if self.normalized:
            # Normalize the adjacency matrix
            degree = np.sum(adjacency, axis=1)
            degree_sqrt_inv = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
            degree_matrix_sqrt_inv = np.diag(degree_sqrt_inv)
            adjacency = degree_matrix_sqrt_inv @ adjacency @ degree_matrix_sqrt_inv
        
        return torch.FloatTensor(adjacency)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the graph."""
        density = np.count_nonzero(self.adjacency) / (self.n_nodes ** 2)
        self._metadata.update({
            'density': density,
            'shape': (self.n_nodes, self.n_nodes)
        })
        
        return self._metadata


class AdjacencyListBuilder(AdjacencyBuilder):
    """
    Build adjacency matrix from adjacency list.
    Strategy: Edge list to adjacency matrix conversion.
    """
    
    def __init__(
        self,
        adjacency_list: Union[List[Tuple[int, int]], np.ndarray],
        n_nodes: int,
        edge_weights: Optional[Union[List[float], np.ndarray]] = None,
        directed: bool = False
    ):
        """
        Args:
            adjacency_list: List of edges as (source, target) pairs
            n_nodes: Number of nodes in the graph
            edge_weights: Optional weights for edges
            directed: Whether the graph is directed
        """
        self.adjacency_list = np.array(adjacency_list)
        self.n_nodes = n_nodes
        self.edge_weights = np.ones(len(adjacency_list)) if edge_weights is None else np.array(edge_weights)
        self.directed = directed
        
        if self.adjacency_list.shape[1] != 2:
            raise ValueError(f"Adjacency list must have shape (E, 2), got {self.adjacency_list.shape}")
        
        if len(self.edge_weights) != len(self.adjacency_list):
            raise ValueError(f"Edge weights length {len(self.edge_weights)} must match adjacency list length {len(self.adjacency_list)}")
        
        self._metadata = {
            'type': 'adjacency_list_based',
            'n_nodes': n_nodes,
            'n_edges': len(adjacency_list),
            'directed': directed
        }
    
    def build(self) -> torch.Tensor:
        """Convert adjacency list to adjacency matrix."""
        # Create empty adjacency matrix
        adjacency = np.zeros((self.n_nodes, self.n_nodes))
        
        # Populate adjacency matrix using vectorized operations
        sources = self.adjacency_list[:, 0].astype(int)
        targets = self.adjacency_list[:, 1].astype(int)
        
        # Use numpy indexing for efficient assignment
        adjacency[sources, targets] = self.edge_weights
        
        # If undirected, make symmetric
        if not self.directed:
            adjacency[targets, sources] = self.edge_weights
        
        return torch.FloatTensor(adjacency)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the graph."""
        density = len(self.adjacency_list) / (self.n_nodes ** 2)
        self._metadata.update({
            'density': density,
            'shape': (self.n_nodes, self.n_nodes)
        })
        
        return self._metadata


class AdjacencyBuilderFactory:
    """
    Factory for creating adjacency matrix builders (Factory pattern).
    Centralizes creation logic for different adjacency representations.
    """
    
    @staticmethod
    def create(
        graph_data: Union[np.ndarray, torch.Tensor, List[Tuple[int, int]]],
        graph_type: str = 'coordinates',
        **kwargs
    ) -> AdjacencyBuilder:
        """
        Factory method to create appropriate adjacency builder.
        
        Args:
            graph_data: Input graph data (coordinates, matrix, or adjacency list)
            graph_type: Type of graph data ('coordinates', 'matrix', 'adjacency_list')
            **kwargs: Additional arguments for the specific builder
        
        Returns:
            Appropriate AdjacencyBuilder instance
        """
        if graph_type == 'coordinates':
            return CoordinateAdjacencyBuilder(coordinates=graph_data, **kwargs)
        
        elif graph_type == 'matrix':
            return MatrixAdjacencyBuilder(adjacency_matrix=graph_data, **kwargs)
        
        elif graph_type == 'adjacency_list':
            if 'n_nodes' not in kwargs:
                raise ValueError("n_nodes must be provided for adjacency_list type")
            return AdjacencyListBuilder(adjacency_list=graph_data, **kwargs)
        
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}. "
                           f"Must be one of: 'coordinates', 'matrix', 'adjacency_list'")