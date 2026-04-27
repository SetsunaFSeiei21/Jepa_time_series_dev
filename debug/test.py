import torch

from src.Azoth.cheb_conv import (
    compute_chebyshev_polynomials,
    compute_scaled_laplacian,
)
from src.Azoth.reasoner import HierarchicalReasoner

# Example usage
if __name__ == "__main__":
    # Create synthetic data
    B = 8
    N = 100
    F = 16
    L = 12
    K = 3

    # Random adjacency matrix
    adjacency = torch.rand(N, N)
    adjacency = (adjacency + adjacency.t()) / 2
    adjacency = (adjacency > 0.5).float()
    torch.diagonal(adjacency).fill_(0)

    # Compute Laplacian and Chebyshev polynomials
    laplacian = compute_scaled_laplacian(adjacency)
    T_k = compute_chebyshev_polynomials(laplacian, K)

    # Initialize hierarchical model
    model = HierarchicalReasoner(hid_dim=F, K=K, num_layers=1, condition_dim=32)

    # Input: single starting time step
    x_start = torch.randn(B, 1, N, F)

    # Forward pass
    output = model(x_start, T_k, output_steps=L)

    print(f"Input shape: {x_start.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Prediction steps: {L}")
    print(f"Number of scale predictors: {len(model.predictors)}")

    # Check that output has correct shape
    assert output.shape == (
        B,
        L,
        N,
        F,
    ), f"Expected shape {(B, L, N, F)}, got {output.shape}"
    print("✓ Output shape correct!")
