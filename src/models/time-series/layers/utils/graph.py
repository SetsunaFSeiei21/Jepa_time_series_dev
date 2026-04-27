import numpy as np
import torch
import torch_geometric as pyg

# def gaussian_kernel(x, K):
#     return torch.exp(-1 * (x ** 2) / (2 * K ** 2))


def exponential_kernel(x, K=1):
    return torch.exp(-1 * x / K)


def dense2sparse(adj_mtx):
    """
    Convert dense adjacency matrix to sparse adjacency matrix.
    """
    return pyg.utils.dense_to_sparse(adj_mtx)
