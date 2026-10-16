"""Graph sources produce the supports (b, k, n, n) fed to a model for each batch.

A static graph is shared by every sample. A per-sample graph (a different (n, n) adjacency
for every sample, computed on the fly from the batch) is a source whose `adjacency` reads
the batch; the transform then runs on (b, n, n) exactly as for a static graph.
"""

from torch import Tensor

from src.data import Batch

from .transforms import GraphTransform


class GraphSource:
    def __init__(self, transform: GraphTransform):
        self.transform = transform

    @property
    def num_supports(self) -> int:
        return self.transform.num_supports

    def adjacency(self, batch: Batch) -> Tensor:
        """(b, n, n) adjacency for the batch."""
        raise NotImplementedError

    def __call__(self, batch: Batch) -> Tensor:
        return self.transform(self.adjacency(batch))


class StaticGraph(GraphSource):
    """The dataset's fixed adjacency, transformed once and shared by all samples."""

    def __init__(self, adj: Tensor, transform: GraphTransform):
        super().__init__(transform)
        self.supports = transform(adj.unsqueeze(0))  # (1, k, n, n)

    def __call__(self, batch: Batch) -> Tensor:
        return self.supports.expand(batch.x.shape[0], -1, -1, -1)
