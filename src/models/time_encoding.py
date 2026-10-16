"""Time encodings shared by all models.

An encoding maps the time features of the input steps, (b, t, f) with values in [0, 1), to
(b, t, num_channels). `BaseModel` appends the result to every node's input channels, so all
models (except those declaring `custom_time_features`) see time in exactly the same way. The
encoding is chosen once for all models with the `time_encoding` config group.

`feature_sizes` maps each feature name to its number of distinct values (from the dataset's
meta.json); `feature * size` is that value's integer index.
"""

import math

import torch
from einops import rearrange
from torch import Tensor, nn


class TimeEncoding(nn.Module):
    num_channels: int

    def __init__(self, feature_sizes: dict[str, int]):
        super().__init__()
        self.feature_sizes = dict(feature_sizes)

    @property
    def num_features(self) -> int:
        return len(self.feature_sizes)


class Raw(TimeEncoding):
    """The time features as they are."""

    def __init__(self, feature_sizes: dict[str, int]):
        super().__init__(feature_sizes)
        self.num_channels = self.num_features

    def forward(self, x_mark: Tensor) -> Tensor:
        return x_mark


class Cyclic(TimeEncoding):
    """sin / cos of every feature's phase, continuous across period boundaries (23:55 -> 0:00)."""

    def __init__(self, feature_sizes: dict[str, int]):
        super().__init__(feature_sizes)
        self.num_channels = 2 * self.num_features

    def forward(self, x_mark: Tensor) -> Tensor:
        phase = 2 * math.pi * x_mark
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


class Embedding(TimeEncoding):
    """A learned table per feature, indexed by the feature's integer value (STID, STAEformer)."""

    def __init__(self, feature_sizes: dict[str, int], embedding_dim: int):
        super().__init__(feature_sizes)
        self.num_channels = embedding_dim * self.num_features
        self.tables = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in feature_sizes.values()
        )
        self.register_buffer(
            "sizes", torch.tensor(list(feature_sizes.values()), dtype=torch.float), persistent=False
        )

    def forward(self, x_mark: Tensor) -> Tensor:
        if not self.tables:  # dataset without timestamps
            return x_mark[..., :0]
        index = (x_mark * self.sizes).round().long()  # (b, t, f)
        index = torch.minimum(index, self.sizes.long() - 1)
        embedded = torch.stack([table(index[..., i]) for i, table in enumerate(self.tables)])
        return rearrange(embedded, "f b t d -> b t (f d)")


class NoTime(TimeEncoding):
    """Drops the time features (ablation)."""

    num_channels = 0

    def forward(self, x_mark: Tensor) -> Tensor:
        return x_mark[..., :0]
