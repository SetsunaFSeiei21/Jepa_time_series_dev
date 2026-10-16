from typing import ClassVar

import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from .layers import LatentPredictor
from .time_encoding import TimeEncoding


class BaseModel(nn.Module):
    """Base class of every forecasting model, split into an encoder, a latent predictor and a
    decoder so that the encoder can be pretrained with JEPA.

    Shape letters: b=batch, t=time, p=predicted step, n=node (or variable), c=channel,
    f=time feature, k=graph support, l=latent token, d=latent width.

    Every model implements three methods:
        encode(x, x_mark, supports) -> z       latent of one window
        predict(z) -> z_next                   latent of the window that follows it
        decode(z_next, x, x_mark, y_mark, supports) -> forecast (b, pred_len, n, out_channels)
    and forecasting is `decode(predict(encode(x)))` (`forward`, never overridden).

    Inputs, all prepared by the trainers:
        x:        (b, t, n, c)          normalized window; missing entries are 0
        x_mark:   (b, t, f)             time features of the window's steps, values in [0, 1)
        y_mark:   (b, pred_len, f)      time features of the steps to forecast
        supports: (b, k, n, n) | None   per-sample graph supports, only for models whose yaml
                                        sets a `graph_transform`

    JEPA pretraining encodes the input window with the model and the following window with an
    EMA copy of it, and trains `predict(encode(input))` to match `encode(target)` through
    `latent_tokens`. So `encode` must accept a window of `his_len` steps for both, and the
    latent of the input window must have the same shape as that of the target window.

    `decode` reads only `z_next` in most models. Decoder-style models (e.g. Autoformer) also
    initialize their decoder from the input window and the future time features, which is why
    `decode` receives them.

    `predict` is built from `make_predictor`, which is the identity at initialization: before
    any pretraining a model therefore computes exactly `decode(encode(x))`, i.e. the original
    architecture, and training from scratch is a clean baseline.

    Time features are handled identically for all models: `encode` calls `add_time`, which
    appends the shared `time_encoding` of `x_mark` to every node; `self.in_channels` already
    counts the appended channels. A model whose design requires its own handling sets
    `custom_time_features = True` (its `in_channels` stays the data channels), never calls
    `add_time` and states why in its docstring.
    """

    custom_time_features: ClassVar[bool] = False

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        num_time_features: int,
        his_len: int,
        pred_len: int,
        time_encoding: TimeEncoding,
        predictor: dict,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_time_features = num_time_features
        self.his_len = his_len
        self.pred_len = pred_len
        self.predictor_config = predictor
        # name -> number of distinct values, in the column order of x_mark / y_mark
        self.time_feature_sizes = dict(time_encoding.feature_sizes)
        if not self.custom_time_features:
            self.time_encoding = time_encoding
            self.in_channels += time_encoding.num_channels

    def add_time(self, x: Tensor, x_mark: Tensor) -> Tensor:
        """(b, t, n, c), (b, t, f) -> (b, t, n, in_channels)"""
        time = repeat(self.time_encoding(x_mark), "b t d -> b t n d", n=x.shape[2])
        return torch.cat([x, time], dim=-1)

    def make_predictor(self, num_tokens: int, dim: int) -> LatentPredictor:
        """The model's latent predictor, with the widths shared by all models (predictor yaml).

        `num_tokens` is the length of the axis the predictor mixes (time steps, patches, ...),
        `dim` the latent width; see `LatentPredictor`.
        """
        return LatentPredictor(num_tokens, dim, **self.predictor_config)

    def forward(
        self, x: Tensor, x_mark: Tensor, y_mark: Tensor, supports: Tensor | None = None
    ) -> Tensor:
        """(b, his_len, n, c) -> (b, pred_len, n, out_channels)"""
        z = self.predict(self.encode(x, x_mark, supports))
        return self.decode(z, x, x_mark, y_mark, supports)

    def encode(self, x: Tensor, x_mark: Tensor, supports: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def predict(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(
        self,
        z: Tensor,
        x: Tensor,
        x_mark: Tensor,
        y_mark: Tensor,
        supports: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    def latent_tokens(self, z: Tensor) -> Tensor:
        """Latent (b, ..., d) with the width last -> tokens (b, l, d) compared by the JEPA loss.

        Models whose latent is not a single tensor with the width last override this.
        """
        return rearrange(z, "b ... d -> b (...) d")
