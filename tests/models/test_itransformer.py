"""iTransformer: the untrained model equals the official forward pass.

`official_forward` re-implements `Model.forecast` of the official code
(https://github.com/thuml/iTransformer, model/iTransformer.py with layers/Embed.py,
layers/SelfAttention_Family.py and layers/Transformer_EncDec.py) with its own tensor operations
(permute / view, torch.einsum attention, kernel-1 Conv1d feed-forward), reading the weights of
our model. The official model z-scores each window itself (`use_norm`); here the trainers do it,
so the reference starts from the normalized window. Without time features (`time_encoding=none`,
the official setting `x_mark_enc=None`) the token sets coincide, and since the latent predictor
is the identity at initialization, `model(x)` must match the reference.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.test_models import setup


def official_attention(layer, x):
    """AttentionLayer(FullAttention) of the official code, without dropout."""
    attention = layer.attention
    batch, length, _ = x.shape
    heads = attention.n_heads
    q = attention.queries_projection(x).view(batch, length, heads, -1)
    k = attention.keys_projection(x).view(batch, length, heads, -1)
    v = attention.values_projection(x).view(batch, length, heads, -1)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("blhe,bshe->bhls", q, k)
    weights = torch.softmax(scale * scores, dim=-1)
    out = torch.einsum("bhls,bshd->blhd", weights, v).contiguous().view(batch, length, -1)
    return attention.out_projection(out)


def official_encoder_layer(layer, x):
    """EncoderLayer of the official code (feed-forward as kernel-1 convolutions)."""
    x = layer.norm1(x + official_attention(layer, x))
    conv1 = layer.linear1.weight.unsqueeze(-1), layer.linear1.bias
    conv2 = layer.linear2.weight.unsqueeze(-1), layer.linear2.bias
    y = F.relu(F.conv1d(x.transpose(-1, 1), *conv1))
    y = F.conv1d(y, *conv2).transpose(-1, 1)
    return layer.norm2(x + y)


def official_forward(model, x):
    """Model.forecast with x_mark_enc=None: (b, t, n) -> (b, p, n)."""
    num_variates = x.shape[2]
    enc_out = model.embedding(x.permute(0, 2, 1))  # DataEmbedding_inverted
    for layer in model.layers:
        enc_out = official_encoder_layer(layer, enc_out)
    enc_out = model.norm(enc_out)
    return model.projection(enc_out).permute(0, 2, 1)[:, :, :num_variates]


@pytest.mark.parametrize("task", ["short", "long"])
def test_matches_official_forward(task):
    cfg, model, _, (x, x_mark, y_mark) = setup("itransformer", task, time_encoding="none")
    assert cfg.model.activation == "relu"  # the activation of official_encoder_layer
    model.eval()
    with torch.no_grad():
        ours = model(x, x_mark, y_mark)
        reference = official_forward(model, x[..., 0]).unsqueeze(-1)
    torch.testing.assert_close(ours, reference, rtol=1e-5, atol=1e-5)
