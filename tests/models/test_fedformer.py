"""FEDformer: the untrained model equals the official forward pass (up to the two documented
deviations marked in tslib_decomposition.py), and the Fourier modes travel with the checkpoint.

`official_forward` re-implements the official forward with the reference code's tensor
operations, reading the weights and Fourier modes of our model. Since the latent predictor is
the identity at initialization, `model(x)` must match it.
"""

import pytest
import torch
from einops import rearrange

from tests.models.tslib_decomposition import build, official_forward


@pytest.mark.parametrize("label_ratio", [1.0, 0.5])
@pytest.mark.parametrize("task", ["short", "long"])
def test_matches_official_forward(task, label_ratio):
    model, (x, x_mark, y_mark) = build("fedformer", task, label_ratio=label_ratio)
    model.eval()
    with torch.no_grad():
        ours = model(x, x_mark, y_mark)
        reference = official_forward(model, rearrange(x, "b t n c -> b t (n c)"), x_mark, y_mark)
    torch.testing.assert_close(rearrange(ours, "b p n c -> b p (n c)"), reference)


def test_modes_restored_from_checkpoint():
    """Randomly selected modes are buffers: a model built under another random state uses the
    checkpoint's modes once it is loaded."""
    model, (x, x_mark, y_mark) = build("fedformer", "long", seed=0)
    other, _ = build("fedformer", "long", seed=1)
    block = model.network.encoder_layers[0].attention.inner
    other_block = other.network.encoder_layers[0].attention.inner
    assert not torch.equal(block.index, other_block.index)
    other.load_state_dict(model.state_dict())
    assert torch.equal(block.index, other_block.index)
    model.eval()
    other.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x, x_mark, y_mark), other(x, x_mark, y_mark))


def test_blocks_shared_across_layers():
    model, _ = build("fedformer", "long")
    encoder, decoder = model.network.encoder_layers, model.network.decoder_layers
    assert encoder[0].attention.inner is encoder[1].attention.inner
    assert decoder[0].self_attention.inner is decoder[1].self_attention.inner
    assert decoder[0].cross_attention.inner is decoder[1].cross_attention.inner
