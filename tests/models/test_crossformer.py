"""Crossformer: the untrained model equals the official forward pass.

`official_forward` re-implements the forward of the official code
(https://github.com/Thinklab-SJTU/Crossformer, cross_models/) with its own tensor operations
(front padding by the first value, DSW embedding, TSA layers, SegMerging that appends the last
`pad_num` segments, decoder layers whose forecasts are summed), reading the weights of our
model. Where our attention has a single key it builds no query / key projections; the
reference then uses freshly drawn ones, which must not change anything. At initialization the
latent predictor is the identity, so `model(x)` must match the reference.
"""

import math

import pytest
import torch
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from torch import nn

from src.models import build_model
from tests.test_models import CONFIG_DIR

NUM_NODES, BATCH, CHANNELS = 5, 3, 1
TIME_FEATURE_SIZES = {"time_of_day": 288, "day_of_week": 7, "day_of_month": 31, "day_of_year": 366}
# (task, his_len = pred_len, seg_len). 96 / 8: scales of 12, 12, 3, 1 segments (merging pads
# one segment, the last scale has a single segment); 90 / 8 needs front padding and cropping;
# 192 / 8: merging 6 segments pads two (official: the last two are appended again).
CASES = [("long", 96, 8), ("long", 90, 8), ("long192", 192, 8)]


def build(task: str, length: int, seg_len: int):
    overrides = [
        "model=crossformer",
        f"task={task}",
        f"task.his_len={length}",
        f"task.pred_len={length}",
        f"model.seg_len={seg_len}",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose("default", overrides=overrides)
    torch.manual_seed(0)
    model = build_model(
        cfg.model,
        cfg.time_encoding,
        cfg.predictor,
        None,
        num_nodes=NUM_NODES,
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        time_feature_sizes=TIME_FEATURE_SIZES,
        his_len=length,
        pred_len=length,
    )
    # Move LayerNorm away from the identity so that its placement is tested too.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                module.weight.add_(0.1 * torch.randn_like(module.weight))
                module.bias.add_(0.1 * torch.randn_like(module.bias))
    return cfg, model.eval()


def official_attention(attention, queries, keys):
    """Official AttentionLayer + FullAttention (eval mode): (B, L, D), (B, S, D) -> (B, L, D)."""
    d_model, heads = attention.value.in_features, attention.n_heads
    query = attention.query or nn.Linear(d_model, d_model)
    key = attention.key or nn.Linear(d_model, d_model)
    B, L, _ = queries.shape
    S = keys.shape[1]
    q = query(queries).view(B, L, heads, -1)
    k = key(keys).view(B, S, heads, -1)
    v = attention.value(keys).view(B, S, heads, -1)
    scale = 1.0 / math.sqrt(q.shape[-1])
    A = torch.softmax(scale * torch.einsum("blhe,bshe->bhls", q, k), dim=-1)
    out = torch.einsum("bhls,bshd->blhd", A, v).contiguous().view(B, L, -1)
    return attention.out(out)


def official_tsa(tsa, x):
    batch = x.shape[0]
    time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")
    dim_in = tsa.time_norm(time_in + official_attention(tsa.time_attention, time_in, time_in))
    dim_in = tsa.time_ff_norm(dim_in + tsa.time_ff(dim_in))
    dim_send = rearrange(dim_in, "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model", b=batch)
    batch_router = repeat(
        tsa.router, "seg_num f d_model -> (repeat seg_num) f d_model", repeat=batch
    )
    dim_buffer = official_attention(tsa.dim_sender, batch_router, dim_send)
    dim_receive = official_attention(tsa.dim_receiver, dim_send, dim_buffer)
    dim_enc = tsa.dim_norm(dim_send + dim_receive)
    dim_enc = tsa.dim_ff_norm(dim_enc + tsa.dim_ff(dim_enc))
    return rearrange(dim_enc, "(b seg_num) ts_d d_model -> b ts_d seg_num d_model", b=batch)


def official_merge(merging, x):
    win_size = merging.win_size
    pad_num = x.shape[2] % win_size
    if pad_num != 0:
        pad_num = win_size - pad_num
        x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)
    x = torch.cat([x[:, :, i::win_size, :] for i in range(win_size)], -1)
    return merging.linear(merging.norm(x))


def official_forward(model, x_seq):
    """x_seq (B, in_len, D) -> (B, out_len, D), cross_models/cross_former.py."""
    batch_size, in_len, ts_dim = x_seq.shape
    seg_len = model.seg_len
    in_len_add = math.ceil(in_len / seg_len) * seg_len - in_len
    if in_len_add != 0:
        x_seq = torch.cat((x_seq[:, :1, :].expand(-1, in_len_add, -1), x_seq), dim=1)
    x_seq = rearrange(x_seq, "b (seg_num seg_len) d -> (b d seg_num) seg_len", seg_len=seg_len)
    x_seq = rearrange(
        model.embedding(x_seq),
        "(b d seg_num) d_model -> b d seg_num d_model",
        b=batch_size,
        d=ts_dim,
    )
    x_seq = model.pre_norm(x_seq + model.position)

    enc_out = [x_seq]
    for block in model.encoder_blocks:
        if isinstance(block.merge, nn.Identity):
            x_seq = block.merge(x_seq)
        else:
            x_seq = official_merge(block.merge, x_seq)
        for layer in block.layers:
            x_seq = official_tsa(layer, x_seq)
        enc_out.append(x_seq)

    x = repeat(model.decoder_position, "ts_d l d -> b ts_d l d", b=batch_size)
    final_predict = None
    for layer, cross in zip(model.decoder_layers, enc_out, strict=True):
        x = official_tsa(layer.self_attention, x)
        x = rearrange(x, "b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model")
        cross = rearrange(cross, "b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model")
        y = x = layer.cross_norm(x + official_attention(layer.cross_attention, x, cross))
        dec_output = layer.mlp_norm(x + layer.mlp(y))
        dec_output = rearrange(
            dec_output, "(b ts_d) seg_dec_num d -> b ts_d seg_dec_num d", b=batch_size
        )
        x = dec_output
        layer_predict = rearrange(
            layer.forecast(dec_output), "b out_d seg_num seg_len -> b (out_d seg_num) seg_len"
        )
        final_predict = layer_predict if final_predict is None else final_predict + layer_predict
    final_predict = rearrange(
        final_predict,
        "b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d",
        out_d=x_seq.shape[1],
    )
    return final_predict[:, : model.pred_len]


@pytest.mark.parametrize(("task", "length", "seg_len"), CASES)
def test_matches_official_forward(task, length, seg_len):
    _, model = build(task, length, seg_len)
    x = torch.randn(BATCH, length, NUM_NODES, CHANNELS)
    marks = torch.zeros(BATCH, length, len(TIME_FEATURE_SIZES))
    with torch.no_grad():
        ours = model(x, marks, marks)
        reference = official_forward(model, x.squeeze(-1))
    torch.testing.assert_close(ours.squeeze(-1), reference, rtol=1e-5, atol=1e-5)


def test_single_key_attention_has_no_query_key():
    """A scale with a single segment builds no query / key projections (they could not act)."""
    _, model = build("long", 96, 8)
    last_scale = model.encoder_blocks[-1].layers[0].time_attention
    assert last_scale.query is None and last_scale.key is None
    assert model.decoder_layers[-1].cross_attention.query is None
    assert model.decoder_layers[0].cross_attention.query is not None


def test_latent_tokens_concatenate_scales():
    _, model = build("long", 96, 8)
    x = torch.randn(BATCH, 96, NUM_NODES, CHANNELS)
    marks = torch.zeros(BATCH, 96, len(TIME_FEATURE_SIZES))
    with torch.no_grad():
        z = model.encode(x, marks)
    assert [scale.shape[2] for scale in z] == [12, 12, 3, 1]
    tokens = model.latent_tokens(z)
    assert tokens.shape == (BATCH, NUM_NODES * (12 + 12 + 3 + 1), model.position.shape[-1])
