"""Forward pass of the official Autoformer / FEDformer, re-implemented with the tensor operations
of the reference code (permute / view / loops) and reading the weights of our models.

Follows https://github.com/thuml/Time-Series-Library (models/Autoformer.py, models/FEDformer.py,
layers/Autoformer_EncDec.py, layers/AutoCorrelation.py, layers/FourierCorrelation.py,
layers/Embed.py). Three deliberate deviations of our FEDformer are marked `DEVIATION` below; the
model docs explain them. Dropout is left out (eval mode).
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

from src.models import build_model

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
NUM_NODES, BATCH = 5, 2
# hourly data: four time features, as on the ETT datasets
TIME_FEATURE_SIZES = {"time_of_day": 24, "day_of_week": 7, "day_of_month": 31, "day_of_year": 366}


def build(model_name: str, task: str, seed: int = 0, **overrides):
    """Model with yaml overrides, built after seeding, and random normalized inputs
    x (b, t, n, 1), x_mark (b, t, f), y_mark (b, p, f)."""
    model_overrides = [f"model.{key}={value}" for key, value in overrides.items()]
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            "default", overrides=[f"model={model_name}", f"task={task}", *model_overrides]
        )
    torch.manual_seed(seed)
    model = build_model(
        cfg.model,
        cfg.time_encoding,
        cfg.predictor,
        None,
        num_nodes=NUM_NODES,
        in_channels=1,
        out_channels=1,
        time_feature_sizes=TIME_FEATURE_SIZES,
        his_len=cfg.task.his_len,
        pred_len=cfg.task.pred_len,
    )
    x = torch.randn(BATCH, cfg.task.his_len, NUM_NODES, 1)
    x_mark = torch.rand(BATCH, cfg.task.his_len, len(TIME_FEATURE_SIZES))
    y_mark = torch.rand(BATCH, cfg.task.pred_len, len(TIME_FEATURE_SIZES))
    return model, (x, x_mark, y_mark)


def series_decomp(x, kernel_size):
    """x (B, L, D) -> seasonal, trend: moving average with edge replication."""
    front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
    end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
    padded = torch.cat([front, x, end], dim=1)
    trend = F.avg_pool1d(padded.permute(0, 2, 1), kernel_size, stride=1).permute(0, 2, 1)
    return x - trend, trend


def my_layernorm(norm, x):
    x_hat = norm.layer_norm(x)
    bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
    return x_hat - bias


def data_embedding(embedding, x, x_mark):
    """TokenEmbedding (circular kernel-3 conv) + optional PositionalEmbedding + timeF."""
    conv = embedding.value_embedding
    pad = conv.kernel_size[0] // 2
    xp = x.permute(0, 2, 1)
    xp = torch.cat([xp[..., -pad:], xp, xp[..., :pad]], dim=-1)
    h = F.conv1d(xp, conv.weight).transpose(1, 2)
    if embedding.positions is not None:
        d_model = h.shape[-1]
        position = torch.arange(0, x.shape[1]).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe = torch.zeros(x.shape[1], d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        h = h + pe.unsqueeze(0)
    return h + x_mark @ embedding.time_embedding.weight.T


def auto_correlation(inner, queries, keys, values):
    """AutoCorrelation with the inference-phase aggregation (per-sample delays)."""
    B, L, H, E = queries.shape
    _, S, _, D = values.shape
    if L > S:
        zeros = torch.zeros_like(queries[:, : (L - S), :]).float()
        values = torch.cat([values, zeros], dim=1)
        keys = torch.cat([keys, zeros], dim=1)
    else:
        values = values[:, :L, :, :]
        keys = keys[:, :L, :, :]
    q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
    k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
    corr = torch.fft.irfft(q_fft * torch.conj(k_fft), dim=-1)
    values = values.permute(0, 2, 3, 1).contiguous()
    init_index = torch.arange(L).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, H, E, 1)
    top_k = int(inner.factor * math.log(L))
    mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
    weights, delay = torch.topk(mean_value, top_k, dim=-1)
    tmp_corr = torch.softmax(weights, dim=-1)
    tmp_values = values.repeat(1, 1, 1, 2)
    delays_agg = torch.zeros_like(values).float()
    for i in range(top_k):
        tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
            1, H, E, L
        )
        pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
        delays_agg = delays_agg + pattern * (
            tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, H, E, L)
        )
    return delays_agg.permute(0, 3, 1, 2)


def compl_mul(order, x, weights):
    return torch.complex(
        torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
        torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real),
    )


def fourier_block(inner, q):
    B, L, H, E = q.shape
    x_ft = torch.fft.rfft(q.permute(0, 2, 3, 1), dim=-1)
    out_ft = torch.zeros(B, H, E, L // 2 + 1, dtype=torch.cfloat)
    weights = torch.complex(inner.weight_real, inner.weight_imag)
    for wi, i in enumerate(inner.index.tolist()):
        # DEVIATION: the reference writes this mode at position wi (its rank among the kept
        # modes) instead of at its own frequency i.
        out_ft[:, :, :, i] = compl_mul("bhi,hio->bho", x_ft[:, :, :, i], weights[:, :, :, wi])
    # DEVIATION: the reference returns (B, H, E, L), which the attention layer then views as
    # (B, L, H * E); here the time axis is moved back first.
    return torch.fft.irfft(out_ft, n=L).permute(0, 3, 1, 2)


def fourier_cross_attention(inner, q, k):
    B, L, H, E = q.shape
    xq_ft = torch.fft.rfft(q.permute(0, 2, 3, 1), dim=-1)
    xk_ft = torch.fft.rfft(k.permute(0, 2, 3, 1), dim=-1)
    xq_ft_ = torch.stack([xq_ft[..., j] for j in inner.index_q.tolist()], dim=-1)
    xk_ft_ = torch.stack([xk_ft[..., j] for j in inner.index_kv.tolist()], dim=-1)
    xqk_ft = compl_mul("bhex,bhey->bhxy", xq_ft_, xk_ft_)
    if inner.activation == "tanh":
        xqk_ft = torch.complex(xqk_ft.real.tanh(), xqk_ft.imag.tanh())
    else:
        xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
        xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
    xqkv_ft = compl_mul("bhxy,bhey->bhex", xqk_ft, xk_ft_)
    weights = torch.complex(inner.weight_real, inner.weight_imag)
    xqkvw = compl_mul("bhex,heox->bhox", xqkv_ft, weights)
    out_ft = torch.zeros(B, H, E, L // 2 + 1, dtype=torch.cfloat)
    for i, j in enumerate(inner.index_q.tolist()):
        out_ft[:, :, :, j] = xqkvw[:, :, :, i]
    # DEVIATION: the reference divides out_ft by in_channels * out_channels (= d_model^2) here,
    # on top of the same factor in the initialization of the weights.
    out = torch.fft.irfft(out_ft, n=L)
    return out.permute(0, 3, 1, 2)  # DEVIATION: time axis moved back, see fourier_block


def attention_layer(layer, queries, keys):
    """AutoCorrelationLayer: projections, inner mechanism, output projection."""
    B, L, _ = queries.shape
    _, S, _ = keys.shape
    H = layer.n_heads
    inner = layer.inner
    name = type(inner).__name__
    q = layer.queries_projection(queries).view(B, L, H, -1)
    if name == "AutoCorrelation":
        k = layer.keys_projection(keys).view(B, S, H, -1)
        v = layer.values_projection(keys).view(B, S, H, -1)
        out = auto_correlation(inner, q, k, v)
    elif name == "FourierBlock":
        out = fourier_block(inner, q)
    else:
        out = fourier_cross_attention(inner, q, layer.keys_projection(keys).view(B, S, H, -1))
    return layer.out_projection(out.reshape(B, L, -1))


def feed_forward(ff, x):
    y = F.conv1d(x.transpose(-1, 1), ff.linear1.weight.unsqueeze(-1))
    y = F.relu(y)
    return F.conv1d(y, ff.linear2.weight.unsqueeze(-1)).transpose(-1, 1)


def official_forward(model, x, x_mark, y_mark):
    """x (B, L, N * C) normalized window -> (B, pred_len, N * C)"""
    net = model.network
    kernel = net.decomposition.moving_average.kernel_size
    label_len, pred_len = net.label_len, net.pred_len
    mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, pred_len, 1)
    zeros = torch.zeros([x.shape[0], pred_len, x.shape[2]])
    seasonal_init, trend_init = series_decomp(x, kernel)
    trend_init = torch.cat([trend_init[:, -label_len:, :], mean], dim=1)
    seasonal_init = torch.cat([seasonal_init[:, -label_len:, :], zeros], dim=1)

    enc = data_embedding(net.encoder_embedding, x, x_mark)
    for layer in net.encoder_layers:
        enc = enc + attention_layer(layer.attention, enc, enc)
        enc, _ = series_decomp(enc, kernel)
        enc, _ = series_decomp(enc + feed_forward(layer.feed_forward, enc), kernel)
    enc = my_layernorm(net.encoder_norm, enc)

    dec = data_embedding(
        net.decoder_embedding, seasonal_init, torch.cat([x_mark[:, -label_len:], y_mark], 1)
    )
    trend = trend_init
    for layer in net.decoder_layers:
        dec, trend1 = series_decomp(dec + attention_layer(layer.self_attention, dec, dec), kernel)
        dec, trend2 = series_decomp(dec + attention_layer(layer.cross_attention, dec, enc), kernel)
        dec, trend3 = series_decomp(dec + feed_forward(layer.feed_forward, dec), kernel)
        residual = (trend1 + trend2 + trend3).permute(0, 2, 1)
        conv = layer.trend_projection
        pad = conv.kernel_size[0] // 2
        residual = torch.cat([residual[..., -pad:], residual, residual[..., :pad]], dim=-1)
        trend = trend + F.conv1d(residual, conv.weight).transpose(1, 2)
    seasonal = net.projection(my_layernorm(net.decoder_norm, dec))
    return (trend + seasonal)[:, -pred_len:, :]
