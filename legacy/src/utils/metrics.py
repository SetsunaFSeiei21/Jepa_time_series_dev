import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_mse(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def crps_normal(mu, sigma, obs):
    sigma = torch.abs(sigma) + 1e-8
    z = (obs - mu) / sigma

    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2))) 
    phi = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)  

    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))

    return crps.mean()


def compute_all_metrics(pred: Tensor, label: Tensor, pred_std: Optional[Tensor] = None):
    result = {}
    mae = F.l1_loss(pred, label).item()
    rmse = F.mse_loss(pred, label).sqrt().item()
    result["MAE"] = mae
    result["RMSE"] = rmse
    if pred_std is not None:
        result["Coverage"] = (
            ((label < (pred + pred_std)) & (label > (pred - pred_std)))
            .float()
            .mean()
            .item()
        )
        result["Interval Width"] = (pred_std.mean() * 2).item()
        result["CRPS"] = crps_normal(pred, pred_std, label).item()
    return result
