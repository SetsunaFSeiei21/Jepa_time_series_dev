import os
import hydra
import torch
import numpy as np
from omegaconf import DictConfig

from src.factories.loader import get_loaders
from src.factories.model import ModelFactory
from src.base.data import BatchData


def load_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded checkpoint: {ckpt_path}")
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    return model


@torch.no_grad()
def analyze_node_rmse(model, loader, scalar, device, prefix="Test", top_k=20):
    model.eval()

    node_sse = None
    node_count = None

    for batch_idx, batch_data in enumerate(loader):
        batch_data: BatchData
        batch = batch_data.to_device(device)

        # 1. 处理 mask，和 ExpertEngine._evaluate() 保持一致
        if batch.input_mask is not None:
            batch.input_seq = torch.where(
                batch.input_mask,
                torch.zeros_like(batch.input_seq),
                batch.input_seq,
            )

        if batch.target_mask is not None:
            batch.target_seq = torch.where(
                batch.target_mask,
                torch.zeros_like(batch.target_seq),
                batch.target_seq,
            )

        batch.input_seq = torch.nan_to_num(
            batch.input_seq,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        batch.target_seq = torch.nan_to_num(
            batch.target_seq,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # 2. valid mask
        if batch.target_mask is None:
            valid_mask = torch.ones_like(batch.target_seq, dtype=torch.bool)
        else:
            valid_mask = ~batch.target_mask

        valid_mask = valid_mask & torch.isfinite(batch.target_seq)

        if valid_mask.sum().item() == 0:
            print(f"Skip batch {batch_idx}: no valid target values.")
            continue

        # 3. 归一化 input
        batch.input_seq = scalar.fit_transform(batch.input_seq)
        batch.input_seq = torch.nan_to_num(
            batch.input_seq,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # 4. forward
        model_input = batch.to_dict()
        pred_norm = model(**model_input)

        # 5. 反归一化回原始尺度，和你原来的 evaluation 一致
        pred = scalar.inverse_transform(pred_norm)

        valid_mask = valid_mask & torch.isfinite(pred)

        if valid_mask.sum().item() == 0:
            print(f"Skip batch {batch_idx}: no finite prediction.")
            continue

        # pred / target shape: (B, T, N, 1)
        err2 = (pred - batch.target_seq) ** 2
        valid_float = valid_mask.float()

        # 对 B、T、channel 求和，只保留 N
        batch_node_sse = (err2 * valid_float).sum(dim=(0, 1, 3))
        batch_node_count = valid_float.sum(dim=(0, 1, 3))

        if node_sse is None:
            node_sse = batch_node_sse.detach().cpu()
            node_count = batch_node_count.detach().cpu()
        else:
            node_sse += batch_node_sse.detach().cpu()
            node_count += batch_node_count.detach().cpu()

    if node_sse is None:
        raise RuntimeError("No valid batches found.")

    eps = 1e-12

    node_mse = node_sse / node_count.clamp(min=1)
    node_rmse = torch.sqrt(node_mse)

    total_sse = node_sse.sum()
    total_count = node_count.sum()
    global_rmse = torch.sqrt(total_sse / total_count.clamp(min=1))

    contribution = node_sse / (total_sse + eps)
    idx = torch.argsort(contribution, descending=True)

    print(f"\n[{prefix}] Global RMSE: {global_rmse.item():.6f}")
    print(f"\n[{prefix}] Top-{top_k} node RMSE contribution:")

    for rank, i in enumerate(idx[:top_k]):
        i = i.item()
        print(
            f"rank={rank+1:02d}, "
            f"node={i}, "
            f"node_rmse={node_rmse[i].item():.6f}, "
            f"contribution={contribution[i].item() * 100:.2f}%, "
            f"count={int(node_count[i].item())}"
        )

    return node_rmse, contribution


@hydra.main(
    version_base="1.3",
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="default",
)
def main(cfg: DictConfig):
    dataset_kwargs = cfg["dataset"]
    model_kwargs = cfg["model"]
    exp_kwargs = cfg["exp"]
    task_exp = cfg["task"]

    ckpt_path = cfg.get("ckpt_path", None)
    split = cfg.get("split", "test")
    top_k = int(cfg.get("top_k", 20))

    if ckpt_path is None:
        raise ValueError(
            "Please provide checkpoint path, e.g. "
            "+ckpt_path=/path/to/best_model_checkpoint.pth"
        )

    his_len = task_exp["his_len"]
    pred_len = task_exp["pred_len"]

    device_id = exp_kwargs["device_id"]
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    meta_path = cfg["meta_path"]

    loaders, adj_mtx, norm_pipeline, n_feat = get_loaders(
        meta_path,
        dataset_kwargs,
        exp_kwargs,
        his_len,
        pred_len,
        device,
    )

    n_node = adj_mtx.shape[0]

    ext_kwargs = dict(
        model_kwargs,
        **task_exp,
        **{
            "node_num": n_node,
            "input_dim": n_feat + 1,
            "output_dim": 1,
        },
    )

    model_name = model_kwargs["_target_"].split(".")[-1]

    model = ModelFactory.create_model(
        model_name,
        ext_kwargs,
        adj_mtx,
        device,
    )

    model = load_checkpoint(model, ckpt_path, device)
    model.to(device)

    if split == "val":
        loader = loaders["val_loader"]
    elif split == "test":
        loader = loaders["test_loader"]
    elif split == "train":
        loader = loaders["train_loader"]
    else:
        raise ValueError(f"Unknown split: {split}")

    analyze_node_rmse(
        model=model,
        loader=loader,
        scalar=norm_pipeline,
        device=device,
        prefix=split,
        top_k=top_k,
    )


if __name__ == "__main__":
    main()