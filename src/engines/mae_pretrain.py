import os
import sys
from datetime import datetime
from time import time
from typing import Any, Dict, Tuple

import torch
from torch import optim
from tqdm import tqdm

from src.base.data import BatchData
from src.base.engine import BaseEngine
from src.losses.mae_loss import masked_mae_loss
from src.utils.ssl_heads import (
    MAEReconstructionHead,
    canonicalize_hidden_for_mae,
)


class MAEPretrainEngine(BaseEngine):
    """
    Engine for MAE-style self-supervised pretraining.

    Training objective:
        1. Randomly mask part of input_seq to zero.
        2. Encode masked input sequence.
        3. Reconstruct original normalized values with a lightweight head.
        4. Compute reconstruction loss only on artificially masked positions.

    This engine does not change the backbone architecture permanently.
    The reconstruction head is only used during MAE pretraining.
    """

    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        logger,
        save_path: str,
        scalar,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        loss_func,
        config: Dict[str, Any],
        model_name: str,
    ):
        super().__init__(device, model, logger, save_path)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scalar = scalar
        self.model_name = model_name

        self.max_epochs = config["max_epochs"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.milestones = config["milestones"]
        self.gamma = config["gamma"]
        self.clip_grad_value = config["clip_grad_value"]
        self.accumulation_steps = config["accumulation_steps"]
        self.save_freq = config["save_freq"]

        self.mask_ratio = float(config.get("mae_mask_ratio", 0.25))
        self.mae_loss_type = config.get("mae_loss_type", "l1")

        # Pure MAE usually reconstructs directly from encoder output.
        # If set true, it will pass hidden through model.predict(...) before reconstruction.
        # For your current first version, keep it False.
        self.mae_use_predictor = bool(config.get("mae_use_predictor", False))

        self.node_num = int(getattr(self.model, "node_num"))
        self.hidden_dim = int(getattr(self.model, "d_model"))

        self.reconstruction_head = MAEReconstructionHead(
            hidden_dim=self.hidden_dim,
            node_num=self.node_num,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.reconstruction_head.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
        )

    def _make_random_mae_mask(
        self,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create an artificial MAE mask.

        valid_mask:
            Boolean tensor with shape (B,T,N,1).
            True means this position is valid original data and can be masked.

        return:
            mae_mask:
                Boolean tensor with shape (B,T,N,1).
                True means artificially masked for reconstruction.
        """
        rand = torch.rand(valid_mask.shape, device=valid_mask.device)
        mae_mask = (rand < self.mask_ratio) & valid_mask

        # Fallback: if the sampled mask is empty, force at least one valid position.
        if mae_mask.sum().item() == 0:
            valid_indices = valid_mask.flatten().nonzero(as_tuple=False).flatten()

            if valid_indices.numel() == 0:
                return mae_mask

            num_to_mask = max(1, int(valid_indices.numel() * self.mask_ratio))
            perm = torch.randperm(valid_indices.numel(), device=valid_mask.device)
            chosen = valid_indices[perm[:num_to_mask]]

            flat_mask = torch.zeros_like(valid_mask.flatten(), dtype=torch.bool)
            flat_mask[chosen] = True
            mae_mask = flat_mask.view_as(valid_mask)

        return mae_mask

    def _encode_for_mae(
        self,
        input_seq: torch.Tensor,
        input_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode masked input sequence and convert hidden representation to canonical
        format for MAE reconstruction.
        """
        hidden = self.model.encode(input_seq, input_features)

        if self.mae_use_predictor:
            hidden = self.model.predict(hidden)

        hidden = canonicalize_hidden_for_mae(
            hidden=hidden,
            model_name=self.model_name,
            node_num=self.node_num,
            hidden_dim=self.hidden_dim,
        )

        return hidden

    def train_epoch(self) -> Tuple[bool, float]:
        self.model.train()
        self.reconstruction_head.train()

        avg_loss = 0.0
        terminate = False

        accumulation_steps = max(1, int(self.accumulation_steps))
        valid_micro_steps = 0

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(self.train_loader, desc="MAE Pretraining", file=sys.stdout)

        for batch_idx, batch_data in enumerate(pbar):
            batch_data: BatchData
            batch = batch_data.to_device(self.device)

            # ------------------------------------------------------------
            # 1. Build clean input and original-valid mask.
            # ------------------------------------------------------------
            input_seq = batch.input_seq

            if batch.input_mask is None:
                original_invalid_mask = torch.zeros_like(input_seq, dtype=torch.bool)
            else:
                original_invalid_mask = batch.input_mask.bool()

            input_seq = torch.where(
                original_invalid_mask,
                torch.zeros_like(input_seq),
                input_seq,
            )

            input_seq = torch.nan_to_num(
                input_seq,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # ------------------------------------------------------------
            # 2. Normalize clean input sequence.
            #    MAE reconstructs normalized values for numerical stability.
            # ------------------------------------------------------------
            input_norm = self.scalar.fit_transform(input_seq)

            finite_mask = torch.isfinite(input_norm)

            input_norm = torch.nan_to_num(
                input_norm,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            valid_mask = (~original_invalid_mask) & finite_mask

            if valid_mask.sum().item() == 0:
                self.logger.warning(
                    f"Skip batch {batch_idx}: no valid input values for MAE."
                )
                continue

            # ------------------------------------------------------------
            # 3. Randomly mask part of the valid normalized input.
            # ------------------------------------------------------------
            mae_mask = self._make_random_mae_mask(valid_mask)

            if mae_mask.sum().item() == 0:
                self.logger.warning(
                    f"Skip batch {batch_idx}: MAE sampled empty mask."
                )
                continue

            masked_input_norm = torch.where(
                mae_mask,
                torch.zeros_like(input_norm),
                input_norm,
            )

            # ------------------------------------------------------------
            # 4. Encode and reconstruct.
            # ------------------------------------------------------------
            hidden = self._encode_for_mae(
                input_seq=masked_input_norm,
                input_features=batch.input_features,
            )

            recon_norm = self.reconstruction_head(
                hidden,
                target_len=input_norm.size(1),
            )

            if recon_norm.shape != input_norm.shape:
                self.logger.error(
                    f"MAE reconstruction shape mismatch at batch {batch_idx}: "
                    f"recon_norm.shape={tuple(recon_norm.shape)}, "
                    f"input_norm.shape={tuple(input_norm.shape)}."
                )
                terminate = True
                break

            if torch.isnan(recon_norm).any() or torch.isinf(recon_norm).any():
                self.logger.error(
                    f"MAE reconstruction contains NaN/Inf at batch {batch_idx}."
                )
                terminate = True
                break

            # ------------------------------------------------------------
            # 5. Compute reconstruction loss only on artificial mask positions.
            # ------------------------------------------------------------
            try:
                loss = masked_mae_loss(
                    pred=recon_norm,
                    target=input_norm,
                    mask=mae_mask,
                    loss_type=self.mae_loss_type,
                )
            except ValueError as exc:
                self.logger.warning(f"Skip batch {batch_idx}: {exc}")
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(
                    f"MAE pretraining loss is NaN/Inf at batch {batch_idx}."
                )
                terminate = True
                break

            # ------------------------------------------------------------
            # 6. Backward with gradient accumulation.
            # ------------------------------------------------------------
            loss_for_backward = loss / accumulation_steps
            loss_for_backward.backward()

            valid_micro_steps += 1

            avg_loss = avg_loss * ((valid_micro_steps - 1) / valid_micro_steps) + (
                loss.item() / valid_micro_steps
            )

            accum_display = valid_micro_steps % accumulation_steps
            if accum_display == 0:
                accum_display = accumulation_steps

            pbar.set_description(
                "MAE Loss: "
                f"{loss.item():.4f}, "
                f"Avg: {avg_loss:.4f}, "
                f"Mask: {mae_mask.sum().item()}, "
                f"Accum: {accum_display}/{accumulation_steps}"
            )

            should_update = valid_micro_steps % accumulation_steps == 0

            if should_update:
                if self.clip_grad_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters())
                        + list(self.reconstruction_head.parameters()),
                        self.clip_grad_value,
                    )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        # ------------------------------------------------------------
        # 7. Step remaining accumulated gradients.
        # ------------------------------------------------------------
        if (
            not terminate
            and valid_micro_steps > 0
            and valid_micro_steps % accumulation_steps != 0
        ):
            if self.clip_grad_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters())
                    + list(self.reconstruction_head.parameters()),
                    self.clip_grad_value,
                )

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        if valid_micro_steps == 0:
            self.logger.error("No valid MAE pretraining batches in this epoch.")
            terminate = True

        return terminate, avg_loss

    def save_checkpoint(self, epoch: int):
        """
        Save MAE checkpoint.

        Downstream finetune.py will load only checkpoint["model_state_dict"].
        The reconstruction head is saved only for debugging/resume reference.
        """
        checkpoint = {
            "epoch": epoch,
            "pretrain_method": "mae",
            "model_state_dict": self.model.state_dict(),
            "reconstruction_head_state_dict": self.reconstruction_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "mae_config": {
                "mask_ratio": self.mask_ratio,
                "loss_type": self.mae_loss_type,
                "mae_use_predictor": self.mae_use_predictor,
            },
        }

        path = os.path.join(self.save_path, f"mae_pretrain_epoch_{epoch}.pth")
        torch.save(checkpoint, path)

    def run(self):
        log_metadata = {
            "training_start_time": datetime.now().isoformat(),
            "mode": "mae_pretrain",
            "max_epochs": self.max_epochs,
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "milestones": str(self.milestones),
            "gamma": self.gamma,
            "clip_grad_value": self.clip_grad_value,
            "accumulation_steps": self.accumulation_steps,
            "save_freq": self.save_freq,
            "save_dir": self.save_path,
            "mae_mask_ratio": self.mask_ratio,
            "mae_loss_type": self.mae_loss_type,
            "mae_use_predictor": self.mae_use_predictor,
            "epoch_logs": {},
        }

        self.initialize_log_file(log_metadata)
        self.logger.info("Starting MAE pretraining...")

        for epoch in range(1, self.max_epochs + 1):
            start = time()
            terminate, train_loss = self.train_epoch()
            train_time = time() - start

            self.logger.info(
                f"[MAE Pretrain] Epoch {epoch}/{self.max_epochs}, "
                f"loss={train_loss:.6f}, time={train_time:.2f}s"
            )

            if terminate:
                break

            if epoch % self.save_freq == 0:
                self.save_checkpoint(epoch)

            train_memory_reserved = torch.cuda.memory_reserved()
            train_memory_allocated = torch.cuda.memory_allocated()
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(f"MAE pretrain loss: {train_loss}")
            self.logger.info(f"MAE pretrain time: {train_time}s")
            self.logger.info(
                f"MAE pretrain memory reserved: {train_memory_reserved / (1024 ** 2):.2f} MB"
            )
            self.logger.info(
                f"MAE pretrain memory allocated: {train_memory_allocated / (1024 ** 2):.2f} MB"
            )
            self.logger.info(f"Current learning rate: {current_lr}")

            pretrain_log = {
                "timestamp": datetime.now().isoformat(),
                "current_learning_rate": current_lr,
                "loss": train_loss,
                "time_seconds": float(train_time),
                "memory_reserved_mb": float(train_memory_reserved / (1024**2)),
                "memory_allocated_mb": float(train_memory_allocated / (1024**2)),
            }

            self._log_epoch_info(epoch=f"MAE Pretrain {epoch}", epoch_log=pretrain_log)

            self.scheduler.step()

        self.logger.info("MAE pretraining completed.")