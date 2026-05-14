import os
import sys
from datetime import datetime
from time import time
from typing import Dict, Any, Tuple
from logging import Logger

import torch
from torch import optim
from tqdm import tqdm

from src.base.data import BatchData
from src.base.engine import BaseEngine
from src.losses.jepa_loss import JEPASIGRegLoss


class JEPAPretrainEngine(BaseEngine):
    """
    Engine for JEPA-style pretraining.

    It calls:
        Ey, Ey_pred = model(..., mode="pretrain")

    Then optimizes:
        MSE(Ey_pred, Ey) + alpha * SIGReg(...)
    """

    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        logger: Logger,
        save_path: str,
        scalar,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        loss_func,
        config: Dict[str, Any],
    ):
        super().__init__(device, model, logger, save_path)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scalar = scalar

        self.max_epochs = config["max_epochs"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.milestones = config["milestones"]
        self.gamma = config["gamma"]
        self.clip_grad_value = config["clip_grad_value"]
        self.accumulation_steps = config["accumulation_steps"]
        self.save_freq = config["save_freq"]

        self.jepa_loss = JEPASIGRegLoss(
            alpha=config.get("jepa_sigreg_alpha", 1.0),
            num_points=config.get("jepa_num_points", 17),
            num_slices=config.get("jepa_num_slices", 1024),
            detach_target=config.get("jepa_detach_target", True),
            reg_on=config.get("jepa_reg_on", "both"),
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
        )

    # def train_epoch(self) -> Tuple[bool, float]:
    #     self.model.train()
    #     avg_loss = 0.0
    #     terminate = False

    #     accumulation_steps = max(1, int(self.accumulation_steps))
    #     total_batches = len(self.train_loader)

    #     self.optimizer.zero_grad(set_to_none=True)

    #     pbar = tqdm(self.train_loader, desc="JEPA Pretraining", file=sys.stdout)

    #     for batch_idx, batch_data in enumerate(pbar):
    #         batch_data: BatchData
    #         batch = batch_data.to_device(self.device)

    #         if batch.input_mask is not None:
    #             batch.input_seq = torch.where(
    #                 batch.input_mask,
    #                 torch.zeros_like(batch.input_seq),
    #                 batch.input_seq,
    #             )

    #         if batch.target_mask is not None:
    #             batch.target_seq = torch.where(
    #                 batch.target_mask,
    #                 torch.zeros_like(batch.target_seq),
    #                 batch.target_seq,
    #             )
                
    #         # Normalize input and target in the same coordinate system.
    #         batch.input_seq = self.scalar.fit_transform(batch.input_seq)
    #         batch.target_seq = self.scalar.transform(batch.target_seq)

    #         # Determine current accumulation group size.
    #         group_start = (batch_idx // accumulation_steps) * accumulation_steps
    #         current_group_size = min(accumulation_steps, total_batches - group_start)

    #         Ey, Ey_pred = self.model(
    #             input_seq=batch.input_seq,
    #             input_features=batch.input_features,
    #             target_seq=batch.target_seq,
    #             target_features=batch.target_features,
    #             mode="pretrain",
    #         )

    #         # Raw loss for logging
    #         loss, loss_dict = self.jepa_loss(Ey_pred, Ey)

    #         if torch.isnan(loss):
    #             self.logger.error("JEPA pretraining loss is NaN. Terminating.")
    #             terminate = True
    #             break

    #         # Scale loss before backward.
    #         loss_for_backward = loss / current_group_size
    #         loss_for_backward.backward()

    #         avg_loss = avg_loss * (batch_idx / (batch_idx + 1)) + loss.item() / (
    #             batch_idx + 1
    #         )

    #         pbar.set_description(
    #             "JEPA Loss: "
    #             f"{loss.item():.4f}, "
    #             f"Align: {loss_dict['loss_align'].item():.4f}, "
    #             f"SIGReg: {loss_dict['loss_reg'].item():.4f}, "
    #             f"Avg: {avg_loss:.4f}, "
    #             f"Accum: {(batch_idx % accumulation_steps) + 1}/{current_group_size}"
    #         )

    #         should_update = (
    #             ((batch_idx + 1) % accumulation_steps == 0)
    #             or ((batch_idx + 1) == total_batches)
    #         )

    #         if should_update:
    #             if self.clip_grad_value > 0:
    #                 torch.nn.utils.clip_grad_norm_(
    #                     self.model.parameters(),
    #                     self.clip_grad_value,
    #                 )

    #             self.optimizer.step()
    #             self.optimizer.zero_grad(set_to_none=True)

    #     return terminate, avg_loss
    
    def train_epoch(self) -> Tuple[bool, float]:
        """
        Train one epoch for JEPA pretraining.

        This version is robust to:
        1. masked/null input values;
        2. masked/null target values;
        3. NaN/Inf hidden representations;
        4. NaN/Inf JEPA loss;
        5. gradient accumulation with valid micro-batches.
        """
        self.model.train()
        avg_loss = 0.0
        terminate = False

        accumulation_steps = max(1, int(self.accumulation_steps))
        valid_micro_steps = 0

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(self.train_loader, desc="JEPA Pretraining", file=sys.stdout)

        for batch_idx, batch_data in enumerate(pbar):
            batch_data: BatchData
            batch = batch_data.to_device(self.device)

            # ------------------------------------------------------------
            # 1. Fill masked input/target before normalization and encoder.
            # ------------------------------------------------------------
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

            # ------------------------------------------------------------
            # 2. Normalize input and target in the same coordinate system.
            # ------------------------------------------------------------
            batch.input_seq = self.scalar.fit_transform(batch.input_seq)
            batch.target_seq = self.scalar.transform(batch.target_seq)

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

            # ------------------------------------------------------------
            # 3. Forward in JEPA pretrain mode.
            # ------------------------------------------------------------
            Ey, Ey_pred = self.model(
                input_seq=batch.input_seq,
                input_features=batch.input_features,
                target_seq=batch.target_seq,
                target_features=batch.target_features,
                mode="pretrain",
            )

            # ------------------------------------------------------------
            # 4. Check hidden representations.
            # ------------------------------------------------------------
            if (
                torch.isnan(Ey).any()
                or torch.isinf(Ey).any()
                or torch.isnan(Ey_pred).any()
                or torch.isinf(Ey_pred).any()
            ):
                self.logger.error(
                    f"JEPA hidden representation contains NaN/Inf at batch {batch_idx}. "
                    f"Ey_nan={torch.isnan(Ey).sum().item()}, "
                    f"Ey_inf={torch.isinf(Ey).sum().item()}, "
                    f"Ey_pred_nan={torch.isnan(Ey_pred).sum().item()}, "
                    f"Ey_pred_inf={torch.isinf(Ey_pred).sum().item()}."
                )
                terminate = True
                break

            # ------------------------------------------------------------
            # 5. Compute JEPA loss.
            # ------------------------------------------------------------
            loss, loss_dict = self.jepa_loss(Ey_pred, Ey)

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(
                    f"JEPA pretraining loss is NaN/Inf at batch {batch_idx}. "
                    f"loss_align={loss_dict['loss_align'].item()}, "
                    f"loss_reg={loss_dict['loss_reg'].item()}."
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
                "JEPA Loss: "
                f"{loss.item():.4f}, "
                f"Align: {loss_dict['loss_align'].item():.4f}, "
                f"SIGReg: {loss_dict['loss_reg'].item():.4f}, "
                f"Avg: {avg_loss:.4f}, "
                f"Accum: {accum_display}/{accumulation_steps}"
            )

            should_update = valid_micro_steps % accumulation_steps == 0

            if should_update:
                if self.clip_grad_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
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
                    self.model.parameters(),
                    self.clip_grad_value,
                )

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        if valid_micro_steps == 0:
            self.logger.error("No valid JEPA pretraining batches in this epoch.")
            terminate = True

        return terminate, avg_loss
    
    def save_checkpoint(self, epoch: int):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        path = os.path.join(self.save_path, f"jepa_pretrain_epoch_{epoch}.pth")
        torch.save(checkpoint, path)

    def run(self):
        log_metadata = {
            "training_start_time": datetime.now().isoformat(),
            "mode": "jepa_pretrain",
            "max_epochs": self.max_epochs,
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "milestones": str(self.milestones),
            "gamma": self.gamma,
            "clip_grad_value": self.clip_grad_value,
            "accumulation_steps": self.accumulation_steps,
            "save_freq": self.save_freq,
            "save_dir": self.save_path,
            "epoch_logs": {},
        }
        self.initialize_log_file(log_metadata)
        self.logger.info("Starting JEPA pretraining...")

        for epoch in range(1, self.max_epochs + 1):
            start = time()
            terminate, train_loss = self.train_epoch()
            train_time = time() - start

            self.logger.info(
                f"[JEPA Pretrain] Epoch {epoch}/{self.max_epochs}, "
                f"loss={train_loss:.6f}, time={train_time:.2f}s"
            )

            if terminate:
                break

            if (epoch) % self.save_freq == 0:
                self.save_checkpoint(epoch)

            train_memory_reserved = torch.cuda.memory_reserved()
            train_memory_allocated = torch.cuda.memory_allocated()
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(f"Pretrain loss: {train_loss}")
            self.logger.info(f"Pretrain time: {train_time}s")
            self.logger.info(
                f"Pretrain memory reserved: {train_memory_reserved / (1024 ** 2):.2f} MB"
            )
            self.logger.info(
                f"Pretrain memory allocated: {train_memory_allocated / (1024 ** 2):.2f} MB"
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
            self._log_epoch_info(epoch=f"Pretrain {epoch}", epoch_log=pretrain_log)

            self.scheduler.step()

        self.logger.info("JEPA pretraining completed.")
