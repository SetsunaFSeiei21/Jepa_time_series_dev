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

    def train_epoch(self) -> Tuple[bool, float]:
        self.model.train()
        avg_loss = 0.0
        terminate = False
        accumulation_counter = 0

        pbar = tqdm(self.train_loader, desc="JEPA Pretraining", file=sys.stdout)

        for batch_idx, batch_data in enumerate(pbar):
            batch_data: BatchData
            batch = batch_data.to_device(self.device)

            # 和原 ExpertEngine 保持一致：只 normalize input_seq。
            # 但 JEPA target_seq 也进入 encoder，所以这里 target_seq 也建议 normalize。
            batch.input_seq = self.scalar.fit_transform(batch.input_seq)
            batch.target_seq = self.scalar.transform(batch.target_seq)

            if accumulation_counter == 0:
                self.optimizer.zero_grad()

            Ey, Ey_pred = self.model(
                input_seq=batch.input_seq,
                input_features=batch.input_features,
                target_seq=batch.target_seq,
                target_features=batch.target_features,
                mode="pretrain",
            )

            loss, loss_dict = self.jepa_loss(Ey_pred, Ey)

            if torch.isnan(loss):
                self.logger.error("JEPA pretraining loss is NaN. Terminating.")
                terminate = True
                break

            loss.backward()

            avg_loss = avg_loss * (batch_idx / (batch_idx + 1)) + loss.item() / (
                batch_idx + 1
            )

            pbar.set_description(
                "JEPA Loss: "
                f"{loss.item():.4f}, "
                f"Align: {loss_dict['loss_align'].item():.4f}, "
                f"SIGReg: {loss_dict['loss_reg'].item():.4f}, "
                f"Avg: {avg_loss:.4f}"
            )

            accumulation_counter += 1

            if accumulation_counter % self.accumulation_steps == 0:
                if self.clip_grad_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad_value,
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulation_counter = 0

        if accumulation_counter > 0:
            if self.clip_grad_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_value,
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

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
