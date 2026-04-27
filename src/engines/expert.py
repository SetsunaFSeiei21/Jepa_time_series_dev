import json
import os
import sys
from datetime import datetime
from logging import Logger
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch import optim
from tqdm import tqdm

from src.base.data import BatchData
from src.base.engine import BaseEngine
from src.utils.metrics import compute_all_metrics
from src.utils.scalar import NormalizationPipeline, ZScoreNormalizer


class ExpertEngine(BaseEngine):

    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        logger: Logger,
        save_path: str,
        scalar: NormalizationPipeline,
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
        self.loss_func = loss_func

        # Training configuration
        self.max_epochs = config["max_epochs"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.milestones = config["milestones"]
        self.gamma = config["gamma"]
        self.clip_grad_value = config["clip_grad_value"]
        self.accumulation_steps = config["accumulation_steps"]
        self.save_freq = config["save_freq"]

        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.milestones, gamma=self.gamma
        )
        self.scalar = scalar

        # Track best model
        self.best_criteria = float("inf")
        self.best_epoch = 0

    def train_epoch(self) -> Tuple[bool, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (terminate_flag, average_loss)
        """
        self.model.train()
        avg_loss = 0.0
        terminate = False

        accumulation_counter = 0

        pbar = tqdm(self.train_loader, desc="Training", file=sys.stdout)
        for batch_idx, batch_data in enumerate(pbar):
            # Prepare batch
            batch_data: BatchData
            batch = batch_data.to_device(self.device)
            batch.input_seq = self.scalar.fit_transform(batch.input_seq)

            # Zero gradients
            if accumulation_counter == 0:
                self.optimizer.zero_grad()

            # Forward pass
            # Model should handle normalization internally
            model_input = batch.to_dict()
            pred_norm = self.model(**model_input)
            pred = self.scalar.inverse_transform(pred_norm)

            # Compute loss
            loss: torch.Tensor = self.loss_func(
                batch.target_seq[~batch.target_mask], pred[~batch.target_mask]
            )

            # Check for NaN
            if torch.isnan(loss):
                self.logger.error("Training loss is NaN. Terminating.")
                terminate = True
                break

            # Backward pass with gradient accumulation
            loss.backward()

            # Update progress bar
            avg_loss = avg_loss * (batch_idx / (batch_idx + 1)) + loss.item() / (
                batch_idx + 1
            )
            pbar.set_description(
                f"Loss: {loss.item():.4f}, Average Loss: {avg_loss:.4f}"
            )

            # Gradient accumulation step
            accumulation_counter += 1
            if accumulation_counter % self.accumulation_steps == 0:
                if self.clip_grad_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_value
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulation_counter = 0

        # Handle remaining accumulated gradients
        if accumulation_counter > 0:
            if self.clip_grad_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_value
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        return terminate, avg_loss

    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.

        Returns:
            Dictionary of validation metrics
        """
        return self._evaluate(self.val_loader, prefix="Validation")

    def test(self) -> Dict[str, float]:
        """
        Test model on test set.

        Returns:
            Dictionary of test metrics
        """
        return self._evaluate(self.test_loader, prefix="Test")

    def _evaluate(self, loader, prefix: str = "Evaluation") -> Dict[str, float]:
        """
        Internal evaluation method.

        Args:
            loader: DataLoader for evaluation
            prefix: Progress bar prefix

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_metrics = None
        valid_iter = 0
        with torch.no_grad():
            pbar = tqdm(loader, desc=prefix, file=sys.stdout)
            for batch_data in pbar:
                # Prepare batch
                batch_data: BatchData

                if batch_data.input_mask.all() or batch_data.target_mask.all():
                    continue

                batch = batch_data.to_device(self.device)
                batch.input_seq = self.scalar.fit_transform(batch.input_seq)

                # Forward pass
                model_input = batch.to_dict()
                pred_norm = self.model(**model_input)
                pred = self.scalar.inverse_transform(pred_norm)

                # Compute metrics
                metrics = compute_all_metrics(
                    pred[~batch.target_mask],
                    batch.target_seq[~batch.target_mask],
                )

                if all_metrics is None:
                    all_metrics = metrics
                else:
                    for k, v in metrics.items():
                        if np.isnan(v):
                            self.logger.warning(f"Found NaN value in evaluation: {k}.")
                            continue
                        all_metrics[k] = all_metrics[k] * (
                            valid_iter / (valid_iter + 1)
                        ) + v / (valid_iter + 1)
                valid_iter += 1
                desc = "{prefix} {metric_name}: ({metric_values}), ".format(
                    prefix="Evaluation",
                    metric_name="-".join(metrics.keys()),
                    metric_values="-".join(
                        "{:.4f}".format(value) for value in metrics.values()
                    ),
                )
                desc += "Average {metric_name}: ({metric_values})".format(
                    metric_name="-".join(all_metrics.keys()),
                    metric_values="-".join(
                        "{:.4f}".format(value) for value in all_metrics.values()
                    ),
                )
                pbar.set_description(desc)

        return all_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_criteria,
            "config": {
                "max_epochs": self.max_epochs,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
            },
        }

        # Regular checkpoint
        checkpoint_path = os.path.join(self.save_path, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)

        # Best model checkpoint
        if is_best:
            best_path = os.path.join(self.save_path, "best_model_checkpoint.pth")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint: Dict = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "best_val_loss" in checkpoint:
            self.best_criteria = checkpoint["best_val_loss"]
        else:
            self.logger.warning(
                "best_val_loss not found in checkpoint. Setting to infinity."
            )
            self.best_criteria = float("inf")
        if "epoch" in checkpoint:
            self.best_epoch = checkpoint["epoch"]
        else:
            self.logger.warning(
                "epoch not found in checkpoint. Setting best_epoch to 0."
            )
            self.best_epoch = 0

        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def run(self):
        """
        Main training loop.
        """
        log_metadata = {
            "training_start_time": datetime.now().isoformat(),
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

        TERMINATE = False
        self.logger.info("Starting training...")

        for epoch in range(self.max_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.max_epochs}")

            # Training
            train_start = time()
            TERMINATE, train_loss = self.train_epoch()
            train_time = time() - train_start
            train_memory_reserved = torch.cuda.memory_reserved()
            train_memory_allocated = torch.cuda.memory_allocated()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Train loss: {train_loss}")
            self.logger.info(f"Train time: {train_time}s")
            self.logger.info(
                f"Train memory reserved: {train_memory_reserved / (1024 ** 2):.2f} MB"
            )
            self.logger.info(
                f"Train memory allocated: {train_memory_allocated / (1024 ** 2):.2f} MB"
            )
            self.logger.info(f"Current learning rate: {current_lr}")

            if TERMINATE:
                self.logger.error("Terminating training due to NaN loss.")
                break

            # Validation
            val_start = time()
            val_metrics = self.validate()
            val_time = time() - val_start
            val_memory_reserved = torch.cuda.memory_reserved()
            val_memory_allocated = torch.cuda.memory_allocated()
            self.logger.info(f"Validation metrics: {val_metrics}")
            self.logger.info(f"Val time: {val_time:.2f}s")
            self.logger.info(
                f"Val memory reserved: {val_memory_reserved / (1024 ** 2):.2f} MB"
            )
            self.logger.info(
                f"Val memory allocated: {val_memory_allocated / (1024 ** 2):.2f} MB"
            )

            # Prepare epoch log entry
            epoch_log = {
                "timestamp": datetime.now().isoformat(),
                "current_learning_rate": current_lr,
                "training": {
                    "loss": float(train_loss),
                    "time_seconds": float(train_time),
                    "memory_reserved_mb": float(train_memory_reserved / (1024**2)),
                    "memory_allocated_mb": float(train_memory_allocated / (1024**2)),
                },
                "validation": {
                    "metrics": val_metrics,
                    "time_seconds": float(val_time),
                    "memory_reserved_mb": float(val_memory_reserved / (1024**2)),
                    "memory_allocated_mb": float(val_memory_allocated / (1024**2)),
                },
            }
            self._log_epoch_info(epoch=epoch, epoch_log=epoch_log)

            # Check for best model
            criteria = val_metrics["MAE"]
            is_best = criteria <= self.best_criteria

            if is_best:
                self.best_criteria = criteria
                self.best_epoch = epoch
                self.logger.info(f"New best model! Val loss: {criteria:.6f}")

            # Save checkpoint
            if (epoch + 1) % self.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            test_start = time()
            test_metrics = self.test()
            test_time = time() - test_start
            test_memory_reserved = torch.cuda.memory_reserved()
            test_memory_allocated = torch.cuda.memory_allocated()
            self.logger.info(f"Test metrics: {test_metrics}")
            self.logger.info(f"Test time: {test_time:.2f}s")
            self.logger.info(
                f"Test memory reserved: {test_memory_reserved / (1024 ** 2):.2f} MB"
            )
            self.logger.info(
                f"Test memory allocated: {test_memory_allocated / (1024 ** 2):.2f} MB"
            )

            # Update scheduler
            self.scheduler.step()

        self.logger.info(
            f"Training completed. Best epoch: {self.best_epoch}, Best val loss: {self.best_criteria:.6f}"
        )

        self.load_checkpoint(os.path.join(self.save_path, "best_model_checkpoint.pth"))
        self.logger.info("Evaluating best model on test sets...")
        test_start = time()
        test_metrics = self.test()
        test_time = time() - test_start
        test_memory_reserved = torch.cuda.memory_reserved()
        test_memory_allocated = torch.cuda.memory_allocated()
        self.logger.info(f"Test metrics: {test_metrics}")
        self.logger.info(f"Test time: {test_time:.2f}s")
        self.logger.info(
            f"Test memory reserved: {test_memory_reserved / (1024 ** 2):.2f} MB"
        )
        self.logger.info(
            f"Test memory allocated: {test_memory_allocated / (1024 ** 2):.2f} MB"
        )
