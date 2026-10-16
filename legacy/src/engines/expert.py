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

    # def train_epoch(self) -> Tuple[bool, float]:
    #     """
    #     Train for one epoch.

    #     Returns:
    #         Tuple of (terminate_flag, average_loss)
    #     """
    #     self.model.train()
    #     avg_loss = 0.0
    #     terminate = False

    #     accumulation_steps = max(1, int(self.accumulation_steps))
    #     total_batches = len(self.train_loader)
    #     self.optimizer.zero_grad(set_to_none=True)

    #     pbar = tqdm(self.train_loader, desc="Training", file=sys.stdout)
    #     for batch_idx, batch_data in enumerate(pbar):
    #         batch_data: BatchData
    #         batch = batch_data.to_device(self.device)
    #         batch.input_seq = self.scalar.fit_transform(batch.input_seq)

    #         # Determine current accumulation group size.
    #         # This handles the last group correctly when total_batches is not divisible
    #         # by accumulation_steps.
    #         group_start = (batch_idx // accumulation_steps) * accumulation_steps
    #         current_group_size = min(accumulation_steps, total_batches - group_start)

    #         # Forward
    #         model_input = batch.to_dict()
    #         pred_norm = self.model(**model_input)
    #         pred = self.scalar.inverse_transform(pred_norm)

    #         # Raw loss for logging
    #         loss: torch.Tensor = self.loss_func(
    #             batch.target_seq[~batch.target_mask],
    #             pred[~batch.target_mask],
    #         )

    #         if torch.isnan(loss):
    #             self.logger.error("Training loss is NaN. Terminating.")
    #             terminate = True
    #             break

    #         # Scale loss before backward so accumulated gradients approximate
    #         # the average gradient over current_group_size micro-batches.
    #         loss_for_backward = loss / current_group_size
    #         loss_for_backward.backward()

    #         # Logging uses the unscaled loss.
    #         avg_loss = avg_loss * (batch_idx / (batch_idx + 1)) + loss.item() / (
    #             batch_idx + 1
    #         )
    #         pbar.set_description(
    #             f"Loss: {loss.item():.4f}, "
    #             f"Average Loss: {avg_loss:.4f}, "
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
        Train for one epoch.

        This version is robust to:
        1. fully-masked target batches;
        2. NaN/Inf values in input, target, or prediction;
        3. small batch size, e.g., batch_size = 1 or 2;
        4. gradient accumulation with skipped invalid micro-batches.

        Returns:
            Tuple of (terminate_flag, average_loss)
        """
        self.model.train()
        avg_loss = 0.0
        terminate = False

        accumulation_steps = max(1, int(self.accumulation_steps))
        valid_micro_steps = 0

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(self.train_loader, desc="Training", file=sys.stdout)

        for batch_idx, batch_data in enumerate(pbar):
            batch_data: BatchData
            batch = batch_data.to_device(self.device)

            # ------------------------------------------------------------
            # 1. Fill masked values before normalization/model forward.
            #    This prevents NaN/null values from entering the model.
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

            # Extra safety: remove any remaining NaN/Inf values.
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
            # 2. Normalize input sequence.
            # ------------------------------------------------------------
            batch.input_seq = self.scalar.fit_transform(batch.input_seq)
            batch.input_seq = torch.nan_to_num(
                batch.input_seq,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # ------------------------------------------------------------
            # 3. Forward.
            # ------------------------------------------------------------
            model_input = batch.to_dict()
            pred_norm = self.model(**model_input)
            pred = self.scalar.inverse_transform(pred_norm)

            # ------------------------------------------------------------
            # 4. Build valid supervision mask.
            # ------------------------------------------------------------
            if batch.target_mask is None:
                supervision_mask = torch.ones_like(batch.target_seq, dtype=torch.bool)
            else:
                supervision_mask = ~batch.target_mask

            # Remove invalid target positions.
            supervision_mask = supervision_mask & torch.isfinite(batch.target_seq)

            valid_target_count = supervision_mask.sum().item()

            if valid_target_count == 0:
                self.logger.warning(
                    f"Skip batch {batch_idx}: no valid target values after masking."
                )
                continue

            # If prediction is NaN/Inf on valid target positions, this is a real model/numerical issue.
            pred_on_valid = pred[supervision_mask]
            target_on_valid = batch.target_seq[supervision_mask]

            if torch.isnan(pred_on_valid).any() or torch.isinf(pred_on_valid).any():
                self.logger.error(
                    f"Prediction contains NaN/Inf at batch {batch_idx}. "
                    f"valid_target_count={valid_target_count}, "
                    f"pred_nan={torch.isnan(pred_on_valid).sum().item()}, "
                    f"pred_inf={torch.isinf(pred_on_valid).sum().item()}."
                )
                terminate = True
                break

            if torch.isnan(target_on_valid).any() or torch.isinf(target_on_valid).any():
                self.logger.error(
                    f"Target contains NaN/Inf at batch {batch_idx}. "
                    f"valid_target_count={valid_target_count}, "
                    f"target_nan={torch.isnan(target_on_valid).sum().item()}, "
                    f"target_inf={torch.isinf(target_on_valid).sum().item()}."
                )
                terminate = True
                break

            # ------------------------------------------------------------
            # 5. Compute safe loss.
            # ------------------------------------------------------------
            loss: torch.Tensor = self.loss_func(
                target_on_valid,
                pred_on_valid,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(
                    f"Training loss is NaN/Inf at batch {batch_idx}. "
                    f"valid_target_count={valid_target_count}."
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
                f"Loss: {loss.item():.4f}, "
                f"Average Loss: {avg_loss:.4f}, "
                f"Valid: {valid_target_count}, "
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
            self.logger.error("No valid training batches in this epoch.")
            terminate = True

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

    # def _evaluate(self, loader, prefix: str = "Evaluation") -> Dict[str, float]:
    #     """
    #     Internal evaluation method.

    #     Args:
    #         loader: DataLoader for evaluation
    #         prefix: Progress bar prefix

    #     Returns:
    #         Dictionary of metrics
    #     """
    #     self.model.eval()
    #     all_metrics = None
    #     valid_iter = 0
    #     with torch.no_grad():
    #         pbar = tqdm(loader, desc=prefix, file=sys.stdout)
    #         for batch_data in pbar:
    #             # Prepare batch
    #             batch_data: BatchData

    #             if batch_data.input_mask.all() or batch_data.target_mask.all():
    #                 continue

    #             batch = batch_data.to_device(self.device)
    #             batch.input_seq = self.scalar.fit_transform(batch.input_seq)

    #             # Forward pass
    #             model_input = batch.to_dict()
    #             pred_norm = self.model(**model_input)
    #             pred = self.scalar.inverse_transform(pred_norm)

    #             # Compute metrics
    #             metrics = compute_all_metrics(
    #                 pred[~batch.target_mask],
    #                 batch.target_seq[~batch.target_mask],
    #             )

    #             if all_metrics is None:
    #                 all_metrics = metrics
    #             else:
    #                 for k, v in metrics.items():
    #                     if np.isnan(v):
    #                         self.logger.warning(f"Found NaN value in evaluation: {k}.")
    #                         continue
    #                     all_metrics[k] = all_metrics[k] * (
    #                         valid_iter / (valid_iter + 1)
    #                     ) + v / (valid_iter + 1)
    #             valid_iter += 1
    #             desc = "{prefix} {metric_name}: ({metric_values}), ".format(
    #                 prefix="Evaluation",
    #                 metric_name="-".join(metrics.keys()),
    #                 metric_values="-".join(
    #                     "{:.4f}".format(value) for value in metrics.values()
    #                 ),
    #             )
    #             desc += "Average {metric_name}: ({metric_values})".format(
    #                 metric_name="-".join(all_metrics.keys()),
    #                 metric_values="-".join(
    #                     "{:.4f}".format(value) for value in all_metrics.values()
    #                 ),
    #             )
    #             pbar.set_description(desc)

    #     return all_metrics
    
    def _evaluate(self, loader, prefix: str = "Evaluation") -> Dict[str, float]:
        """
        Internal evaluation method.

        This version is robust to:
        1. fully-masked target batches;
        2. NaN/Inf in input, target, or prediction;
        3. empty metric tensors after masking.

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

            for batch_idx, batch_data in enumerate(pbar):
                batch_data: BatchData

                batch = batch_data.to_device(self.device)

                # ------------------------------------------------------------
                # 1. Fill masked values before normalization/model forward.
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
                # 2. Build valid target mask.
                # ------------------------------------------------------------
                if batch.target_mask is None:
                    valid_mask = torch.ones_like(batch.target_seq, dtype=torch.bool)
                else:
                    valid_mask = ~batch.target_mask

                valid_mask = valid_mask & torch.isfinite(batch.target_seq)

                if valid_mask.sum().item() == 0:
                    self.logger.warning(
                        f"Skip {prefix} batch {batch_idx}: no valid target values."
                    )
                    continue

                # ------------------------------------------------------------
                # 3. Forward.
                # ------------------------------------------------------------
                batch.input_seq = self.scalar.fit_transform(batch.input_seq)
                batch.input_seq = torch.nan_to_num(
                    batch.input_seq,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                model_input = batch.to_dict()
                pred_norm = self.model(**model_input)
                pred = self.scalar.inverse_transform(pred_norm)

                valid_mask = valid_mask & torch.isfinite(pred)

                if valid_mask.sum().item() == 0:
                    self.logger.warning(
                        f"Skip {prefix} batch {batch_idx}: no finite prediction on valid targets."
                    )
                    continue

                # ------------------------------------------------------------
                # 4. Compute metrics.
                # ------------------------------------------------------------
                metrics = compute_all_metrics(
                    pred[valid_mask],
                    batch.target_seq[valid_mask],
                )

                if all_metrics is None:
                    all_metrics = metrics
                else:
                    for k, v in metrics.items():
                        if np.isnan(v) or np.isinf(v):
                            self.logger.warning(
                                f"Found NaN/Inf value in evaluation metric: {k}."
                            )
                            continue

                        all_metrics[k] = all_metrics[k] * (
                            valid_iter / (valid_iter + 1)
                        ) + v / (valid_iter + 1)

                valid_iter += 1

                desc = "{prefix} {metric_name}: ({metric_values}), ".format(
                    prefix=prefix,
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

        if all_metrics is None:
            self.logger.error(f"No valid batches found during {prefix}.")
            return {
                "MAE": float("inf"),
                "RMSE": float("inf"),
            }

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

        for epoch in range(1, self.max_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{self.max_epochs}")

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
            if (epoch) % self.save_freq == 0 or is_best:
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
        test_log = {
            "metrics": test_metrics,
            "time_seconds": float(test_time),
            "memory_reserved_mb": float(test_memory_reserved / (1024**2)),
            "memory_allocated_mb": float(test_memory_allocated / (1024**2)),
        }
        self._log_epoch_info(epoch="test", epoch_log=test_log)
