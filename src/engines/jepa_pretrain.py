# import os
# import sys
# from datetime import datetime
# from time import time
# from typing import Dict, Any, Tuple
# from logging import Logger

import copy
import math
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
        
        # ------------------------------------------------------------
        # EMA target encoder configuration
        # ------------------------------------------------------------
        self.ema_momentum_start = float(
            config.get("jepa_ema_momentum_start", 0.996)
        )
        self.ema_momentum_end = float(
            config.get("jepa_ema_momentum_end", 1.0)
        )

        if not 0.0 <= self.ema_momentum_start <= 1.0:
            raise ValueError(
                "jepa_ema_momentum_start must be in [0, 1], "
                f"got {self.ema_momentum_start}"
            )

        if not 0.0 <= self.ema_momentum_end <= 1.0:
            raise ValueError(
                "jepa_ema_momentum_end must be in [0, 1], "
                f"got {self.ema_momentum_end}"
            )

        if self.ema_momentum_start > self.ema_momentum_end:
            raise ValueError(
                "jepa_ema_momentum_start must not be greater than "
                "jepa_ema_momentum_end."
            )

        accumulation_steps = max(1, int(self.accumulation_steps))

        optimizer_steps_per_epoch = math.ceil(
            len(self.train_loader) / accumulation_steps
        )

        self.total_ema_steps = max(
            1,
            self.max_epochs * optimizer_steps_per_epoch,
        )

        self.ema_step = 0
        
        # ------------------------------------------------------------
        # Create EMA target encoder
        # ------------------------------------------------------------
        #
        # The complete model is copied for a generic implementation,
        # but only target_model.encode(...) is used during pretraining.
        self.target_model = copy.deepcopy(self.model).to(self.device)

        # The target network must never receive gradients.
        self.target_model.requires_grad_(False)

        # Disable dropout and use stable normalization behavior.
        self.target_model.eval()

        # self.jepa_loss = JEPASIGRegLoss(
        #     alpha=config.get("jepa_sigreg_alpha", 1.0),
        #     num_points=config.get("jepa_num_points", 17),
        #     num_slices=config.get("jepa_num_slices", 1024),
        #     detach_target=config.get("jepa_detach_target", True),
        #     reg_on=config.get("jepa_reg_on", "both"),
        # ).to(self.device)
        
# PatchTST declares jepa_feature_dim=2.
# Other models default to the last dimension.
        jepa_feature_dim = getattr(
            self.model,
            "jepa_feature_dim",
            -1,
        )

        self.jepa_loss = JEPASIGRegLoss(
            alpha=config.get("jepa_sigreg_alpha", 1.0),
            num_points=config.get("jepa_num_points", 17),
            num_slices=config.get("jepa_num_slices", 1024),
            detach_target=config.get("jepa_detach_target", True),
            reg_on=config.get("jepa_reg_on", "pred"),
            feature_dim=jepa_feature_dim,
        ).to(self.device)

        # self.optimizer = optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        # )
        
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        if len(trainable_parameters) == 0:
            raise RuntimeError("Online JEPA model has no trainable parameters.")

        self.optimizer = optim.AdamW(
            trainable_parameters,
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
    
    def _get_ema_momentum(self) -> float:
        """
        Cosine schedule from ema_momentum_start to ema_momentum_end.

        At the beginning:
            momentum = ema_momentum_start

        Near the end:
            momentum -> ema_momentum_end
        """
        if self.total_ema_steps <= 1:
            return self.ema_momentum_start

        progress = min(
            self.ema_step / (self.total_ema_steps - 1),
            1.0,
        )

        momentum = (
            self.ema_momentum_end
            - (
                self.ema_momentum_end
                - self.ema_momentum_start
            )
            * (math.cos(math.pi * progress) + 1.0)
            / 2.0
        )

        return float(momentum)


    @torch.no_grad()
    def _update_target_encoder(self) -> float:
        """
        Update the target encoder with exponential moving average:

            theta_target =
                momentum * theta_target
                + (1 - momentum) * theta_online

        Model buffers, such as BatchNorm running statistics, are copied
        directly from the online model.
        """
        momentum = self._get_ema_momentum()

        online_parameters = dict(self.model.named_parameters())
        target_parameters = dict(self.target_model.named_parameters())

        if online_parameters.keys() != target_parameters.keys():
            raise RuntimeError(
                "Online model and target model parameter structures do not match."
            )

        for name, target_parameter in target_parameters.items():
            online_parameter = online_parameters[name]

            target_parameter.mul_(momentum).add_(
                online_parameter,
                alpha=1.0 - momentum,
            )

        # Copy BatchNorm statistics and other registered buffers.
        online_buffers = dict(self.model.named_buffers())
        target_buffers = dict(self.target_model.named_buffers())

        if online_buffers.keys() != target_buffers.keys():
            raise RuntimeError(
                "Online model and target model buffer structures do not match."
            )

        for name, target_buffer in target_buffers.items():
            target_buffer.copy_(online_buffers[name])

        self.ema_step += 1

        return momentum


    def _forward_jepa(self, batch: BatchData):
        """
        JEPA forward with separate online and EMA target encoders.

        Online path:
            input -> online encoder -> Ex -> predictor -> Ey_pred

        Target path:
            target -> EMA target encoder -> Ey

        Only the online path receives gradients.
        """
        required_methods = ("encode", "predict")

        for method_name in required_methods:
            if not hasattr(self.model, method_name):
                raise AttributeError(
                    f"Online model {type(self.model).__name__} "
                    f"does not implement {method_name}()."
                )

            if not hasattr(self.target_model, method_name):
                raise AttributeError(
                    f"Target model {type(self.target_model).__name__} "
                    f"does not implement {method_name}()."
                )

        # Online encoder: receives gradients.
        Ex = self.model.encode(
            batch.input_seq,
            batch.input_features,
        )

        Ey_pred = self.model.predict(Ex)

        # EMA target encoder: no gradients.
        with torch.no_grad():
            Ey = self.target_model.encode(
                batch.target_seq,
                batch.target_features,
            )

        if Ey_pred.shape != Ey.shape:
            raise RuntimeError(
                "JEPA online/target embedding shape mismatch: "
                f"Ey_pred.shape={tuple(Ey_pred.shape)}, "
                f"Ey.shape={tuple(Ey.shape)}"
            )

        return Ey, Ey_pred
    
    def train_epoch(self) -> Tuple[bool, float]:
        """
        Train one JEPA pretraining epoch.

        Online network:
            receives gradients and is updated by AdamW.

        Target network:
            receives no gradients and is updated by EMA after every
            optimizer step.
        """
        self.model.train()
        self.target_model.eval()

        avg_loss = 0.0
        terminate = False

        accumulation_steps = max(
            1,
            int(self.accumulation_steps),
        )

        total_batches = len(self.train_loader)

        if total_batches == 0:
            self.logger.error("JEPA train_loader contains no batches.")
            return True, 0.0

        valid_micro_steps = 0
        last_ema_momentum = self._get_ema_momentum()

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            self.train_loader,
            desc="JEPA Pretraining",
            file=sys.stdout,
        )

        for batch_idx, batch_data in enumerate(pbar):
            batch_data: BatchData
            batch = batch_data.to_device(self.device)

            # ------------------------------------------------------------
            # 1. Replace masked values before normalization.
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
            batch.input_seq = self.scalar.fit_transform(
                batch.input_seq
            )

            batch.target_seq = self.scalar.transform(
                batch.target_seq
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
            # 3. Online encoder + EMA target encoder.
            # ------------------------------------------------------------
            Ey, Ey_pred = self._forward_jepa(batch)

            # ------------------------------------------------------------
            # 4. Validate hidden representations.
            # ------------------------------------------------------------
            has_invalid_embedding = (
                torch.isnan(Ey).any()
                or torch.isinf(Ey).any()
                or torch.isnan(Ey_pred).any()
                or torch.isinf(Ey_pred).any()
            )

            if has_invalid_embedding:
                self.logger.error(
                    "JEPA hidden representation contains NaN/Inf at "
                    f"batch {batch_idx}. "
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
            loss, loss_dict = self.jepa_loss(
                Ey_pred,
                Ey,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(
                    f"JEPA loss is NaN/Inf at batch {batch_idx}. "
                    f"loss_align={loss_dict['loss_align'].item()}, "
                    f"loss_reg={loss_dict['loss_reg'].item()}."
                )

                terminate = True
                break

            # ------------------------------------------------------------
            # 6. Correct gradient accumulation scaling.
            #
            # Example:
            #     total batches = 19
            #     accumulation_steps = 8
            #
            # Groups:
            #     8, 8, 3
            #
            # The last group divides by 3 rather than by 8.
            # ------------------------------------------------------------
            group_start = (
                batch_idx // accumulation_steps
            ) * accumulation_steps

            current_group_size = min(
                accumulation_steps,
                total_batches - group_start,
            )

            loss_for_backward = loss / current_group_size
            loss_for_backward.backward()

            valid_micro_steps += 1

            avg_loss = (
                avg_loss
                * ((valid_micro_steps - 1) / valid_micro_steps)
                + loss.item() / valid_micro_steps
            )

            micro_step_in_group = (
                batch_idx - group_start + 1
            )

            # Update at the end of a complete or final partial group.
            should_update = (
                micro_step_in_group == current_group_size
            )

            if should_update:
                if self.clip_grad_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad_value,
                    )

                # Update online network.
                self.optimizer.step()

                # Update EMA target network after online parameters change.
                last_ema_momentum = (
                    self._update_target_encoder()
                )

                self.optimizer.zero_grad(set_to_none=True)

            pbar.set_description(
                "JEPA Loss: "
                f"{loss.item():.4f}, "
                f"Align: {loss_dict['loss_align'].item():.4f}, "
                f"SIGReg: {loss_dict['loss_reg'].item():.4f}, "
                f"Avg: {avg_loss:.4f}, "
                f"Accum: {micro_step_in_group}/{current_group_size}, "
                f"EMA: {last_ema_momentum:.6f}"
            )

        if valid_micro_steps == 0:
            self.logger.error(
                "No valid JEPA pretraining batches in this epoch."
            )
            terminate = True

        return terminate, avg_loss
    
    def save_checkpoint(self, epoch: int):
        checkpoint = {
            "epoch": epoch,

            # Online encoder + predictor.
            # Finetuning continues to load this field.
            "model_state_dict": self.model.state_dict(),

            # EMA target encoder for resuming JEPA pretraining.
            "target_model_state_dict": (
                self.target_model.state_dict()
            ),

            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),

            "scheduler_state_dict": (
                self.scheduler.state_dict()
            ),

            "ema_step": self.ema_step,
            "ema_momentum_start": self.ema_momentum_start,
            "ema_momentum_end": self.ema_momentum_end,
            "total_ema_steps": self.total_ema_steps,
        }

        path = os.path.join(
            self.save_path,
            f"jepa_pretrain_epoch_{epoch}.pth",
        )

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
