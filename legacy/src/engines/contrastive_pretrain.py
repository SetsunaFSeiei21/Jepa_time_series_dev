import os
import sys
from datetime import datetime
from time import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch import optim
from tqdm import tqdm

from src.base.data import BatchData
from src.base.engine import BaseEngine
from src.utils.contrastive import (
    FrequencyMasking,
    LinearProjectionHead,
    SymmetricInfoNCELoss,
    canonicalize_hidden_for_contrastive,
    temporal_mean_pool_and_flatten,
)


class ContrastivePretrainEngine(BaseEngine):
    """
    基于频域增强和 InfoNCE 的时间序列对比预训练 Engine。

    训练流程：

        X
        ├── FrequencyMasking 1 -> X1 -> Encoder -> Z1
        └── FrequencyMasking 2 -> X2 -> Encoder -> Z2

        Z1/Z2
            -> 统一为 (B,T,N,D)
            -> Linear projection head
            -> 时间维 mean pooling
            -> reshape 为 (B*N,D)
            -> symmetric InfoNCE

    projection head 只用于预训练，不会加载到下游微调模型。
    """

    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        logger,
        save_path: str,
        scalar,
        train_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        model_name: str,
    ):
        super().__init__(
            device=device,
            model=model,
            logger=logger,
            save_path=save_path,
        )

        self.train_loader = train_loader
        self.scalar = scalar
        self.model_name = model_name

        # ============================================================
        # 通用训练配置
        # ============================================================
        self.max_epochs = int(config["max_epochs"])
        self.lr = float(config["lr"])
        self.weight_decay = float(config["weight_decay"])
        self.milestones = list(config["milestones"])
        self.gamma = float(config["gamma"])
        self.clip_grad_value = float(config["clip_grad_value"])
        self.accumulation_steps = max(
            1,
            int(config["accumulation_steps"]),
        )
        self.save_freq = int(config["save_freq"])

        # ============================================================
        # 模型表示信息
        # ============================================================
        if not hasattr(self.model, "node_num"):
            raise AttributeError(
                f"{self.model_name} does not have attribute node_num."
            )

        if not hasattr(self.model, "d_model"):
            raise AttributeError(
                f"{self.model_name} does not have attribute d_model. "
                f"Contrastive pretraining needs model.d_model."
            )

        self.node_num = int(self.model.node_num)
        self.hidden_dim = int(self.model.d_model)

        # ============================================================
        # 对比学习组件
        # ============================================================
        self.augmentation = FrequencyMasking(
            mask_ratio=float(
                config.get(
                    "contrastive_freq_mask_ratio",
                    0.2,
                )
            ),
            preserve_dc=bool(
                config.get(
                    "contrastive_preserve_dc",
                    True,
                )
            ),
        )

        self.projection_head = LinearProjectionHead(
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.contrastive_loss = SymmetricInfoNCELoss(
            temperature=float(
                config.get(
                    "contrastive_temperature",
                    0.2,
                )
            )
        ).to(self.device)

        # Backbone 与 projection head 一起优化。
        self.trainable_parameters = (
            list(self.model.parameters())
            + list(self.projection_head.parameters())
        )

        self.optimizer = optim.AdamW(
            self.trainable_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
        )

    def _prepare_input(
        self,
        batch: BatchData,
    ) -> torch.Tensor:
        """
        清理并归一化输入。

        返回：
            x: (B,T,N,1)
        """

        x = batch.input_seq

        # 将数据集中原本无效的位置置零。
        if batch.input_mask is not None:
            x = torch.where(
                batch.input_mask,
                torch.zeros_like(x),
                x,
            )

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # 只对原始输入拟合一次归一化参数。
        #
        # 两个增强 view 必须位于同一个归一化坐标系，
        # 因此不能分别对 x1、x2 调用 fit_transform。
        x = self.scalar.fit_transform(x)

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return x

    def _encode_project_pool(
        self,
        view: torch.Tensor,
        input_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        编码一个增强 view。

        输入：
            view:
                (B,T,N,1)

            input_features:
                (B,T,F)

        输出：
            representation:
                (B*N,D)
        """

        # 调用现有 backbone 的 encoder。
        hidden = self.model.encode(
            view,
            input_features,
        )

        # 不同模型统一为 (B,Tz,Nz,D)。
        hidden = canonicalize_hidden_for_contrastive(
            hidden=hidden,
            model_name=self.model_name,
            node_num=self.node_num,
            hidden_dim=self.hidden_dim,
        )

        # Linear projection：
        # (B,Tz,Nz,D) -> (B,Tz,Nz,D)
        projected = self.projection_head(hidden)

        # 时间维 mean pooling，然后展开节点：
        # (B,Tz,Nz,D) -> (B,Nz,D) -> (B*Nz,D)
        representation = temporal_mean_pool_and_flatten(
            projected
        )

        return representation

    def train_epoch(
        self,
    ) -> Tuple[bool, float]:
        """
        执行一个 epoch 的对比预训练。

        返回：
            terminate:
                是否因为 NaN/Inf 等问题终止训练。

            avg_loss:
                当前 epoch 的平均 InfoNCE loss。
        """

        self.model.train()
        self.projection_head.train()

        terminate = False
        avg_loss = 0.0
        valid_micro_steps = 0

        self.optimizer.zero_grad(
            set_to_none=True
        )

        pbar = tqdm(
            self.train_loader,
            desc="Contrastive Pretraining",
            file=sys.stdout,
        )

        for batch_idx, batch_data in enumerate(pbar):
            batch_data: BatchData

            batch = batch_data.to_device(
                self.device
            )

            # ========================================================
            # 1. 清理并归一化输入
            # ========================================================
            x = self._prepare_input(batch)

            # ========================================================
            # 2. 两次独立频域增强
            # ========================================================
            x1 = self.augmentation(x)
            x2 = self.augmentation(x)

            # ========================================================
            # 3. Encoder + projection + pooling
            # ========================================================
            p1 = self._encode_project_pool(
                view=x1,
                input_features=batch.input_features,
            )

            p2 = self._encode_project_pool(
                view=x2,
                input_features=batch.input_features,
            )

            if p1.shape != p2.shape:
                raise RuntimeError(
                    f"Contrastive representation shape mismatch "
                    f"at batch {batch_idx}: "
                    f"p1={tuple(p1.shape)}, "
                    f"p2={tuple(p2.shape)}."
                )

            # Autoformer/FEDformer 的 N=1。
            # 如果最后一个 batch 恰好只有一个样本，则无法构造负样本。
            if p1.size(0) < 2:
                self.logger.warning(
                    f"Skip contrastive batch {batch_idx}: "
                    f"only {p1.size(0)} pooled sample is available. "
                    f"InfoNCE requires at least two samples."
                )
                continue

            # ========================================================
            # 4. InfoNCE
            # ========================================================
            loss = self.contrastive_loss(
                p1,
                p2,
            )

            if not torch.isfinite(loss):
                self.logger.error(
                    f"Contrastive loss contains NaN/Inf "
                    f"at batch {batch_idx}."
                )
                terminate = True
                break

            # ========================================================
            # 5. 梯度累积
            # ========================================================
            loss_for_backward = (
                loss / self.accumulation_steps
            )

            loss_for_backward.backward()

            valid_micro_steps += 1

            avg_loss += (
                loss.item() - avg_loss
            ) / valid_micro_steps

            current_accumulation = (
                valid_micro_steps
                % self.accumulation_steps
            )

            if current_accumulation == 0:
                current_accumulation = (
                    self.accumulation_steps
                )

            # 每个 anchor 的负样本数为 M-1。
            negative_num = p1.size(0) - 1

            pbar.set_description(
                f"Contrastive Loss: {loss.item():.4f}, "
                f"Average: {avg_loss:.4f}, "
                f"Samples: {p1.size(0)}, "
                f"Negatives/Anchor: {negative_num}, "
                f"Accum: "
                f"{current_accumulation}/"
                f"{self.accumulation_steps}"
            )

            should_update = (
                valid_micro_steps
                % self.accumulation_steps
                == 0
            )

            if should_update:
                if self.clip_grad_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_parameters,
                        self.clip_grad_value,
                    )

                self.optimizer.step()

                self.optimizer.zero_grad(
                    set_to_none=True
                )

        # ============================================================
        # 6. 处理不足 accumulation_steps 的最后一组梯度
        # ============================================================
        if (
            not terminate
            and valid_micro_steps > 0
            and valid_micro_steps
            % self.accumulation_steps
            != 0
        ):
            if self.clip_grad_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.trainable_parameters,
                    self.clip_grad_value,
                )

            self.optimizer.step()

            self.optimizer.zero_grad(
                set_to_none=True
            )

        if valid_micro_steps == 0:
            self.logger.error(
                "No valid contrastive batches "
                "were found in this epoch. "
                "Check the real batch_size."
            )
            terminate = True

        return terminate, avg_loss

    def save_checkpoint(
        self,
        epoch: int,
    ):
        """
        保存对比预训练 checkpoint。

        下游 finetune.py 只加载 model_state_dict，
        因此 projection_head 会被自动丢弃。
        """

        checkpoint = {
            "epoch": epoch,
            "pretrain_method": (
                "contrastive_fft_infonce"
            ),

            # 下游真正需要加载的 backbone。
            "model_state_dict": (
                self.model.state_dict()
            ),

            # 只为断点恢复或调试保存。
            # finetune.py 不会读取这个字段。
            "projection_head_state_dict": (
                self.projection_head.state_dict()
            ),

            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),

            "scheduler_state_dict": (
                self.scheduler.state_dict()
            ),

            "contrastive_config": {
                "temperature": (
                    self.contrastive_loss.temperature
                ),
                "freq_mask_ratio": (
                    self.augmentation.mask_ratio
                ),
                "preserve_dc": (
                    self.augmentation.preserve_dc
                ),
            },
        }

        checkpoint_path = os.path.join(
            self.save_path,
            (
                f"contrastive_pretrain_"
                f"epoch_{epoch}.pth"
            ),
        )

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        self.logger.info(
            f"Saved contrastive checkpoint: "
            f"{checkpoint_path}"
        )

    def run(self):
        """
        对比预训练主循环。
        """

        log_metadata = {
            "training_start_time": (
                datetime.now().isoformat()
            ),
            "mode": "contrastive_pretrain",
            "model_name": self.model_name,
            "max_epochs": self.max_epochs,
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "milestones": str(self.milestones),
            "gamma": self.gamma,
            "clip_grad_value": (
                self.clip_grad_value
            ),
            "accumulation_steps": (
                self.accumulation_steps
            ),
            "save_freq": self.save_freq,
            "temperature": (
                self.contrastive_loss.temperature
            ),
            "freq_mask_ratio": (
                self.augmentation.mask_ratio
            ),
            "preserve_dc": (
                self.augmentation.preserve_dc
            ),
            "save_dir": self.save_path,
            "epoch_logs": {},
        }

        self.initialize_log_file(
            log_metadata
        )

        self.logger.info(
            "Starting contrastive pretraining..."
        )

        for epoch in range(
            1,
            self.max_epochs + 1,
        ):
            # 正确统计每个 epoch 的峰值显存。
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(
                    self.device
                )
                torch.cuda.synchronize(
                    self.device
                )

            start_time = time()

            terminate, train_loss = (
                self.train_epoch()
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize(
                    self.device
                )

            train_time = (
                time() - start_time
            )

            if torch.cuda.is_available():
                peak_memory_allocated_mb = (
                    torch.cuda.max_memory_allocated(
                        self.device
                    )
                    / (1024 ** 2)
                )

                peak_memory_reserved_mb = (
                    torch.cuda.max_memory_reserved(
                        self.device
                    )
                    / (1024 ** 2)
                )
            else:
                peak_memory_allocated_mb = 0.0
                peak_memory_reserved_mb = 0.0

            current_lr = (
                self.optimizer
                .param_groups[0]["lr"]
            )

            self.logger.info(
                f"[Contrastive Pretrain] "
                f"Epoch {epoch}/{self.max_epochs}, "
                f"loss={train_loss:.6f}, "
                f"time={train_time:.2f}s, "
                f"peak allocated="
                f"{peak_memory_allocated_mb:.2f} MB, "
                f"peak reserved="
                f"{peak_memory_reserved_mb:.2f} MB"
            )

            epoch_log = {
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "loss": float(train_loss),
                "time_seconds": float(
                    train_time
                ),
                "peak_memory_allocated_mb": float(
                    peak_memory_allocated_mb
                ),
                "peak_memory_reserved_mb": float(
                    peak_memory_reserved_mb
                ),
                "current_learning_rate": float(
                    current_lr
                ),
            }

            self._log_epoch_info(
                epoch=f"Contrastive {epoch}",
                epoch_log=epoch_log,
            )

            if terminate:
                self.logger.error(
                    "Contrastive pretraining "
                    "terminated early."
                )
                break

            # 固定频率保存，同时保证最后一个 epoch 一定保存。
            if (
                epoch % self.save_freq == 0
                or epoch == self.max_epochs
            ):
                self.save_checkpoint(epoch)

            self.scheduler.step()

        self.logger.info(
            "Contrastive pretraining completed."
        )