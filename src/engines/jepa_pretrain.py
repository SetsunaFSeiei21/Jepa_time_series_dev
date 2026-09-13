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
from collections import Counter
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
        # Reference-ratio loss balancing
        # ------------------------------------------------------------

        self.balance_reference_loss = bool(
            config.get(
                "balance_reference_loss",
                False,
            )
        )

        self.reference_sample_counts = {}
        self.reference_loss_weights = None

        if self.balance_reference_loss:
            self._initialize_reference_loss_balance()
        
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
        
    def _initialize_reference_loss_balance(
        self,
    ) -> None:
        """
        Compute inverse-frequency loss weights for every
        reference ratio.

        For ratio k:

            weight_k = total_samples
                    / (num_ratios * samples_k)

        This ensures:

            samples_k * weight_k

        is identical for every ratio.
        """

        dataset = self.train_loader.dataset

        if not hasattr(
            dataset,
            "sample_index_map",
        ):
            raise AttributeError(
                "balance_reference_loss=true requires "
                "the training dataset to expose "
                "sample_index_map."
            )

        reference_counter = Counter()

        for (
            reference_index,
            _,
        ) in dataset.sample_index_map:
            reference_index = int(
                reference_index
            )

            # -1 denotes an ordinary sample without
            # a reference-bank condition.
            if reference_index >= 0:
                reference_counter[
                    reference_index
                ] += 1

        if len(reference_counter) == 0:
            raise RuntimeError(
                "balance_reference_loss=true, but no "
                "reference-conditioned samples were found."
            )

        sorted_indices = sorted(
            reference_counter.keys()
        )

        expected_indices = list(
            range(
                len(sorted_indices)
            )
        )

        if sorted_indices != expected_indices:
            raise RuntimeError(
                "Reference indices must be contiguous and "
                "start from zero. "
                f"Found indices={sorted_indices}."
            )

        total_samples = sum(
            reference_counter.values()
        )

        num_ratios = len(
            reference_counter
        )

        weights = torch.empty(
            num_ratios,
            dtype=torch.float32,
            device=self.device,
        )

        for reference_index in sorted_indices:
            sample_count = reference_counter[
                reference_index
            ]

            weights[reference_index] = (
                float(total_samples)
                / (
                    float(num_ratios)
                    * float(sample_count)
                )
            )

        self.reference_sample_counts = {
            int(reference_index): int(
                reference_counter[
                    reference_index
                ]
            )
            for reference_index
            in sorted_indices
        }

        self.reference_loss_weights = (
            weights
        )

        ratio_values = getattr(
            dataset,
            "reference_ratios",
            None,
        )

        if ratio_values is None:
            ratio_values = sorted_indices

        balance_description = []

        for reference_index in sorted_indices:
            balance_description.append(
                {
                    "reference_index": (
                        reference_index
                    ),
                    "ratio": float(
                        ratio_values[
                            reference_index
                        ]
                    ),
                    "sample_count": (
                        self.reference_sample_counts[
                            reference_index
                        ]
                    ),
                    "loss_weight": float(
                        self.reference_loss_weights[
                            reference_index
                        ].item()
                    ),
                }
            )

        self.logger.info(
            "Reference-ratio loss balancing enabled: "
            f"{balance_description}"
        )
    
    def _compute_jepa_loss(
        self,
        Ey_pred: torch.Tensor,
        Ey: torch.Tensor,
        reference_index: torch.Tensor,
    ):
        """
        Compute ordinary or reference-ratio-balanced
        JEPA loss.

        For a batch containing ratio groups k:

            loss =
                sum_k [
                    (batch_count_k / batch_size)
                    * weight_k
                    * loss_k
                ]

        With batch_size=1, this becomes:

            loss = weight_k * loss_k
        """

        # --------------------------------------------------------
        # Ordinary JEPA behavior
        # --------------------------------------------------------

        if not self.balance_reference_loss:
            loss, loss_dict = self.jepa_loss(
                Ey_pred,
                Ey,
            )

            loss_dict = dict(
                loss_dict
            )

            loss_dict["loss_raw"] = (
                loss_dict["loss"]
            )

            loss_dict["balance_weight"] = (
                torch.ones(
                    (),
                    device=loss.device,
                    dtype=loss.dtype,
                )
            )

            return loss, loss_dict

        # --------------------------------------------------------
        # Reference-balanced behavior
        # --------------------------------------------------------

        if reference_index is None:
            raise RuntimeError(
                "Reference-ratio loss balancing is enabled, "
                "but the current batch has no "
                "reference_index."
            )

        reference_index = (
            reference_index
            .reshape(-1)
            .to(
                device=Ey_pred.device,
                dtype=torch.long,
            )
        )

        batch_size = Ey_pred.shape[0]

        if (
            reference_index.numel()
            != batch_size
        ):
            raise RuntimeError(
                "reference_index batch size does not "
                "match JEPA embedding batch size: "
                f"{reference_index.numel()} "
                f"vs {batch_size}."
            )

        if (
            reference_index.min().item() < 0
            or reference_index.max().item()
            >= self.reference_loss_weights.numel()
        ):
            raise IndexError(
                "reference_index is outside the "
                "reference-loss weight table."
            )

        balanced_loss = Ey_pred.new_zeros(
            ()
        )

        balanced_align = Ey_pred.new_zeros(
            ()
        )

        balanced_reg = Ey_pred.new_zeros(
            ()
        )

        raw_loss = Ey_pred.new_zeros(
            ()
        )

        raw_align = Ey_pred.new_zeros(
            ()
        )

        raw_reg = Ey_pred.new_zeros(
            ()
        )

        unique_reference_indices = (
            torch.unique(
                reference_index
            )
        )

        for current_index in (
            unique_reference_indices
        ):
            ratio_mask = (
                reference_index
                == current_index
            )

            ratio_batch_count = (
                ratio_mask.sum()
            )

            ratio_fraction = (
                ratio_batch_count.to(
                    dtype=Ey_pred.dtype
                )
                / float(batch_size)
            )

            ratio_loss, ratio_loss_dict = (
                self.jepa_loss(
                    Ey_pred[ratio_mask],
                    Ey[ratio_mask],
                )
            )

            ratio_weight = (
                self.reference_loss_weights[
                    current_index
                ].to(
                    device=Ey_pred.device,
                    dtype=Ey_pred.dtype,
                )
            )

            coefficient = (
                ratio_fraction
                * ratio_weight
            )

            # Differentiable balanced objective.
            balanced_loss = (
                balanced_loss
                + coefficient
                * ratio_loss
            )

            # Detached diagnostics.
            balanced_align = (
                balanced_align
                + coefficient.detach()
                * ratio_loss_dict[
                    "loss_align"
                ]
            )

            balanced_reg = (
                balanced_reg
                + coefficient.detach()
                * ratio_loss_dict[
                    "loss_reg"
                ]
            )

            raw_loss = (
                raw_loss
                + ratio_fraction.detach()
                * ratio_loss.detach()
            )

            raw_align = (
                raw_align
                + ratio_fraction.detach()
                * ratio_loss_dict[
                    "loss_align"
                ]
            )

            raw_reg = (
                raw_reg
                + ratio_fraction.detach()
                * ratio_loss_dict[
                    "loss_reg"
                ]
            )

        batch_weights = (
            self.reference_loss_weights
            .index_select(
                dim=0,
                index=reference_index,
            )
            .to(
                device=Ey_pred.device,
                dtype=Ey_pred.dtype,
            )
        )

        return balanced_loss, {
            "loss": (
                balanced_loss.detach()
            ),
            "loss_raw": (
                raw_loss.detach()
            ),
            "loss_align": (
                balanced_align.detach()
            ),
            "loss_reg": (
                balanced_reg.detach()
            ),
            "loss_align_raw": (
                raw_align.detach()
            ),
            "loss_reg_raw": (
                raw_reg.detach()
            ),
            "balance_weight": (
                batch_weights.mean().detach()
            ),
        }
    
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
    
    @torch.no_grad()
    def _sync_spectral_buffers_to_target(self):
        """
        Keep the online and target spectral contexts identical.

        Spectral buffers are running statistics rather than EMA model
        parameters, so they should be copied directly.
        """

        spectral_buffer_names = (
            "global_spectrum",
            "spectrum_seen_samples",
            "last_spectrum_update_norm",
        )

        if not hasattr(
            self.model,
            "global_spectrum",
        ):
            return

        for name in spectral_buffer_names:
            if not hasattr(self.model, name):
                raise AttributeError(
                    f"Online model is missing spectral buffer: {name}"
                )

            if not hasattr(self.target_model, name):
                raise AttributeError(
                    f"Target model is missing spectral buffer: {name}"
                )

            online_buffer = getattr(
                self.model,
                name,
            )

            target_buffer = getattr(
                self.target_model,
                name,
            )

            target_buffer.copy_(online_buffer)

    def _format_jepa_embedding(
        self,
        model: torch.nn.Module,
        embedding,
    ) -> torch.Tensor:
        """
        Convert model-specific JEPA representations into a Tensor
        accepted by JEPASIGRegLoss.
        """

        # Crossformer returns a multi-scale list.
        if isinstance(embedding, (list, tuple)):
            if not hasattr(model, "_pack_multiscale"):
                raise TypeError(
                    f"{type(model).__name__} returned a list/tuple embedding "
                    "but does not implement _pack_multiscale()."
                )

            embedding = model._pack_multiscale(embedding)

        if not isinstance(embedding, torch.Tensor):
            raise TypeError(
                f"Expected JEPA embedding to be a Tensor after formatting, "
                f"got {type(embedding)}."
            )

        # iTransformer may append time-feature tokens after node tokens.
        if type(model).__name__.lower() == "itransformer":
            embedding = embedding[:, : model.node_num, :]

            # (B, N, D) -> (B, 1, N, D)
            embedding = embedding.unsqueeze(1)

        return embedding

    def _forward_jepa(self, batch: BatchData):
        """
        Online encoder:
            input -> online encoder -> predictor -> Ey_pred

        EMA target encoder:
            target -> target encoder -> Ey
        """

        if not hasattr(self.model, "encode"):
            raise AttributeError(
                f"{type(self.model).__name__} does not implement encode()."
            )

        if not hasattr(self.model, "predict"):
            raise AttributeError(
                f"{type(self.model).__name__} does not implement predict()."
            )

        if not hasattr(self.target_model, "encode"):
            raise AttributeError(
                f"{type(self.target_model).__name__} does not implement encode()."
            )

        encode_kwargs = {}

        if batch.reference_index is not None:
            encode_kwargs["reference_index"] = (
                batch.reference_index
            )

        # Online path.
        Ex = self.model.encode(
            batch.input_seq,
            batch.input_features,
            **encode_kwargs,
        )

        Ey_pred = self.model.predict(Ex)

        # EMA target path.
        with torch.no_grad():
            Ey = self.target_model.encode(
                batch.target_seq,
                batch.target_features,
                **encode_kwargs,
            )

        # Handle Crossformer, iTransformer and ordinary Tensor outputs.
        Ey_pred = self._format_jepa_embedding(
            self.model,
            Ey_pred,
        )

        Ey = self._format_jepa_embedding(
            self.target_model,
            Ey,
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
            Receives gradients and is updated by AdamW after each
            gradient-accumulation group.

        Target network:
            Receives no gradients and is updated by EMA after each
            optimizer step.

        Global spectral context C:
            Spectra from all micro-batches in one gradient-accumulation
            group are first averaged. C is then updated once using that
            group-level mean spectrum.

        For accumulation_steps = 8:

            S_group
                = mean(S_1, S_2, ..., S_8)

            C_new
                = (1 - spectral_update_rate) * C_old
                + spectral_update_rate * S_group

        Therefore, all micro-batches in the same accumulation group use
        the same C.
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
            self.logger.error(
                "JEPA train_loader contains no batches."
            )
            return True, 0.0

        valid_micro_steps = 0

        last_ema_momentum = (
            self._get_ema_momentum()
        )

        # ------------------------------------------------------------
        # Spectral diagnostics
        # ------------------------------------------------------------
        #
        # These statistics now count group-level C updates rather than
        # individual micro-batch updates.
        # ------------------------------------------------------------
        spectrum_update_sum = 0.0
        spectrum_update_count = 0
        last_spectrum_update_norm = 0.0

        # ------------------------------------------------------------
        # Spectrum accumulator for one gradient-accumulation group
        # ------------------------------------------------------------
        #
        # spectrum_group_sum stores:
        #
        #     sum_i sample_count_i * batch_spectrum_i
        #
        # spectrum_group_samples stores:
        #
        #     sum_i sample_count_i
        #
        # This produces a sample-weighted mean spectrum and correctly
        # handles a smaller final batch.
        # ------------------------------------------------------------
        spectrum_group_sum = None
        spectrum_group_samples = 0

        self.optimizer.zero_grad(
            set_to_none=True
        )

        pbar = tqdm(
            self.train_loader,
            desc="JEPA Pretraining",
            file=sys.stdout,
        )

        for batch_idx, batch_data in enumerate(pbar):
            batch_data: BatchData

            batch = batch_data.to_device(
                self.device
            )

            # ------------------------------------------------------------
            # 1. Replace masked values before normalization
            # ------------------------------------------------------------
            if batch.input_mask is not None:
                batch.input_seq = torch.where(
                    batch.input_mask,
                    torch.zeros_like(
                        batch.input_seq
                    ),
                    batch.input_seq,
                )

            if batch.target_mask is not None:
                batch.target_seq = torch.where(
                    batch.target_mask,
                    torch.zeros_like(
                        batch.target_seq
                    ),
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
            # 2. Normalize input and target using the same coordinate
            #    system
            # ------------------------------------------------------------
            batch.input_seq = (
                self.scalar.fit_transform(
                    batch.input_seq
                )
            )

            batch.target_seq = (
                self.scalar.transform(
                    batch.target_seq
                )
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
            # 3. Compute current micro-batch spectrum
            # ------------------------------------------------------------
            batch_spectrum = None
            batch_spectrum_samples = 0

            if hasattr(
                self.model,
                "compute_batch_spectrum",
            ):
                (
                    batch_spectrum,
                    batch_spectrum_samples,
                ) = self.model.compute_batch_spectrum(
                    batch.input_seq
                )

            # ------------------------------------------------------------
            # 4. Accumulate spectra inside the current effective batch
            # ------------------------------------------------------------
            #
            # Do not update C here.
            #
            # Each micro-batch spectrum is multiplied by its number of
            # samples. This is equivalent to computing the mean spectrum
            # over the combined effective batch.
            # ------------------------------------------------------------
            if (
                batch_spectrum is not None
                and batch_spectrum_samples > 0
                and hasattr(
                    self.model,
                    "update_global_spectrum",
                )
            ):
                weighted_batch_spectrum = (
                    batch_spectrum
                    * float(
                        batch_spectrum_samples
                    )
                )

                if spectrum_group_sum is None:
                    spectrum_group_sum = (
                        weighted_batch_spectrum.clone()
                    )
                else:
                    spectrum_group_sum.add_(
                        weighted_batch_spectrum
                    )

                spectrum_group_samples += int(
                    batch_spectrum_samples
                )

            # ------------------------------------------------------------
            # 5. Online encoder + EMA target encoder
            # ------------------------------------------------------------
            #
            # All micro-batches in the current accumulation group use
            # the same old C because C is not updated until should_update.
            # ------------------------------------------------------------
            Ey, Ey_pred = self._forward_jepa(
                batch
            )

            # ------------------------------------------------------------
            # 6. Validate hidden representations
            # ------------------------------------------------------------
            has_invalid_embedding = (
                torch.isnan(Ey).any()
                or torch.isinf(Ey).any()
                or torch.isnan(Ey_pred).any()
                or torch.isinf(Ey_pred).any()
            )

            if has_invalid_embedding:
                self.logger.error(
                    "JEPA hidden representation contains "
                    "NaN/Inf at "
                    f"batch {batch_idx}. "
                    f"Ey_nan="
                    f"{torch.isnan(Ey).sum().item()}, "
                    f"Ey_inf="
                    f"{torch.isinf(Ey).sum().item()}, "
                    f"Ey_pred_nan="
                    f"{torch.isnan(Ey_pred).sum().item()}, "
                    f"Ey_pred_inf="
                    f"{torch.isinf(Ey_pred).sum().item()}."
                )

                terminate = True
                break

            # ------------------------------------------------------------
            # 7. Compute JEPA loss
            # ------------------------------------------------------------
            loss, loss_dict = (
                self._compute_jepa_loss(
                    Ey_pred=Ey_pred,
                    Ey=Ey,
                    reference_index=(
                        batch.reference_index
                    ),
                )
            )

            if (
                torch.isnan(loss)
                or torch.isinf(loss)
            ):
                self.logger.error(
                    "JEPA loss is NaN/Inf at "
                    f"batch {batch_idx}. "
                    f"loss_align="
                    f"{loss_dict['loss_align'].item()}, "
                    f"loss_reg="
                    f"{loss_dict['loss_reg'].item()}."
                )

                terminate = True
                break

            # ------------------------------------------------------------
            # 8. Determine current accumulation-group size
            # ------------------------------------------------------------
            #
            # Example:
            #
            #     total_batches = 19
            #     accumulation_steps = 8
            #
            # Groups:
            #
            #     8, 8, 3
            #
            # The final group divides its loss by 3 instead of 8.
            # ------------------------------------------------------------
            group_start = (
                batch_idx // accumulation_steps
            ) * accumulation_steps

            current_group_size = min(
                accumulation_steps,
                total_batches - group_start,
            )

            micro_step_in_group = (
                batch_idx
                - group_start
                + 1
            )

            # ------------------------------------------------------------
            # 9. Backward with gradient accumulation
            # ------------------------------------------------------------
            loss_for_backward = (
                loss / current_group_size
            )

            loss_for_backward.backward()

            valid_micro_steps += 1

            avg_loss = (
                avg_loss
                * (
                    (valid_micro_steps - 1)
                    / valid_micro_steps
                )
                + loss.item()
                / valid_micro_steps
            )

            # Complete group or final partial group.
            should_update = (
                micro_step_in_group
                == current_group_size
            )

            # ------------------------------------------------------------
            # 10. Update model, C and target encoder once per group
            # ------------------------------------------------------------
            if should_update:
                # --------------------------------------------------------
                # 10.1 Gradient clipping
                # --------------------------------------------------------
                if self.clip_grad_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad_value,
                    )

                # --------------------------------------------------------
                # 10.2 Update online model parameters
                # --------------------------------------------------------
                self.optimizer.step()

                # --------------------------------------------------------
                # 10.3 Compute group-level mean spectrum and update C
                # --------------------------------------------------------
                #
                # For equal micro-batch sizes:
                #
                #     group_spectrum
                #         = mean(S_1, ..., S_8)
                #
                # For unequal sizes, this is the sample-weighted mean.
                # --------------------------------------------------------
                if (
                    spectrum_group_sum is not None
                    and spectrum_group_samples > 0
                    and hasattr(
                        self.model,
                        "update_global_spectrum",
                    )
                ):
                    group_spectrum = (
                        spectrum_group_sum
                        / float(
                            spectrum_group_samples
                        )
                    )

                    last_spectrum_update_norm = (
                        self.model.update_global_spectrum(
                            group_spectrum,
                            spectrum_group_samples,
                        )
                    )

                    spectrum_update_sum += (
                        last_spectrum_update_norm
                    )

                    spectrum_update_count += 1

                # --------------------------------------------------------
                # 10.4 Update EMA target encoder
                # --------------------------------------------------------
                #
                # _update_target_encoder() updates target parameters and
                # copies registered buffers from the online model.
                # Therefore, the newly updated C is also copied.
                # --------------------------------------------------------
                last_ema_momentum = (
                    self._update_target_encoder()
                )

                # Keep this explicit synchronization for safety and
                # readability. It ensures that target C is identical to
                # online C before the next accumulation group.
                self._sync_spectral_buffers_to_target()

                # --------------------------------------------------------
                # 10.5 Clear parameter gradients
                # --------------------------------------------------------
                self.optimizer.zero_grad(
                    set_to_none=True
                )

                # --------------------------------------------------------
                # 10.6 Clear spectral accumulators
                # --------------------------------------------------------
                spectrum_group_sum = None
                spectrum_group_samples = 0

            pbar.set_description(
                "JEPA Balanced: "
                f"{loss.item():.4f}, "
                f"Raw: "
                f"{loss_dict['loss_raw'].item():.4f}, "
                f"W: "
                f"{loss_dict['balance_weight'].item():.4f}, "
                f"Align: "
                f"{loss_dict['loss_align'].item():.4f}, "
                f"SIGReg: "
                f"{loss_dict['loss_reg'].item():.4f}, "
                f"Avg: {avg_loss:.4f}, "
                f"Accum: "
                f"{micro_step_in_group}/"
                f"{current_group_size}, "
                f"EMA: "
                f"{last_ema_momentum:.6f}, "
                f"C-delta: "
                f"{last_spectrum_update_norm:.6e}"
            )

        if valid_micro_steps == 0:
            self.logger.error(
                "No valid JEPA pretraining batches "
                "in this epoch."
            )

            terminate = True

        # Mean norm over group-level C updates.
        if spectrum_update_count > 0:
            self.last_epoch_spectrum_update_mean = (
                spectrum_update_sum
                / spectrum_update_count
            )
        else:
            self.last_epoch_spectrum_update_mean = 0.0

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
            if hasattr(
                self.model,
                "get_spectral_diagnostics",
            ):
                spectral_diagnostics = (
                    self.model.get_spectral_diagnostics()
                )

                self.logger.info(
                    "Spectral context: "
                    f"seen_samples="
                    f"{spectral_diagnostics['seen_samples']}, "
                    f"last_update_norm="
                    f"{spectral_diagnostics['last_update_norm']:.8e}, "
                    f"epoch_mean_update_norm="
                    f"{self.last_epoch_spectrum_update_mean:.8e}, "
                    f"context_gate="
                    f"{spectral_diagnostics['context_gate']:.6f}, "
                    f"spectrum_norm="
                    f"{spectral_diagnostics['spectrum_norm']:.6f}"
                )
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
            if hasattr(
                self.model,
                "get_spectral_diagnostics",
            ):
                pretrain_log["spectral_context"] = {
                    **self.model.get_spectral_diagnostics(),
                    "epoch_mean_update_norm": float(
                        self.last_epoch_spectrum_update_mean
                    ),
                }
            self._log_epoch_info(epoch=f"Pretrain {epoch}", epoch_log=pretrain_log)

            self.scheduler.step()

        self.logger.info("JEPA pretraining completed.")
