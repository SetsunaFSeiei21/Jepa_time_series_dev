import math
from importlib import import_module
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


_BasePatchTST = import_module(
    "src.models.time-series.patchtst"
).PatchTST


def _to_logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError(
            "Gate initialization must lie "
            "strictly in (0,1)."
        )

    return math.log(
        probability
        / (1.0 - probability)
    )


class PatchTST(_BasePatchTST):
    def __init__(
        self,
        spectral_smoothing_kernel: int = 5,
        spectral_normalize: bool = True,
        spectral_remove_dc: bool = True,
        spectral_eps: float = 1e-8,
        spectral_dropout: float = 0.0,
        reference_gate_init: float = 0.05,
        local_gate_init: float = 0.05,
        time_gate_init: float = 0.05,
        **args,
    ):
        super().__init__(**args)

        if self.decomposition:
            raise NotImplementedError(
                "Reference-spectral PatchTST "
                "supports decomposition=False only."
            )

        if self.his_len != self.pred_len:
            raise ValueError(
                "Reference-spectral JEPA requires "
                "his_len == pred_len."
            )

        if spectral_smoothing_kernel <= 0:
            raise ValueError(
                "spectral_smoothing_kernel "
                "must be positive."
            )

        if spectral_eps <= 0:
            raise ValueError(
                "spectral_eps must be positive."
            )

        self.spectral_smoothing_kernel = int(
            spectral_smoothing_kernel
        )

        self.spectral_normalize = bool(
            spectral_normalize
        )

        self.spectral_remove_dc = bool(
            spectral_remove_dc
        )

        self.spectral_eps = float(
            spectral_eps
        )

        self.spectral_freq_bins = (
            self.his_len // 2 + 1
        )

        # Shape after initialization:
        #     (K,N,F)
        #
        # persistent=False:
        #     C bank is not saved into checkpoint.
        self.register_buffer(
            "reference_spectra",
            torch.empty(
                0,
                self.node_num,
                self.spectral_freq_bins,
            ),
            persistent=False,
        )

        self.register_buffer(
            "reference_ready",
            torch.tensor(
                False,
                dtype=torch.bool,
            ),
            persistent=False,
        )

        self.reference_projector = nn.Sequential(
            nn.Linear(
                self.spectral_freq_bins,
                self.d_model,
            ),
            nn.GELU(),
            nn.Linear(
                self.d_model,
                self.d_model,
            ),
            nn.LayerNorm(
                self.d_model,
            ),
        )

        self.local_projector = nn.Sequential(
            nn.Linear(
                self.spectral_freq_bins,
                self.d_model,
            ),
            nn.GELU(),
            nn.Linear(
                self.d_model,
                self.d_model,
            ),
            nn.LayerNorm(
                self.d_model,
            ),
        )

        self.time_projector = nn.Sequential(
            nn.Linear(
                1,
                self.d_model,
            ),
            nn.GELU(),
            nn.Linear(
                self.d_model,
                self.d_model,
            ),
            nn.LayerNorm(
                self.d_model,
            ),
        )

        self.condition_dropout = nn.Dropout(
            spectral_dropout
        )

        self.reference_gate_logit = nn.Parameter(
            torch.tensor(
                _to_logit(
                    reference_gate_init
                ),
                dtype=torch.float32,
            )
        )

        self.local_gate_logit = nn.Parameter(
            torch.tensor(
                _to_logit(
                    local_gate_init
                ),
                dtype=torch.float32,
            )
        )

        self.time_gate_logit = nn.Parameter(
            torch.tensor(
                _to_logit(
                    time_gate_init
                ),
                dtype=torch.float32,
            )
        )

    @torch.no_grad()
    def _compute_segment_spectrum(
        self,
        seq: Tensor,
    ) -> Tensor:
        seq = self._to_bnt(
            seq,
            name="spectral_sequence",
        )

        seq = torch.nan_to_num(
            seq.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if self.spectral_remove_dc:
            seq = (
                seq
                - seq.mean(
                    dim=-1,
                    keepdim=True,
                )
            )

        std = seq.std(
            dim=-1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(
            self.spectral_eps
        )

        seq = seq / std

        spectrum = torch.fft.rfft(
            seq,
            dim=-1,
            norm="ortho",
        )

        log_power = torch.log1p(
            spectrum.abs().square()
        )

        batch_size, node_num, freq_bins = (
            log_power.shape
        )

        flattened = log_power.reshape(
            batch_size * node_num,
            1,
            freq_bins,
        )

        kernel = min(
            self.spectral_smoothing_kernel,
            freq_bins,
        )

        if kernel % 2 == 0:
            kernel -= 1

        if kernel > 1:
            flattened = F.avg_pool1d(
                flattened,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
                count_include_pad=False,
            )

        if (
            flattened.size(-1)
            != self.spectral_freq_bins
        ):
            flattened = F.interpolate(
                flattened,
                size=self.spectral_freq_bins,
                mode="linear",
                align_corners=False,
            )

        log_power = flattened.reshape(
            batch_size,
            node_num,
            self.spectral_freq_bins,
        )

        if self.spectral_normalize:
            log_power = log_power / (
                log_power.sum(
                    dim=-1,
                    keepdim=True,
                )
                + self.spectral_eps
            )

        return log_power

    @torch.no_grad()
    def set_reference_series_bank(
        self,
        reference_sequences: Sequence[Tensor],
        reference_masks: Optional[
            Sequence[Optional[Tensor]]
        ] = None,
    ):
        if len(reference_sequences) == 0:
            raise ValueError(
                "reference_sequences must not "
                "be empty."
            )

        if reference_masks is None:
            reference_masks = [
                None
                for _ in reference_sequences
            ]

        if (
            len(reference_sequences)
            != len(reference_masks)
        ):
            raise ValueError(
                "reference_sequences and "
                "reference_masks must have "
                "the same length."
            )

        spectra = []

        for (
            reference_seq,
            reference_mask,
        ) in zip(
            reference_sequences,
            reference_masks,
        ):
            if reference_mask is not None:
                reference_seq = torch.where(
                    reference_mask,
                    torch.zeros_like(
                        reference_seq
                    ),
                    reference_seq,
                )

            spectrum = (
                self._compute_segment_spectrum(
                    reference_seq
                )
            )

            spectrum = spectrum.mean(
                dim=0,
                keepdim=True,
            )

            spectra.append(spectrum)

        reference_spectra = torch.cat(
            spectra,
            dim=0,
        )

        self.reference_spectra = (
            reference_spectra.to(
                device=(
                    self.reference_spectra.device
                ),
                dtype=(
                    self.reference_spectra.dtype
                ),
            )
        )

        self.reference_ready.fill_(True)

    def _select_reference_context(
        self,
        reference_index: Tensor,
        batch_size: int,
        patch_num: int,
    ) -> Tensor:
        if not bool(
            self.reference_ready.item()
        ):
            raise RuntimeError(
                "Reference spectrum bank has "
                "not been initialized."
            )

        if reference_index is None:
            raise ValueError(
                "reference_index is required."
            )

        reference_index = (
            reference_index.to(
                device=(
                    self.reference_spectra.device
                ),
                dtype=torch.long,
            )
        )

        if reference_index.dim() != 1:
            raise ValueError(
                "reference_index must have "
                "shape (B,)."
            )

        if (
            reference_index.size(0)
            != batch_size
        ):
            raise ValueError(
                "reference_index batch size "
                "does not match input batch size."
            )

        if (
            reference_index.min().item() < 0
            or reference_index.max().item()
            >= self.reference_spectra.size(0)
        ):
            raise IndexError(
                "reference_index is outside "
                "the reference bank."
            )

        selected_spectrum = (
            self.reference_spectra.index_select(
                dim=0,
                index=reference_index,
            )
        )

        context = self.reference_projector(
            selected_spectrum
        )

        return (
            context
            .unsqueeze(2)
            .expand(
                -1,
                -1,
                patch_num,
                -1,
            )
        )

    def _build_time_context(
        self,
        seq_features: Tensor,
        model,
        batch_size: int,
        node_num: int,
        patch_num: int,
    ) -> Tensor:
        if seq_features is None:
            raise ValueError(
                "Timestamp features are required."
            )

        if seq_features.dim() != 3:
            raise ValueError(
                "seq_features must have shape "
                "(B,T,D_time)."
            )

        # Last feature is tau_rel.
        relative_time = (
            seq_features[..., -1:]
            .transpose(1, 2)
        )

        relative_time = (
            model.padding_patch_layer(
                relative_time
            )
        )

        time_patches = relative_time.unfold(
            dimension=-1,
            size=model.patch_len,
            step=model.stride,
        )

        if (
            time_patches.size(2)
            != patch_num
        ):
            raise RuntimeError(
                "Timestamp patch count does not "
                "match sequence patch count."
            )

        time_patches = (
            time_patches
            .mean(dim=-1)
            .permute(0, 2, 1)
        )

        time_context = self.time_projector(
            time_patches
        )

        return (
            time_context
            .unsqueeze(1)
            .expand(
                batch_size,
                node_num,
                -1,
                -1,
            )
        )

    def _encode_conditioned(
        self,
        seq: Tensor,
        seq_features: Tensor,
        reference_index: Tensor,
        model,
    ) -> Tensor:
        raw_seq = seq

        if model.revin:
            seq = model.revin_layer(
                seq,
                torch.tensor(
                    True,
                    dtype=torch.bool,
                    device=seq.device,
                ),
            )

        seq = model.padding_patch_layer(
            seq
        )

        (
            batch_size,
            node_num,
            padded_len,
        ) = seq.shape

        patches = seq.reshape(
            -1,
            1,
            1,
            padded_len,
        )

        patches = model.unfold(
            patches
        )

        patches = (
            patches
            .permute(0, 2, 1)
            .reshape(
                batch_size,
                node_num,
                -1,
                model.patch_len,
            )
        )

        patch_tokens = model.backbone.W_P(
            patches
        )

        patch_num = patch_tokens.size(2)

        local_spectrum = (
            self._compute_segment_spectrum(
                raw_seq
            )
        )

        local_context = (
            self.local_projector(
                local_spectrum
            )
            .unsqueeze(2)
            .expand(
                -1,
                -1,
                patch_num,
                -1,
            )
        )

        reference_context = (
            self._select_reference_context(
                reference_index=(
                    reference_index
                ),
                batch_size=batch_size,
                patch_num=patch_num,
            )
        )

        time_context = (
            self._build_time_context(
                seq_features=seq_features,
                model=model,
                batch_size=batch_size,
                node_num=node_num,
                patch_num=patch_num,
            )
        )

        x = (
            patch_tokens
            + model.backbone.W_pos[
                None,
                None,
                :,
                :,
            ]
            + torch.sigmoid(
                self.reference_gate_logit
            )
            * self.condition_dropout(
                reference_context
            )
            + torch.sigmoid(
                self.local_gate_logit
            )
            * self.condition_dropout(
                local_context
            )
            + torch.sigmoid(
                self.time_gate_logit
            )
            * self.condition_dropout(
                time_context
            )
        )

        x = model.backbone.dropout(x)

        x = x.reshape(
            batch_size * node_num,
            patch_num,
            self.d_model,
        )

        if model.backbone.res_attention:
            scores = None

            for layer in model.backbone.layers:
                x, scores = layer(
                    x,
                    prev=scores,
                )
        else:
            for layer in model.backbone.layers:
                x = layer(x)

        x = x.reshape(
            batch_size,
            node_num,
            patch_num,
            self.d_model,
        )

        return x.permute(
            0,
            1,
            3,
            2,
        )

    def encode(
        self,
        seq,
        seq_features=None,
        reference_index=None,
        *args,
        **kwargs,
    ):
        self._check_non_decomposition()

        seq = self._to_bnt(
            seq,
            name="seq",
        )

        if seq.size(-1) != self.his_len:
            raise ValueError(
                "Reference-spectral PatchTST "
                f"expects length {self.his_len}, "
                f"but got {seq.size(-1)}."
            )

        return self._encode_conditioned(
            seq=seq,
            seq_features=seq_features,
            reference_index=reference_index,
            model=self.model,
        )

    def forward(
        self,
        input_seq,
        input_features=None,
        target_features=None,
        target_seq=None,
        label=None,
        reference_index=None,
        mode="finetune",
        *args,
        **kwargs,
    ):
        if label is None:
            label = target_seq

        if mode == "pretrain":
            if label is None:
                raise ValueError(
                    "Pretraining requires label "
                    "or target_seq."
                )

            Ex = self.encode(
                input_seq,
                input_features,
                reference_index=(
                    reference_index
                ),
            )

            Ey = self.encode(
                label,
                target_features,
                reference_index=(
                    reference_index
                ),
            )

            Ey_pred = self.predict(Ex)

            if Ey_pred.shape != Ey.shape:
                raise RuntimeError(
                    "JEPA hidden shape mismatch: "
                    f"{tuple(Ey_pred.shape)} vs "
                    f"{tuple(Ey.shape)}."
                )

            return Ey, Ey_pred

        if mode == "finetune":
            Ex = self.encode(
                input_seq,
                input_features,
                reference_index=(
                    reference_index
                ),
            )

            Ey_pred = self.predict(Ex)

            return self.decode(
                Ey_pred,
                input_seq=input_seq,
            )

        raise ValueError(
            f"Unsupported mode: {mode}"
        )