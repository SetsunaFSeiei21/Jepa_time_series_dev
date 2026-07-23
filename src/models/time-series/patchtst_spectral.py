"""
PatchTST with a running global spectral context for JEPA.

C is not a freely learned token. It is a registered buffer storing the
sample-weighted running mean of the smoothed log-power spectrum of
normalized historical inputs.

For the current batch:
1. The model uses C estimated from previous batches.
2. JEPA forward/backward is performed.
3. C is updated after backward.
"""

import math
from importlib import import_module
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# The original folder name contains "-", so use importlib.
_BasePatchTST = import_module(
    "src.models.time-series.patchtst"
).PatchTST


class PatchTST(_BasePatchTST):
    """
    PatchTST augmented with a global smoothed spectral context.

    Original hidden representation:
        hidden: (B, N, D, P)

    Global spectral context:
        C: (1, N, F)

    After projection:
        spectral_context: (B, N, D, P)

    Final representation:
        hidden = hidden + sigmoid(gate) * spectral_context
    """

    def __init__(
        self,
        spectral_smoothing_kernel: int = 5,
        spectral_gate_init: float = 0.10,
        spectral_dropout: float = 0.0,
        spectral_normalize: bool = True,
        spectral_remove_dc: bool = False,
        spectral_eps: float = 1e-8,
        **args,
    ):
        super().__init__(**args)

        if self.decomposition:
            raise NotImplementedError(
                "Spectral PatchTST currently supports "
                "decomposition=False only."
            )

        if self.his_len != self.pred_len:
            raise ValueError(
                "Spectral PatchTST JEPA currently requires "
                "his_len == pred_len."
            )

        if spectral_smoothing_kernel <= 0:
            raise ValueError(
                "spectral_smoothing_kernel must be positive."
            )

        if spectral_smoothing_kernel % 2 == 0:
            raise ValueError(
                "spectral_smoothing_kernel must be odd."
            )

        if not 0.0 < spectral_gate_init < 1.0:
            raise ValueError(
                "spectral_gate_init must lie strictly in (0, 1)."
            )

        if spectral_eps <= 0:
            raise ValueError("spectral_eps must be positive.")

        self.spectral_smoothing_kernel = int(
            spectral_smoothing_kernel
        )
        self.spectral_normalize = bool(spectral_normalize)
        self.spectral_remove_dc = bool(spectral_remove_dc)
        self.spectral_eps = float(spectral_eps)

        # rFFT only retains non-negative frequency bins.
        self.spectral_freq_bins = self.his_len // 2 + 1

        if (
            self.spectral_smoothing_kernel
            > self.spectral_freq_bins
        ):
            raise ValueError(
                "spectral_smoothing_kernel cannot exceed "
                f"the number of rFFT bins: "
                f"{self.spectral_freq_bins}."
            )

        # ------------------------------------------------------------
        # Global spectrum C
        # ------------------------------------------------------------
        #
        # Shape:
        #     (1, N, F)
        #
        # N: number of time-series variables
        # F: number of rFFT frequency bins
        #
        # register_buffer means:
        # 1. C is not updated by gradient descent;
        # 2. C moves with model.to(device);
        # 3. C is automatically saved in state_dict.
        # ------------------------------------------------------------
        self.register_buffer(
            "global_spectrum",
            torch.zeros(
                1,
                self.node_num,
                self.spectral_freq_bins,
            ),
        )

        # Number of training samples accumulated into C.
        self.register_buffer(
            "spectrum_seen_samples",
            torch.tensor(0, dtype=torch.long),
        )

        # Used only for monitoring whether C is stabilizing.
        self.register_buffer(
            "last_spectrum_update_norm",
            torch.tensor(0.0),
        )

        # ------------------------------------------------------------
        # Map C from frequency space F to hidden space D.
        # ------------------------------------------------------------
        #
        # Input:
        #     (1, N, F)
        #
        # Output:
        #     (1, N, D)
        #
        # The same projector is shared across variables.
        # ------------------------------------------------------------
        self.spectral_projector = nn.Sequential(
            nn.Linear(
                self.spectral_freq_bins,
                self.d_model,
            ),
            nn.GELU(),
            nn.Linear(
                self.d_model,
                self.d_model,
            ),
        )

        self.spectral_context_norm = nn.LayerNorm(
            self.d_model
        )

        self.spectral_context_dropout = nn.Dropout(
            spectral_dropout
        )

        # Learnable gate:
        #
        #     hidden <- hidden + sigmoid(gate) * context
        #
        # Initialize sigmoid(gate) to spectral_gate_init.
        gate_logit = math.log(
            spectral_gate_init
            / (1.0 - spectral_gate_init)
        )

        self.spectral_gate_logit = nn.Parameter(
            torch.tensor(
                gate_logit,
                dtype=torch.float32,
            )
        )

    @torch.no_grad()
    def compute_batch_spectrum(
        self,
        input_seq: Tensor,
    ) -> Tuple[Tensor, int]:
        """
        Compute the channel-wise smoothed log-power spectrum.

        Args:
            input_seq:
                (B, T, N, 1),
                (B, T, N), or
                (B, N, T)

        Returns:
            batch_spectrum:
                (1, N, F)

            sample_count:
                number of samples contributing to this mean
        """

        # Reuse PatchTST's layout conversion.
        seq = self._to_bnt(
            input_seq,
            name="spectral_input",
        )

        # seq: (B, N, T)
        if seq.size(-1) != self.his_len:
            raise ValueError(
                "Spectral update expects temporal length "
                f"{self.his_len}, but got "
                f"{seq.size(-1)}."
            )

        # Use float32 because half-precision FFT support can be
        # restricted for arbitrary lengths/devices.
        seq = torch.nan_to_num(
            seq.detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Optional explicit DC removal.
        # The repo already normalizes each history window, so this
        # is usually not necessary.
        if self.spectral_remove_dc:
            seq = seq - seq.mean(
                dim=-1,
                keepdim=True,
            )

        # ------------------------------------------------------------
        # rFFT
        # ------------------------------------------------------------
        #
        # seq:
        #     (B, N, T)
        #
        # spectrum:
        #     (B, N, F), complex tensor
        # ------------------------------------------------------------
        spectrum = torch.fft.rfft(
            seq,
            dim=-1,
            norm="ortho",
        )

        # Log-power spectrum:
        #
        #     log(1 + |FFT(x)|^2)
        #
        # This compresses the dynamic range.
        log_power = torch.log1p(
            spectrum.abs().square()
        )

        # ------------------------------------------------------------
        # Smooth neighboring frequency bins.
        # ------------------------------------------------------------
        if self.spectral_smoothing_kernel > 1:
            batch_size, node_num, freq_bins = (
                log_power.shape
            )

            flattened = log_power.reshape(
                batch_size * node_num,
                1,
                freq_bins,
            )

            flattened = F.avg_pool1d(
                flattened,
                kernel_size=(
                    self.spectral_smoothing_kernel
                ),
                stride=1,
                padding=(
                    self.spectral_smoothing_kernel // 2
                ),
                count_include_pad=False,
            )

            log_power = flattened.reshape(
                batch_size,
                node_num,
                freq_bins,
            )

        # Normalize frequency energy for each sample/channel.
        #
        # C then describes spectral shape rather than absolute scale.
        if self.spectral_normalize:
            denominator = log_power.sum(
                dim=-1,
                keepdim=True,
            )

            log_power = log_power / (
                denominator + self.spectral_eps
            )

        # Average only across the batch dimension.
        #
        # We preserve channel-specific spectra:
        #     (B, N, F) -> (1, N, F)
        batch_spectrum = log_power.mean(
            dim=0,
            keepdim=True,
        )

        batch_spectrum = batch_spectrum.to(
            device=self.global_spectrum.device,
            dtype=self.global_spectrum.dtype,
        )

        return batch_spectrum, int(seq.size(0))

    @torch.no_grad()
    def update_global_spectrum(
        self,
        batch_spectrum: Tensor,
        sample_count: int,
    ) -> float:
        """
        Update C with an exact sample-weighted online mean.

        Suppose C has already observed n samples, while the current
        batch contains m samples:

            C_new
                = C_old
                + m / (n + m) * (C_batch - C_old)

        With batch size 1:

            update coefficient = 1 / seen_batches
        """

        if sample_count <= 0:
            raise ValueError(
                "sample_count must be positive."
            )

        if (
            tuple(batch_spectrum.shape)
            != tuple(self.global_spectrum.shape)
        ):
            raise ValueError(
                "Spectrum shape mismatch: "
                f"expected "
                f"{tuple(self.global_spectrum.shape)}, "
                f"got {tuple(batch_spectrum.shape)}."
            )

        batch_spectrum = batch_spectrum.detach().to(
            device=self.global_spectrum.device,
            dtype=self.global_spectrum.dtype,
        )

        old_count = int(
            self.spectrum_seen_samples.item()
        )

        new_count = old_count + sample_count

        update_weight = (
            float(sample_count) / float(new_count)
        )

        delta = (
            batch_spectrum
            - self.global_spectrum
        )

        applied_update = (
            update_weight * delta
        )

        self.global_spectrum.add_(
            applied_update
        )

        self.spectrum_seen_samples.fill_(
            new_count
        )

        update_norm = applied_update.norm()

        self.last_spectrum_update_norm.copy_(
            update_norm.to(
                dtype=(
                    self.last_spectrum_update_norm.dtype
                )
            )
        )

        return float(update_norm.item())

    def _spectral_context(
        self,
        hidden: Tensor,
    ) -> Tensor:
        """
        Project C to the PatchTST hidden representation.

        Input:
            global_spectrum:
                (1, N, F)

        Output:
            context:
                (B, N, D, P)
        """

        # (1, N, F) -> (1, N, D)
        context = self.spectral_projector(
            self.global_spectrum
        )

        context = self.spectral_context_norm(
            context
        )

        context = self.spectral_context_dropout(
            context
        )

        # (1, N, D) -> (1, N, D, 1)
        context = context.unsqueeze(-1)

        # (1, N, D, 1) -> (B, N, D, P)
        context = context.expand(
            hidden.size(0),
            -1,
            -1,
            hidden.size(-1),
        )

        gate = torch.sigmoid(
            self.spectral_gate_logit
        )

        # Before the first batch updates C, disable context injection.
        ready = (
            self.spectrum_seen_samples > 0
        ).to(
            dtype=hidden.dtype,
            device=hidden.device,
        )

        return ready * gate * context

    def _encode_with_single_backbone(
        self,
        seq,
        model,
    ):
        """
        Original PatchTST encoding followed by spectral injection.
        """

        hidden = super()._encode_with_single_backbone(
            seq,
            model,
        )

        spectral_context = self._spectral_context(
            hidden
        )

        return hidden + spectral_context

    def get_spectral_diagnostics(self):
        """
        Statistics used for logging.
        """

        return {
            "seen_samples": int(
                self.spectrum_seen_samples.item()
            ),
            "last_update_norm": float(
                self.last_spectrum_update_norm.item()
            ),
            "context_gate": float(
                torch.sigmoid(
                    self.spectral_gate_logit
                ).detach().item()
            ),
            "spectrum_norm": float(
                self.global_spectrum.norm().item()
            ),
        }