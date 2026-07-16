from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Hidden = Union[torch.Tensor, Sequence[torch.Tensor]]


class FrequencyMasking:
    """
    对时间序列执行频域 Mask 增强。

    输入：
        x: (B, T, N, C)

    流程：
        1. 沿时间维 T 执行 RFFT；
        2. 随机将部分频率分量置零；
        3. 执行 IRFFT 回到时域。

    输出：
        augmented: (B, T, N, C)
    """

    def __init__(
        self,
        mask_ratio: float = 0.2,
        preserve_dc: bool = True,
    ):
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(
                f"mask_ratio must be in [0, 1), got {mask_ratio}."
            )

        self.mask_ratio = float(mask_ratio)
        self.preserve_dc = bool(preserve_dc)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected x as (B,T,N,C), got {tuple(x.shape)}."
            )

        original_dtype = x.dtype

        # torch.fft 对 float32 更稳定。
        x_float = x.float()

        # 沿时间维执行实数 FFT。
        # (B,T,N,C) -> (B,F,N,C)
        spectrum = torch.fft.rfft(
            x_float,
            dim=1,
        )

        # 每个样本、节点、频率独立采样 Mask。
        keep_mask = (
            torch.rand(
                spectrum.shape,
                device=spectrum.device,
                dtype=x_float.dtype,
            )
            >= self.mask_ratio
        )

        # 默认保留直流分量，避免整体均值发生过强漂移。
        if self.preserve_dc:
            keep_mask[:, 0, ...] = True

        masked_spectrum = spectrum * keep_mask

        # 回到时域，并显式指定原始时间长度。
        augmented = torch.fft.irfft(
            masked_spectrum,
            n=x.size(1),
            dim=1,
        )

        return augmented.to(original_dtype)


def canonicalize_hidden_for_contrastive(
    hidden: Hidden,
    model_name: str,
    node_num: int,
    hidden_dim: int,
) -> torch.Tensor:
    """
    将不同 backbone 的 encoder 输出统一为：

        (B, Tz, Nz, D)

    各模型对应关系：

    Autoformer / FEDformer:
        (B,T,D) -> (B,T,1,D)

    iTransformer:
        (B,N+F,D) -> 删除时间特征 token -> (B,1,N,D)

    PatchTST:
        (B,N,D,P) -> (B,P,N,D)

    Crossformer:
        list[(B,N,S_i,D)]
            -> list[(B,S_i,N,D)]
            -> (B,sum(S_i),N,D)

    其他已经是四维表示的模型：
        尽可能识别并转换为 (B,T,N,D)
    """

    name = model_name.lower()

    # ============================================================
    # Crossformer：encoder 输出是多尺度张量列表
    # ============================================================
    if isinstance(hidden, (list, tuple)):
        chunks = []

        for index, h in enumerate(hidden):
            if h.dim() != 4:
                raise ValueError(
                    f"Hidden scale {index} must have shape "
                    f"(B,N,S,D), got {tuple(h.shape)}."
                )

            if h.size(1) != node_num:
                raise ValueError(
                    f"Hidden scale {index} has unexpected node dimension. "
                    f"Expected node_num={node_num}, got {tuple(h.shape)}."
                )

            if h.size(-1) != hidden_dim:
                raise ValueError(
                    f"Hidden scale {index} has unexpected hidden dimension. "
                    f"Expected hidden_dim={hidden_dim}, "
                    f"got {tuple(h.shape)}."
                )

            # (B,N,S,D) -> (B,S,N,D)
            h = h.permute(
                0,
                2,
                1,
                3,
            ).contiguous()

            chunks.append(h)

        # 沿 token/时间尺度维拼接。
        # (B,sum(S_i),N,D)
        return torch.cat(
            chunks,
            dim=1,
        )

    if not isinstance(hidden, torch.Tensor):
        raise TypeError(
            f"Unsupported hidden type: {type(hidden)}."
        )

    # ============================================================
    # iTransformer
    #
    # encoder 输出：
    #     (B,N+F,D)
    #
    # 其中前 N 个 token 是变量 token，
    # 后 F 个 token 是时间特征 token。
    # ============================================================
    if name == "itransformer":
        if hidden.dim() != 3:
            raise ValueError(
                f"iTransformer hidden must be 3D, "
                f"got {tuple(hidden.shape)}."
            )

        if hidden.size(-1) != hidden_dim:
            raise ValueError(
                f"iTransformer hidden dimension mismatch. "
                f"Expected hidden_dim={hidden_dim}, "
                f"got {tuple(hidden.shape)}."
            )

        if hidden.size(1) < node_num:
            raise ValueError(
                f"iTransformer has fewer tokens than node_num={node_num}. "
                f"Hidden shape: {tuple(hidden.shape)}."
            )

        # 仅保留前 N 个变量 token。
        hidden = hidden[:, :node_num, :]

        # (B,N,D) -> (B,1,N,D)
        return hidden.unsqueeze(1).contiguous()

    # ============================================================
    # PatchTST
    #
    # encoder 输出：
    #     (B,N,D,P)
    #
    # 将 patch 维视为时间 token 维。
    # ============================================================
    if name == "patchtst":
        if hidden.dim() != 4:
            raise ValueError(
                f"PatchTST hidden must have shape (B,N,D,P), "
                f"got {tuple(hidden.shape)}."
            )

        if hidden.size(1) != node_num:
            raise ValueError(
                f"PatchTST node dimension mismatch. "
                f"Expected node_num={node_num}, "
                f"got {tuple(hidden.shape)}."
            )

        if hidden.size(2) != hidden_dim:
            raise ValueError(
                f"PatchTST hidden dimension mismatch. "
                f"Expected hidden_dim={hidden_dim}, "
                f"got {tuple(hidden.shape)}."
            )

        # (B,N,D,P) -> (B,P,N,D)
        return hidden.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

    # ============================================================
    # Autoformer / FEDformer
    #
    # encoder 输出：
    #     (B,T,D)
    #
    # 这类模型已经在 embedding 阶段融合了所有变量，
    # 因此没有显式节点维，只能设置 Nz=1。
    # ============================================================
    if hidden.dim() == 3:
        if hidden.size(-1) != hidden_dim:
            raise ValueError(
                f"Hidden dimension mismatch. "
                f"Expected hidden_dim={hidden_dim}, "
                f"got {tuple(hidden.shape)}."
            )

        # (B,T,D) -> (B,T,1,D)
        return hidden.unsqueeze(2)

    # ============================================================
    # 其他四维 hidden
    # ============================================================
    if hidden.dim() == 4:
        if hidden.size(-1) != hidden_dim:
            raise ValueError(
                f"Hidden dimension mismatch. "
                f"Expected hidden_dim={hidden_dim}, "
                f"got {tuple(hidden.shape)}."
            )

        # 已经是 (B,T,N,D)
        if hidden.size(2) == node_num:
            return hidden

        # 如果是 (B,N,T,D)，则转换为 (B,T,N,D)
        if hidden.size(1) == node_num:
            return hidden.permute(
                0,
                2,
                1,
                3,
            ).contiguous()

    raise ValueError(
        f"Cannot canonicalize hidden shape {tuple(hidden.shape)} "
        f"for model={model_name}, node_num={node_num}, "
        f"hidden_dim={hidden_dim}."
    )


class LinearProjectionHead(nn.Module):
    """
    对比学习使用的轻量 Linear projection head。

    输入：
        z: (B,T,N,D)

    输出：
        p: (B,T,N,D)

    该模块只在对比预训练阶段使用，
    下游微调时不会加载。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.projection = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if z.dim() != 4:
            raise ValueError(
                f"Projection input must have shape (B,T,N,D), "
                f"got {tuple(z.shape)}."
            )

        if z.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Projection hidden dimension mismatch. "
                f"Expected {self.hidden_dim}, got {z.size(-1)}."
            )

        # nn.Linear 默认作用于最后一个维度。
        return self.projection(z)


def temporal_mean_pool_and_flatten(
    z: torch.Tensor,
) -> torch.Tensor:
    """
    沿 token/时间维执行 mean pooling。

    输入：
        z: (B,T,N,D)

    过程：
        (B,T,N,D)
            -> mean(dim=1)
            -> (B,N,D)
            -> reshape
            -> (B*N,D)

    输出：
        pooled: (B*N,D)
    """

    if z.dim() != 4:
        raise ValueError(
            f"Expected z as (B,T,N,D), got {tuple(z.shape)}."
        )

    # 沿时间/token 维平均。
    pooled = z.mean(dim=1)

    # (B,N,D) -> (B*N,D)
    pooled = pooled.reshape(
        -1,
        pooled.size(-1),
    )

    return pooled


class SymmetricInfoNCELoss(nn.Module):
    """
    对称的 batch 内 InfoNCE。

    输入：
        p1: (M,D)
        p2: (M,D)

    其中：
        M = B*N

    p1[i] 和 p2[i] 是正样本对；
    p1[i] 和 p2[j], i != j 是负样本对。
    """

    def __init__(
        self,
        temperature: float = 0.2,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {temperature}."
            )

        self.temperature = float(temperature)

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
    ) -> torch.Tensor:
        if p1.dim() != 2 or p2.dim() != 2:
            raise ValueError(
                f"InfoNCE expects 2D tensors, "
                f"got p1={tuple(p1.shape)}, "
                f"p2={tuple(p2.shape)}."
            )

        if p1.shape != p2.shape:
            raise ValueError(
                f"p1 and p2 must have the same shape, "
                f"got p1={tuple(p1.shape)}, "
                f"p2={tuple(p2.shape)}."
            )

        sample_num = p1.size(0)

        if sample_num < 2:
            raise ValueError(
                "InfoNCE needs at least two actual pooled samples. "
                "Gradient accumulation does not create additional negatives. "
                "Please increase the real batch_size."
            )

        # L2 归一化，使点积等价于 cosine similarity。
        p1 = F.normalize(
            p1,
            dim=-1,
        )

        p2 = F.normalize(
            p2,
            dim=-1,
        )

        # (M,D) @ (D,M) -> (M,M)
        logits_12 = torch.matmul(
            p1,
            p2.transpose(0, 1),
        )

        logits_12 = logits_12 / self.temperature

        # 第 i 行的正样本位于第 i 列。
        labels = torch.arange(
            sample_num,
            device=logits_12.device,
        )

        # view 1 -> view 2
        loss_12 = F.cross_entropy(
            logits_12,
            labels,
        )

        # view 2 -> view 1
        loss_21 = F.cross_entropy(
            logits_12.transpose(0, 1),
            labels,
        )

        return 0.5 * (
            loss_12 + loss_21
        )