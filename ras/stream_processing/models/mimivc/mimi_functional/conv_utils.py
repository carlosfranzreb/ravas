# conv.py (stateless streaming rewrite)
# Copyright (c) Kyutai, all rights reserved.
# Licensed under the license in the root directory of this source tree.

import math
import typing as tp

from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils import weight_norm


CONV_NORMALIZATIONS = frozenset(["none", "weight_norm"])
M = tp.TypeVar("M", bound=nn.Module)


class TransposedLayerNorm(nn.Module):
    """LayerNorm for [B, C, T] inputs."""

    def __init__(self, **kwargs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(**kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x.transpose(1, 2)


def apply_parametrization_norm(module: M, norm: str = "none") -> M:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    else:
        return module


def get_extra_padding_for_conv1d(
    x: Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(
    x: Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: Tensor, paddings: tp.Tuple[int, int]):
    padding_left, padding_right = paddings
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):
    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm_type = norm

    def forward(self, x):
        return self.conv(x)


class NormConvTranspose1d(nn.Module):
    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )
        self.norm_type = norm

    def forward(self, x):
        return self.convtr(x)
