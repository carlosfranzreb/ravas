# conv.py (stateless streaming rewrite)
# Copyright (c) Kyutai, all rights reserved.
# Licensed under the license in the root directory of this source tree.

from dataclasses import dataclass
import math
import typing as tp
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from moshi.modules.streaming import State


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
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
):
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(
    x: torch.Tensor,
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


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
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


@dataclass
class _StreamingConv1dState(State):
    previous: torch.Tensor
    first: torch.Tensor

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.previous[:] = torch.where(
            reset_mask.view(-1, 1, 1), torch.zeros_like(self.previous), self.previous
        )
        self.first[:] = torch.where(reset_mask, torch.ones_like(self.first), self.first)


class StreamingConv1d(nn.Module):
    """Stateless-style causal Conv1d with state passed explicitly."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        pad_mode: str = "constant",
    ):
        super().__init__()
        assert pad_mode in ["constant", "replicate"], pad_mode
        assert causal
        if stride > 1 and dilation > 1:
            warnings.warn(
                f"StreamingConv1d with stride={stride}, dilation={dilation} may be unusual."
            )
        self.pad_mode = pad_mode
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    @property
    def _stride(self):
        return self.conv.conv.stride[0]

    @property
    def _kernel_size(self):
        return self.conv.conv.kernel_size[0]

    @property
    def _effective_kernel_size(self):
        d = self.conv.conv.dilation[0]
        return (self._kernel_size - 1) * d + 1

    @property
    def _padding_total(self):
        return self._effective_kernel_size - self._stride

    def _init_streaming_state(self, batch_size: int) -> _StreamingConv1dState:
        kernel = self._effective_kernel_size
        stride = self._stride
        param = next(iter(self.parameters()))
        prev = torch.zeros(
            batch_size,
            self.conv.conv.in_channels,
            kernel - stride,
            device=param.device,
            dtype=param.dtype,
        )
        first = torch.ones(batch_size, device=param.device, dtype=torch.bool)
        return _StreamingConv1dState(batch_size, param.device, prev, first)

    def forward(
        self, x: torch.Tensor, state: _StreamingConv1dState | None = None
    ) -> tuple[torch.Tensor, _StreamingConv1dState]:
        B, C, T = x.shape
        S = self._stride
        assert T % S == 0, "Input length must be multiple of stride"
        if state is None:
            state = self._init_streaming_state(B)
        TP = state.previous.shape[-1]
        if TP and self.pad_mode == "replicate":
            init = x[..., :1]
            state.previous[:] = torch.where(
                state.first.view(-1, 1, 1) & state.exec_mask.view(-1, 1, 1),
                init,
                state.previous,
            )
        if TP:
            x = torch.cat([state.previous, x], dim=-1)
        y = self.conv(x)
        if TP:
            state.previous[:] = torch.where(
                state.exec_mask.view(-1, 1, 1), x[..., -TP:], state.previous
            )
            if self.pad_mode == "replicate":
                state.first = torch.where(
                    state.exec_mask, torch.zeros_like(state.first), state.first
                )
        return y, state


@dataclass
class _StreamingConvTr1dState(State):
    partial: torch.Tensor

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.partial[:] = torch.where(
            reset_mask.view(-1, 1, 1), torch.zeros_like(self.partial), self.partial
        )


class StreamingConvTranspose1d(nn.Module):
    """Stateless-style causal ConvTranspose1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: tp.Dict[str, tp.Any] = {},
    ):
        super().__init__()
        assert trim_right_ratio == 1.0
        assert causal
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    @property
    def _stride(self):
        return self.convtr.convtr.stride[0]

    @property
    def _kernel_size(self):
        return self.convtr.convtr.kernel_size[0]

    def _init_streaming_state(self, batch_size: int) -> _StreamingConvTr1dState:
        param = next(iter(self.parameters()))
        K, S = self._kernel_size, self._stride
        partial = torch.zeros(
            batch_size,
            self.convtr.convtr.out_channels,
            K - S,
            device=param.device,
            dtype=param.dtype,
        )
        return _StreamingConvTr1dState(batch_size, param.device, partial)

    def forward(
        self, x: torch.Tensor, state: _StreamingConvTr1dState | None = None
    ) -> tuple[torch.Tensor, _StreamingConvTr1dState]:
        B, C, T = x.shape
        K, S = self._kernel_size, self._stride
        y = self.convtr(x)
        if state is None:
            y = unpad1d(y, (0, K - S))
            state = self._init_streaming_state(B)
        else:
            PT = state.partial.shape[-1]
            if PT > 0:
                y[..., :PT] += state.partial
                bias = self.convtr.convtr.bias
                for_partial = y[..., -PT:]
                if bias is not None:
                    for_partial -= bias[:, None]
                state.partial[:] = torch.where(
                    state.exec_mask.view(-1, 1, 1), for_partial, state.partial
                )
                y = y[..., :-PT]
        return y, state
