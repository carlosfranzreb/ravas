# conv.py (stateless streaming rewrite)
# Copyright (c) Kyutai, all rights reserved.
# Licensed under the license in the root directory of this source tree.

import typing as tp
import warnings

import torch
from torch import nn, Tensor

from .conv_utils import NormConv1d, NormConvTranspose1d


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

    def _init_streaming_state(self) -> list[Tensor]:
        param = next(iter(self.parameters()))
        batch_size = 1
        prev = torch.zeros(
            batch_size,
            self.conv.conv.in_channels,
            self._effective_kernel_size - self._stride,
            device=param.device,
            dtype=param.dtype,
        )
        return [prev]

    @property
    def n_states(self) -> int:
        return 1

    def forward(self, x: Tensor, prev: Tensor) -> tuple[Tensor, Tensor]:
        """Input length must be multiple of stride."""

        n_feats_prev = prev.shape[-1]
        if n_feats_prev > 0:
            x = torch.cat([prev, x], dim=-1)
            y = self.conv(x)
            new_prev = x[..., -n_feats_prev:]
        else:
            y = self.conv(x)
            new_prev = prev

        return y, new_prev


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

    def _init_streaming_state(self) -> list[Tensor]:
        param = next(iter(self.parameters()))
        batch_size = 1
        K, S = self._kernel_size, self._stride
        partial = torch.zeros(
            batch_size,
            self.convtr.convtr.out_channels,
            K - S,
            device=param.device,
            dtype=param.dtype,
        )
        return [partial]

    @property
    def n_states(self) -> int:
        return 1

    def forward(self, x: Tensor, partial: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the convolution, overlap-add the state, and update the state.
        """
        y = self.convtr(x)

        n_feats_partial = partial.shape[-1]
        if n_feats_partial > 0:
            y[..., :n_feats_partial] += partial
            for_partial = y[..., -n_feats_partial:]

            bias = self.convtr.convtr.bias
            if bias is not None:
                for_partial -= bias[:, None]

            new_partial = for_partial
            y = y[..., :-n_feats_partial]
        else:
            new_partial = partial

        return y, new_partial
