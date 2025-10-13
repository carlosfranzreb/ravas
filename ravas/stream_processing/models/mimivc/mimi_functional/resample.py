# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from torch import nn, Tensor

from .conv import StreamingConv1d, StreamingConvTranspose1d


class ConvDownsample1d(nn.Module):
    """
    Downsampling by some integer amount `stride` using convolutions
    with a kernel size of twice the stride.
    If `causal` is True, the output uses a causal convolution.
    """

    def __init__(
        self,
        stride: int,
        dimension: tp.Optional[int] = None,
        causal: bool = False,
        channel_wise: bool = False,
    ):
        super().__init__()
        self.channel_wise = channel_wise
        groups = 1
        in_channels = dimension
        out_channels = dimension
        if channel_wise:
            groups = dimension

        self.conv = StreamingConv1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            causal=causal,
            groups=groups,
            bias=False,
            pad_mode="replicate",
        )

    def _init_streaming_state(self) -> list[Tensor]:
        return self.conv._init_streaming_state()

    def forward(self, x: Tensor, conv_state: Tensor) -> tuple[Tensor, Tensor]:
        return self.conv(x, conv_state)


class ConvTrUpsample1d(nn.Module):
    """
    Upsample by some integer amount `stride` using transposed convolutions.
    """

    def __init__(
        self,
        stride: int,
        dimension: tp.Optional[int] = None,
        causal: bool = False,
        channel_wise: bool = False,
    ):
        super().__init__()
        self.channel_wise = channel_wise
        groups = 1
        in_channels = dimension
        out_channels = dimension
        if channel_wise:
            groups = dimension

        self.convtr = StreamingConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=2 * stride,
            stride=stride,
            causal=causal,
            groups=groups,
            bias=False,
        )

    def _init_streaming_state(self) -> list[Tensor]:
        return self.convtr._init_streaming_state()

    def forward(self, x: Tensor, conv_state: Tensor) -> tuple[Tensor, Tensor]:
        return self.convtr(x, conv_state)
