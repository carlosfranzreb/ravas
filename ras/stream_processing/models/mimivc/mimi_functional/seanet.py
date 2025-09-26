# seanet.py (stateless streaming rewrite)
import typing as tp
import numpy as np
from torch import nn, Tensor

from .conv import StreamingConv1d, StreamingConvTranspose1d


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model, functional/state-passing."""

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        act = getattr(nn, activation)
        hidden = dim // compress
        layers: tp.List[nn.Module] = []
        for i, (k, d) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            layers += [
                act(**activation_params),
                StreamingConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=k,
                    dilation=d,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.ModuleList(layers)

    def _init_streaming_state(self, batch_size: int) -> list[Tensor]:
        states = list()
        for layer in self.block:
            if isinstance(layer, (StreamingConv1d, StreamingConvTranspose1d)):
                states.append(layer._init_streaming_state(batch_size))
            else:
                states.append(None)

        return states

    def forward(self, x, states: tp.List[tp.Any]):
        new_states = list()
        y = x
        for layer_idx, layer in enumerate(self.block):
            if isinstance(layer, (StreamingConv1d, StreamingConvTranspose1d)):
                y, new_state = layer(y, states[layer_idx])
                new_states.append(new_state)
            else:
                y = layer(y)
                new_states.append(None)

        y += x
        return y, new_states


class SEANetModel(nn.Module):
    """SEANet encoder with explicit state passing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def _init_streaming_state(self, batch_size: int) -> list[Tensor]:
        states = list()
        for layer in self.model:
            if isinstance(
                layer, (SEANetResnetBlock, StreamingConv1d, StreamingConvTranspose1d)
            ):
                states.append(layer._init_streaming_state(batch_size))
            else:
                states.append(None)

        return states

    def forward(self, x, states: tp.List[tp.Any]):
        new_states = list()
        y = x
        for layer, st in zip(self.model, states):
            if isinstance(layer, SEANetResnetBlock):
                y, sub_states = layer(y, st)
                new_states.append(sub_states)
            elif isinstance(layer, (StreamingConv1d, StreamingConvTranspose1d)):
                y, new_st = layer(y, st)
                new_states.append(new_st)
            else:
                y = layer(y)
                new_states.append(None)

        return y, new_states


class SEANetEncoder(SEANetModel):
    """SEANet encoder with explicit state passing."""

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        disable_norm_outer_blocks: int = 0,
        mask_fn: tp.Optional[nn.Module] = None,
        mask_position: tp.Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(self.ratios))
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        act = getattr(nn, activation)

        modules: tp.List[nn.Module] = []
        mult = 1
        modules.append(
            StreamingConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm="none" if disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        )
        if mask_fn is not None and mask_position == 0:
            modules.append(mask_fn)
        for i, ratio in enumerate(self.ratios):
            block_norm = "none" if disable_norm_outer_blocks >= i + 2 else norm
            for j in range(n_residual_layers):
                modules.append(
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=block_norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                )
            modules += [
                act(**activation_params),
                StreamingConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2
            if mask_fn is not None and mask_position == i + 1:
                modules.append(mask_fn)

        modules += [
            act(**activation_params),
            StreamingConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm=(
                    "none"
                    if disable_norm_outer_blocks == len(self.ratios) + 2
                    else norm
                ),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        self.model = nn.ModuleList(modules)


class SEANetDecoder(SEANetModel):
    """SEANet decoder with explicit state passing."""

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        disable_norm_outer_blocks: int = 0,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = int(np.prod(ratios))
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        act = getattr(nn, activation)

        modules: tp.List[nn.Module] = []
        mult = int(2 ** len(ratios))
        modules.append(
            StreamingConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=("none" if disable_norm_outer_blocks == len(ratios) + 2 else norm),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        )
        for i, ratio in enumerate(ratios):
            block_norm = (
                "none"
                if disable_norm_outer_blocks >= len(ratios) + 2 - (i + 1)
                else norm
            )
            modules += [
                act(**activation_params),
                StreamingConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]
            for j in range(n_residual_layers):
                modules.append(
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=block_norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                )
            mult //= 2

        modules += [
            act(**activation_params),
            StreamingConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm="none" if disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            modules.append(final_act(**(final_activation_params or {})))

        self.model = nn.ModuleList(modules)
