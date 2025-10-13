import typing as tp

import torch
from torch import nn, Tensor

from moshi.utils.compile import torch_compile_lazy


class LayerNormF32(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        x_f32 = input.float()
        out_f32 = super().forward(x_f32)
        return out_f32.to(input.dtype)


@torch_compile_lazy
def _rms_norm(
    x: Tensor,
    alpha: Tensor,
    dtype: tp.Optional[torch.dtype],
    eps: float,
):
    assert x.dim() == 3, f"RMSNorm expects 3D inputs but got {x.shape}"
    x_dtype = x.dtype
    if dtype is not None:
        x = x.to(dtype)
    var = eps + torch.mean(x**2, dim=2, keepdim=True)
    y = (x * (alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
    return y


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: tp.Optional[torch.dtype] = None,
        device=None,
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.alpha = nn.Parameter(
            torch.full((1, 1, dim), 1.0, requires_grad=True, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor):
        return _rms_norm(x, self.alpha, self.dtype, self.eps)


class LayerScale(nn.Module):
    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        channel_last: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full(
                (channels,), init, requires_grad=True, device=device, dtype=dtype
            )
        )

    def forward(self, x: Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    elif norm_type == "layer_norm_f32":
        kwargs.pop("dtype", None)
        return LayerNormF32(dim, eps=1e-8, **kwargs)
    elif norm_type in {"rms_norm"}:
        return RMSNorm(dim, eps=1e-5, **kwargs)
    elif norm_type in {"rms_norm_f32"}:
        kwargs.pop("dtype", None)
        return RMSNorm(dim, eps=1e-8, dtype=torch.float, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def create_sin_embedding(
    positions: Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def apply_weights_per_step(
    modules: nn.ModuleList,
    x: Tensor,
    offset: int | None,
) -> Tensor:
    if len(modules) == 1:
        return modules[0](x)

    assert offset is not None, "Out of sync execution with weights per step."

    ys: list[Tensor] = []
    B, T, C = x.shape
    for t in range(T):
        module_index = t + offset
        y = modules[module_index](x[:, t : t + 1])
        ys.append(y)
    out = torch.cat(ys, 1)
    return out
