# transformer.py (stateless streaming rewrite)
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from dataclasses import dataclass
import typing as tp
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F
from moshi.utils.compile import no_compile, torch_compile_lazy
from moshi.utils import quantize
from moshi.utils.quantize import replace_linear_with_qlinear
from moshi.modules.gating import make_gating
from moshi.modules.rope import RotaryEmbedding
from moshi.modules.streaming import State
from moshi.modules.lora import LoRALinear
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class LayerNormF32(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_f32 = input.float()
        out_f32 = super().forward(x_f32)
        return out_f32.to(input.dtype)


@torch_compile_lazy
def _rms_norm(
    x: torch.Tensor,
    alpha: torch.Tensor,
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

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
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
    positions: torch.Tensor,
    dim: int,
    max_period: float = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def set_attention_context(model: nn.Module, context: tp.Optional[int] = None) -> None:
    for module in model.modules():
        if isinstance(module, StreamingMultiheadAttention):
            module.context = context


class KVCacheResult(tp.NamedTuple):
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor

    @staticmethod
    def from_kv(keys: torch.Tensor, values: torch.Tensor) -> "KVCacheResult":
        B, H, T, D = keys.shape
        assert tuple(values.shape[:-1]) == (B, H, T)
        positions = torch.arange(T, device=keys.device, dtype=torch.long)
        return KVCacheResult(keys, values, positions.expand(B, -1))


class RingKVCache:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int,
        respect_exec_mask: bool = True,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.capacity = capacity
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            device=device,
            dtype=dtype,
        )
        self.respect_exec_mask = respect_exec_mask
        if self.respect_exec_mask:
            self.end_offset = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            self.end_offset = torch.zeros(1, device=device, dtype=torch.long)

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.end_offset[:] = torch.where(
            reset_mask,
            torch.zeros_like(self.end_offset),
            self.end_offset,
        )

    def complete(
        self, k: torch.Tensor, v: torch.Tensor, exec_mask: torch.Tensor
    ) -> KVCacheResult:
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        B, H, T, D = k.shape
        assert T > 0
        indexes = torch.arange(
            T, device=self.end_offset.device, dtype=self.end_offset.dtype
        )
        indexes = indexes + self.end_offset.view(-1, 1)
        indexes = indexes % self.capacity
        if self.respect_exec_mask:
            this_indexes = indexes.view(B, 1, T, 1)
            this_indexes = this_indexes.expand(-1, H, T, D)
            self.cache[0].scatter_(2, this_indexes, k)
            self.cache[1].scatter_(2, this_indexes, v)
        else:
            self.cache[0].index_copy_(2, indexes[0], k)
            self.cache[1].index_copy_(2, indexes[0], v)

        keys = self.cache[0]
        values = self.cache[1]

        indexes = torch.arange(
            self.capacity, device=self.end_offset.device, dtype=torch.long
        )

        last_offset = self.end_offset.view(-1, 1) + T - 1
        end_index = last_offset % self.capacity
        delta = indexes - end_index

        positions = torch.where(
            delta <= 0,
            last_offset + delta,
            last_offset + delta - self.capacity,
        )
        if self.respect_exec_mask:
            self.end_offset[:] = torch.where(
                exec_mask, self.end_offset + T, self.end_offset
            )
        else:
            self.end_offset.add_(T)
        invalid = indexes >= self.end_offset.view(-1, 1)
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)

        return KVCacheResult(keys, values, positions)


def apply_weights_per_step(
    modules: nn.ModuleList,
    schedule: list[int] | None,
    x: torch.Tensor,
    offset: int | None,
) -> torch.Tensor:
    if len(modules) == 1:
        return modules[0](x)

    assert offset is not None, "Out of sync execution with weights per step."

    ys: list[torch.Tensor] = []
    B, T, C = x.shape
    for t in range(T):
        module_index = t + offset
        if schedule is not None:
            module_index = schedule[module_index]
        y = modules[module_index](x[:, t : t + 1])
        ys.append(y)
    out = torch.cat(ys, 1)
    return out


@dataclass
class _MHAState(State):
    kv_cache: RingKVCache | None
    offset: torch.Tensor
    offset_cpu: int
    k_cross: torch.Tensor | None = None
    v_cross: torch.Tensor | None = None

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.offset[:] = torch.where(
            reset_mask, torch.zeros_like(self.offset), self.offset
        )
        if self.kv_cache is not None:
            self.kv_cache.reset(reset_mask)
        self.offset_cpu = 0


class StreamingMultiheadAttention(nn.Module):
    """Stateless-style StreamingMultiheadAttention.

    forward(...) -> (out, new_state)
    """

    _fsdp_final = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        weights_per_step: int = 0,
        weights_per_step_schedule: list[int] | None = None,
        cross_attention: bool = False,
        cache_cross_attention: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.num_heads = num_heads
        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule
        self.cross_attention = cross_attention
        self.cache_cross_attention = cache_cross_attention
        if cross_attention:
            assert (
                not weights_per_step
            ), "weights_per_step not supported for cross attention."
            assert rope is None, "rope and cross_attention makes no sense."
            assert not causal, "causal and cross attention makes no sense."

        out_dim = 3 * embed_dim
        mult = 1
        if weights_per_step:
            if weights_per_step_schedule:
                assert len(weights_per_step_schedule) == weights_per_step
                mult = max(weights_per_step_schedule) + 1
            else:
                mult = weights_per_step
        self.mult = mult

        self.out_projs = nn.ModuleList(
            [
                nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)
                for _ in range(mult)
            ]
        )
        self.in_projs = nn.ModuleList(
            [
                nn.Linear(embed_dim, out_dim, bias=False, **factory_kwargs)
                for _ in range(mult)
            ]
        )

        self._register_load_state_dict_pre_hook(
            StreamingMultiheadAttention._load_hook, with_module=True
        )

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        mappings = {
            "in_proj_weight": "in_projs.{i}.weight",
            "in_proj.weight": "in_projs.{i}.weight",
            "in_proj.lora_A.weight": "in_projs.{i}.lora_A.weight",
            "in_proj.lora_B.weight": "in_projs.{i}.lora_B.weight",
            "out_proj.weight": "out_projs.{i}.weight",
            "out_proj.lora_A.weight": "out_projs.{i}.lora_A.weight",
            "out_proj.lora_B.weight": "out_projs.{i}.lora_B.weight",
        }

        mult = module.mult
        for suffix in ["", "_scb"]:
            for source, target in mappings.items():
                this_source = prefix + source + suffix
                if this_source in state_dict:
                    weight = state_dict[this_source]
                    _, *OD = weight.shape
                    weight = weight.view(mult, -1, *OD)
                    for i in range(mult):
                        this_target = prefix + target.format(i=i) + suffix
                        state_dict[this_target] = weight[i]
                    state_dict.pop(this_source)

    def _init_streaming_state(self, batch_size: int) -> _MHAState:
        in_proj = self.in_projs[0]
        if isinstance(in_proj, LoRALinear):
            device = in_proj.lora_A.weight.device
            dtype = in_proj.lora_A.weight.dtype
        elif isinstance(in_proj, nn.Linear):
            device = in_proj.weight.device
            dtype = in_proj.weight.dtype
        elif isinstance(in_proj, quantize.QLinear):
            device = in_proj.weight.device
            dtype = torch.float16
        else:
            raise RuntimeError(f"Unknown type {type(in_proj)} for linear.")

        dim_per_head = self.embed_dim // self.num_heads
        if self.cross_attention:
            kv_cache = None
        else:
            if self.context is None:
                if self.weights_per_step:
                    capacity = self.weights_per_step
                else:
                    raise RuntimeError(
                        "Cannot create a streaming KVCache without a context to estimate capacity."
                    )
            else:
                capacity = self.context

            kv_cache = RingKVCache(
                batch_size,
                self.num_heads,
                dim_per_head,
                capacity,
                respect_exec_mask=not self.weights_per_step,
                device=device,
                dtype=dtype,
            )
        return _MHAState(
            batch_size,
            device,
            kv_cache,
            offset=torch.zeros(batch_size, device=device, dtype=torch.long),
            offset_cpu=0,
        )

    def _complete_kv(self, k, v, state: _MHAState | None) -> KVCacheResult:
        if state is None or state.kv_cache is None:
            return KVCacheResult.from_kv(k, v)
        else:
            return state.kv_cache.complete(k, v, state.exec_mask)

    def _compute_cross_attention(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.cross_attention
        in_proj = self.in_projs[0]
        assert in_proj.bias is None
        assert isinstance(in_proj, nn.Linear)
        dim = in_proj.weight.shape[0] // 3
        kv = nn.functional.linear(key, in_proj.weight[dim:])
        k, v = rearrange(kv, "b t (p h d) -> p b h t d", p=2, h=self.num_heads)
        return k, v

    def update_cross_attention_src(
        self, cross_attention_src: torch.Tensor, state: _MHAState | None
    ) -> _MHAState:
        """Return updated state with cached cross-attention K/V computed and stored."""
        if state is None:
            # initialize ephemeral state for shape inference; caller should use _init_streaming_state instead
            raise RuntimeError(
                "update_cross_attention_src requires a valid _MHAState (call _init_streaming_state)."
            )
        assert self.cross_attention
        k, v = self._compute_cross_attention(cross_attention_src, cross_attention_src)
        if state.k_cross is None:
            state.k_cross = k
            state.v_cross = v
        else:
            assert state.v_cross is not None
            state.k_cross[:] = k
            state.v_cross[:] = v
        return state

    def _get_cross_attention(
        self, key: torch.Tensor, value: torch.Tensor, state: _MHAState | None
    ):
        if state is not None and state.k_cross is not None:
            assert state.v_cross is not None
            return state.k_cross, state.v_cross, state
        k, v = self._compute_cross_attention(key, value)
        if state is not None and self.cache_cross_attention:
            state.k_cross = k
            state.v_cross = v
        return k, v, state

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        state: _MHAState | None = None,
    ) -> tuple[torch.Tensor, _MHAState]:
        """
        Stateless-style forward. Inputs:
            query, key, value: [B, T, C] (or for q/k/v after projection)
            state: previous _MHAState or None
        Returns:
            (out: [B, T, C], new_state: _MHAState)
        """
        B, T = query.shape[:2]

        if state is None:
            state = self._init_streaming_state(B)

        offset = state.offset
        offset_cpu = state.offset_cpu

        if self.cross_attention:
            assert len(self.in_projs) == 1
            in_proj = self.in_projs[0]
            assert in_proj.bias is None
            assert isinstance(in_proj, nn.Linear)
            dim = in_proj.weight.shape[0] // 3
            q = nn.functional.linear(query, in_proj.weight[:dim])
            q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
            k, v, state = self._get_cross_attention(key, value, state)
        else:
            projected = apply_weights_per_step(
                self.in_projs, self.weights_per_step_schedule, query, offset_cpu
            )

            q, k, v = rearrange(
                projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
            )
        if self.rope:
            q, k = self.rope(q, k, offset, time_before_heads=False)

        k, v, pos_k = self._complete_kv(k, v, state)
        pos_k = pos_k[:, None]
        if self.causal:
            pos_q = offset.view(-1, 1, 1) + torch.arange(
                T, device=q.device, dtype=torch.long
            ).view(-1, 1)
            delta = pos_q - pos_k
            attn_bias = (pos_k >= 0) & (delta >= 0)
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)
            attn_bias = attn_bias[:, None]
        else:
            attn_bias = None
        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = apply_weights_per_step(
            self.out_projs, self.weights_per_step_schedule, x, offset_cpu
        )

        if not self.cross_attention:
            state.offset[:] = torch.where(
                state.exec_mask, state.offset + T, state.offset
            )
            state.offset_cpu += T
        return x, state


@dataclass
class _LayerState(State):
    offset_cpu: int = 0

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.offset_cpu = 0


class StreamingTransformerLayer(nn.Module):
    """Stateless-style layer: forward(x, layer_state, mha_state, cross_state) -> (x_out, new_layer_state, new_mha_state[, new_cross_state])"""

    _fsdp_final = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        norm: str = "layer_norm",
        layer_scale: tp.Optional[float] = None,
        gating: str = "none",
        weights_per_step: int = 0,
        weights_per_step_schedule: list[int] | None = None,
        activation=F.gelu,
        skip_self_attn: bool = False,
        cross_attention: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        attn_kwargs: tp.Dict[str, tp.Any] = {
            "embed_dim": d_model,
            "num_heads": num_heads,
        }
        self.skip_self_attn = skip_self_attn
        if not skip_self_attn:
            self.self_attn: StreamingMultiheadAttention = StreamingMultiheadAttention(
                causal=causal,
                context=context,
                rope=rope,
                weights_per_step=weights_per_step,
                weights_per_step_schedule=weights_per_step_schedule,
                **attn_kwargs,
                **factory_kwargs,
            )
            self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)

        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule
        self.gating: tp.Optional[nn.Module] = None
        self.linear1: tp.Optional[nn.Module] = None
        self.linear2: tp.Optional[nn.Module] = None
        self.activation = activation

        num_weights = 1
        if weights_per_step is not None:
            num_weights = weights_per_step
            if weights_per_step_schedule is not None:
                assert len(weights_per_step_schedule) == weights_per_step
                num_weights = max(weights_per_step_schedule) + 1
        if gating == "none":
            assert (
                not weights_per_step
            ), "weights_per_step without gating not supported for now."
            assert not isinstance(
                dim_feedforward, list
            ), "List dim_feedforward without gating not supported for now."
            self.linear1 = nn.Linear(
                d_model, dim_feedforward, bias=False, **factory_kwargs
            )
            self.linear2 = nn.Linear(
                dim_feedforward, d_model, bias=False, **factory_kwargs
            )
        else:
            self.linear1 = None
            self.linear2 = None
            if weights_per_step:
                if isinstance(dim_feedforward, int):
                    dim_feedforward = [dim_feedforward] * num_weights
                assert isinstance(dim_feedforward, list), dim_feedforward
                self.gating = nn.ModuleList(
                    [
                        make_gating(gating, d_model, dim, **factory_kwargs)
                        for dim in dim_feedforward
                    ]
                )
            else:
                assert isinstance(dim_feedforward, int)
                self.gating = make_gating(
                    gating, d_model, dim_feedforward, **factory_kwargs
                )

        self.cross_attention: StreamingMultiheadAttention | None = None
        if cross_attention:
            self.cross_attention = StreamingMultiheadAttention(
                cross_attention=True, **attn_kwargs, **factory_kwargs
            )
            self.norm_cross = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)

        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
            if cross_attention:
                self.layer_scale_cross = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)
            if cross_attention:
                self.layer_scale_cross = LayerScale(
                    d_model, layer_scale, **factory_kwargs
                )

    def _init_streaming_state(self, batch_size: int) -> _LayerState:
        device = next(iter(self.parameters())).device
        return _LayerState(batch_size, device, offset_cpu=0)

    # _ff_block expects to return (tensor, new_layer_state)
    def _ff_block(
        self, x: torch.Tensor, state: _LayerState | None
    ) -> tuple[torch.Tensor, _LayerState]:
        if state is None:
            # user should initialize with _init_streaming_state
            raise RuntimeError(
                "_ff_block requires a valid _LayerState (call _init_streaming_state)."
            )
        offset = state.offset_cpu
        x_orig = x
        x = self.norm2(x)
        if self.gating is None:
            assert self.linear1 is not None
            assert self.linear2 is not None
            update = self.linear2(self.activation(self.linear1(x)))
        else:
            if self.weights_per_step:
                assert isinstance(self.gating, nn.ModuleList)
                update = apply_weights_per_step(
                    self.gating, self.weights_per_step_schedule, x, offset
                )
            else:
                update = self.gating(x)
        out = x_orig.to(update) + self.layer_scale_2(update)
        # update state
        state.offset_cpu += out.shape[1]
        return out, state

    def _sa_block(
        self, x: torch.Tensor, mha_state: _MHAState | None
    ) -> tuple[torch.Tensor, _MHAState | None]:
        if self.skip_self_attn:
            return x, mha_state
        x_orig = x
        x = self.norm1(x)
        out, new_mha_state = self.self_attn(x, x, x, mha_state)
        return x_orig.to(out) + self.layer_scale_1(out), new_mha_state

    def _cross_attention_block(
        self,
        x: torch.Tensor,
        cross_attention_src: torch.Tensor,
        cross_state: _MHAState | None,
    ) -> tuple[torch.Tensor, _MHAState | None]:
        assert self.cross_attention is not None
        x_orig = x
        x = self.norm_cross(x)
        update, new_cross_state = self.cross_attention(
            x, cross_attention_src, cross_attention_src, cross_state
        )
        return x_orig + self.layer_scale_cross(update), new_cross_state

    def forward(
        self,
        x: torch.Tensor,
        layer_state: _LayerState | None = None,
        mha_state: _MHAState | None = None,
        cross_state: _MHAState | None = None,
        cross_attention_src: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, _LayerState, _MHAState | None, _MHAState | None]:
        """
        Returns:
            x_out, new_layer_state, new_mha_state, new_cross_state
        """
        with ExitStack() as stack:
            if x.device.type != "cuda":
                stack.enter_context(no_compile())
            x, new_mha_state = self._sa_block(x, mha_state)
            new_cross_state = None
            if self.cross_attention is not None:
                assert cross_attention_src is not None
                x, new_cross_state = self._cross_attention_block(
                    x, cross_attention_src, cross_state
                )
            else:
                assert cross_attention_src is None
            x, new_layer_state = self._ff_block(x, layer_state)
            return x, new_layer_state, new_mha_state, new_cross_state


@dataclass
class _TransformerState(State):
    offsets: torch.Tensor

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.offsets[:] = torch.where(
            reset_mask, torch.zeros_like(self.offsets), self.offsets
        )


class StreamingTransformer(nn.Module):
    """Top-level stateless transformer.

    forward(x, transformer_state, layer_states, mha_states, cross_states=None) -> (y, new_transformer_state, new_layer_states, new_mha_states, new_cross_states)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        positional_embedding: str = "sin",
        max_period: float = 10_000,
        positional_scale: float = 1.0,
        betas: tp.Optional[tp.Tuple[float, float]] = None,
        layer_class: tp.Type[StreamingTransformerLayer] = StreamingTransformerLayer,
        quantize: bool = False,
        checkpointing: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.betas = betas

        assert positional_embedding in {"sin", "rope", "sin_rope", "none"}
        self.rope: tp.Optional[RotaryEmbedding] = None
        if self.positional_embedding in {"rope", "sin_rope"}:
            self.rope = RotaryEmbedding(max_period=max_period)

        self.checkpointing = checkpointing

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                layer_class(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    causal=causal,
                    context=context,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                    **kwargs,
                )
            )
            if quantize:
                self.layers[-1].to(device=device, dtype=dtype)
                replace_linear_with_qlinear(self.layers[-1])

    def _init_streaming_state(self, batch_size: int) -> _TransformerState:
        device = next(self.parameters()).device
        return _TransformerState(
            batch_size,
            device,
            offsets=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def forward(
        self,
        x: torch.Tensor,
        transformer_state: _TransformerState | None = None,
        layer_states: list[_LayerState] | None = None,
        mha_states: list[_MHAState] | None = None,
        cross_states: list[_MHAState] | None = None,
        cross_attention_srcs: list[torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        _TransformerState,
        list[_LayerState],
        list[_MHAState],
        list[_MHAState] | None,
    ]:
        """
        Inputs:
            x: [B, T, C]
            transformer_state: optional _TransformerState
            layer_states: list of per-layer _LayerState (len == num_layers)
            mha_states: list of per-layer _MHAState for self-attention (or None)
            cross_states: list of per-layer _MHAState for cross-attention (or None)
            cross_attention_srcs: list of cross-attention tensors (or None)
        Returns:
            x_out, new_transformer_state, new_layer_states, new_mha_states, new_cross_states
        """
        B, T, C = x.shape
        dtype_input = x.dtype

        if transformer_state is None:
            transformer_state = self._init_streaming_state(B)

        offsets = transformer_state.offsets

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offsets.view(-1, 1, 1)
            pos_emb = create_sin_embedding(
                positions, C, max_period=self.max_period, dtype=x.dtype
            )
            x = x + self.positional_scale * pos_emb

        num_layers = len(self.layers)
        if layer_states is None:
            layer_states = [
                self.layers[i]._init_streaming_state(B) for i in range(num_layers)
            ]
        if mha_states is None:
            mha_states = []
            for i in range(num_layers):
                layer = self.layers[i]
                if hasattr(layer, "self_attn"):
                    mha_states.append(layer.self_attn._init_streaming_state(B))
                else:
                    mha_states.append(None)
        if cross_states is None:
            cross_states = [None] * num_layers

        new_layer_states: list[_LayerState] = []
        new_mha_states: list[_MHAState] = []
        new_cross_states: list[_MHAState] | None = (
            [] if cross_states is not None else None
        )

        for i, layer in enumerate(self.layers):
            l_state = layer_states[i] if layer_states is not None else None
            m_state = mha_states[i] if mha_states is not None else None
            c_state = cross_states[i] if cross_states is not None else None
            cross_src = None
            if cross_attention_srcs is not None:
                cross_src = cross_attention_srcs[i]
            if self.checkpointing:
                # checkpoint wrapper needs a function with only tensors; use a lambda that captures state
                def _run_layer(
                    inp_x,
                    _layer=layer,
                    _l_state=l_state,
                    _m_state=m_state,
                    _c_state=c_state,
                    _cross_src=cross_src,
                ):
                    out, nl, nm, nc = _layer(
                        inp_x, _l_state, _m_state, _c_state, _cross_src
                    )
                    # return tuple-collapsed to a single tensor for checkpoint; we cannot checkpoint non-tensor outputs
                    # therefore avoid checkpointing when using stateful returns in this stateless API.
                    return out

                x = torch_checkpoint(
                    _run_layer,
                    x,
                    use_reentrant=False,
                    determinism_check="none",
                    preserve_rng_state=False,
                )
                # NOTE: checkpointing with stateful outputs is unsupported in this stateless API: states won't be captured.
                # For ONNX/export you should disable checkpointing (checkpointing is primarily for training memory).
                # We'll proceed without updating per-layer states in that code path.
                # To be safe, re-run layer without checkpoint to update states:
                out, nl_state, nm_state, nc_state = layer(
                    x, l_state, m_state, c_state, cross_src
                )
                x = out
                new_layer_states.append(nl_state)
                new_mha_states.append(nm_state)
                if new_cross_states is not None:
                    new_cross_states.append(nc_state)
            else:
                out, nl_state, nm_state, nc_state = layer(
                    x, l_state, m_state, c_state, cross_src
                )
                x = out
                new_layer_states.append(nl_state)
                new_mha_states.append(nm_state)
                if new_cross_states is not None:
                    new_cross_states.append(nc_state)

        # update transformer offsets for the batch elements that are active
        transformer_state.offsets[:] = torch.where(
            transformer_state.exec_mask,
            transformer_state.offsets + T,
            transformer_state.offsets,
        )
        return (
            x.to(dtype_input),
            transformer_state,
            new_layer_states,
            new_mha_states,
            new_cross_states,
        )


class ProjectedTransformer(nn.Module):
    """Transformer with optional projections, stateless-style."""

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tp.Tuple[int, ...],
        d_model: int,
        *,
        conv_layout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.transformer = StreamingTransformer(d_model=d_model, **kwargs)
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.conv_layout = conv_layout
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(
                    nn.Linear(d_model, output_dimension, bias=False)
                )

    def forward(
        self,
        x,
        transformer_state: _TransformerState | None = None,
        layer_states: list[_LayerState] | None = None,
        mha_states: list[_MHAState] | None = None,
        cross_states: list[_MHAState] | None = None,
        cross_attention_srcs: list[torch.Tensor] | None = None,
    ):
        """
        Returns:
            ys (list of outputs), new_transformer_state, new_layer_states, new_mha_states, new_cross_states
        """
        if self.conv_layout:
            x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z, new_transformer_state, new_layer_states, new_mha_states, new_cross_states = (
            self.transformer(
                x,
                transformer_state,
                layer_states,
                mha_states,
                cross_states,
                cross_attention_srcs,
            )
        )
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            if self.conv_layout:
                y = y.transpose(1, 2)
            ys.append(y)
        return (
            ys,
            new_transformer_state,
            new_layer_states,
            new_mha_states,
            new_cross_states,
        )
