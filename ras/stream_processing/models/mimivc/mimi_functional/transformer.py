# transformer.py (stateless streaming rewrite)
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
import typing as tp

from einops import rearrange
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from moshi.utils.compile import no_compile
from moshi.utils import quantize
from moshi.utils.quantize import replace_linear_with_qlinear
from moshi.modules.rope import RotaryEmbedding
from moshi.modules.lora import LoRALinear

from .transformer_utils import (
    apply_weights_per_step,
    create_norm_fn,
    LayerScale,
    create_sin_embedding,
)


class StreamingMultiheadAttention(nn.Module):
    """Stateless-style StreamingMultiheadAttention.

    forward(...) -> (out, new_state)
    """

    _fsdp_final = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool,
        context: tp.Optional[int],
        rope: tp.Optional[RotaryEmbedding],
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

        out_dim = 3 * embed_dim
        self.mult = 1

        self.out_projs = nn.ModuleList(
            [
                nn.Linear(embed_dim, embed_dim, bias=False, **factory_kwargs)
                for _ in range(self.mult)
            ]
        )
        self.in_projs = nn.ModuleList(
            [
                nn.Linear(embed_dim, out_dim, bias=False, **factory_kwargs)
                for _ in range(self.mult)
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

    def _init_streaming_state(self, batch_size: int) -> list[Tensor]:
        in_proj = self.in_projs[0]
        device = in_proj.weight.device
        dtype = in_proj.weight.dtype

        # create ring KV cache and offset
        dim_per_head = self.embed_dim // self.num_heads
        kv_cache_capacity = self.context
        kv_cache_cache = torch.zeros(
            (2, batch_size, self.num_heads, kv_cache_capacity, dim_per_head),
            device=device,
            dtype=dtype,
        )
        kv_cache_end_offset = torch.zeros(1, device=device, dtype=torch.long)

        # create other states
        offset = torch.zeros(batch_size, device=device, dtype=torch.long)
        offset_cpu = torch.tensor(0)

        return [
            kv_cache_cache,
            kv_cache_end_offset,
            offset,
            offset_cpu,
        ]

    def _complete_kv(
        self,
        keys: Tensor,
        values: Tensor,
        kv_cache_cache: Tensor,
        kv_cache_end_offset: Tensor,
    ) -> list[Tensor, Tensor, Tensor, Tensor, Tensor]:

        B, H, T, D = keys.shape
        kv_cache_capacity = kv_cache_cache.shape[3]
        exec_mask = torch.ones(B, dtype=torch.bool, device=keys.device)
        indexes = torch.arange(
            T, device=kv_cache_end_offset.device, dtype=kv_cache_end_offset.dtype
        )
        indexes = indexes + kv_cache_end_offset.view(-1, 1)
        indexes = indexes % kv_cache_capacity

        this_indexes = indexes.view(B, 1, T, 1)
        this_indexes = this_indexes.expand(-1, H, T, D)
        kv_cache_cache[0].scatter_(2, this_indexes, keys)
        kv_cache_cache[1].scatter_(2, this_indexes, values)

        keys = kv_cache_cache[0]
        values = kv_cache_cache[1]

        indexes = torch.arange(
            kv_cache_capacity, device=kv_cache_end_offset.device, dtype=torch.long
        )

        last_offset = kv_cache_end_offset.view(-1, 1) + T - 1
        end_index = last_offset % kv_cache_capacity
        delta = indexes - end_index

        positions = torch.where(
            delta <= 0,
            last_offset + delta,
            last_offset + delta - kv_cache_capacity,
        )
        kv_cache_end_offset[:] = torch.where(
            exec_mask, kv_cache_end_offset + T, kv_cache_end_offset
        )

        invalid = indexes >= kv_cache_end_offset.view(-1, 1)
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)

        return keys, values, positions, kv_cache_cache, kv_cache_end_offset

    def forward(
        self, query: Tensor, state: list[Tensor]
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Stateless-style forward. Inputs:
            query, key, value: [B, T, C] (or for q/k/v after projection)
            state: previous MHA state
        Returns:
            (out: [B, T, C], new_state)
        """
        B, T = query.shape[:2]
        [kv_cache_cache, kv_cache_end_offset, offset, offset_cpu] = state

        projected = apply_weights_per_step(self.in_projs, query, offset_cpu)

        q, k, v = rearrange(
            projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
        )

        if self.rope:
            q, k = self.rope(q, k, offset, time_before_heads=False)

        k, v, pos_k, kv_cache_cache, kv_cache_end_offset = self._complete_kv(
            k, v, kv_cache_cache, kv_cache_end_offset
        )
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
        x = apply_weights_per_step(self.out_projs, x, offset_cpu)

        # update offsets
        new_offset = offset + T
        new_offset_cpu = offset_cpu + T

        return x, [
            kv_cache_cache,
            kv_cache_end_offset,
            new_offset,
            new_offset_cpu,
        ]


class StreamingTransformerLayer(nn.Module):
    """Stateless-style layer."""

    _fsdp_final = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        causal: bool,
        context: int,
        rope: RotaryEmbedding,
        norm: str,
        layer_scale: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        attn_kwargs = {
            "embed_dim": d_model,
            "num_heads": num_heads,
        }

        self.self_attn = StreamingMultiheadAttention(
            causal=causal,
            context=context,
            rope=rope,
            **attn_kwargs,
            **factory_kwargs,
        )
        self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)

        self.activation = F.gelu
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False, **factory_kwargs)

        self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)
        self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)

    def _init_streaming_state(self, batch_size: int) -> tuple[Tensor, list[Tensor]]:
        """
        Returns a list of Tensors, comprising all the states required by the
        transformer layer (the layer's offset_cpu followed by the states of the
        multi-head self-attention.)
        """
        states = [torch.tensor(0)]  # offset_cpu
        states += self.self_attn._init_streaming_state(batch_size)
        return states

    # _ff_block expects to return (tensor, new_layer_state)
    def _ff_block(self, x: Tensor, offset: Tensor) -> tuple[Tensor, Tensor]:
        x_orig = x
        x = self.norm2(x)
        update = self.linear2(self.activation(self.linear1(x)))
        out = x_orig.to(update) + self.layer_scale_2(update)

        return out, offset + out.shape[1]

    def _sa_block(
        self, x: Tensor, mha_state: list[Tensor]
    ) -> tuple[Tensor, list[Tensor]]:
        x_orig = x
        x = self.norm1(x)
        out, new_mha_state = self.self_attn(x, mha_state)
        return x_orig.to(out) + self.layer_scale_1(out), new_mha_state

    def forward(self, x: Tensor, state: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        """
        Returns:
            x_out, new states (offset followed by MHA)
        """
        with ExitStack() as stack:
            if x.device.type != "cuda":
                stack.enter_context(no_compile())

            offset = state[0]
            mha_state = state[1:]
            x, new_mha_state = self._sa_block(x, mha_state)
            x, new_offset = self._ff_block(x, offset)
            new_state = [new_offset, *new_mha_state]

            return x, new_state


class StreamingTransformer(nn.Module):
    """Top-level stateless transformer.
    TODO: consume `transformer` prefix in state_dict to replace ProjectedTransformer with this one.
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
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.rope = RotaryEmbedding(max_period=max_period)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                StreamingTransformerLayer(
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

    def _init_streaming_state(self, batch_size: int) -> tuple[Tensor, list[Tensor]]:
        """Returns the transformer's offsets and the layer states."""
        device = next(self.parameters()).device
        offsets = torch.zeros(batch_size, device=device, dtype=torch.long)
        layer_states = list()
        for layer in self.layers:
            layer_states.append(layer._init_streaming_state(batch_size))

        return offsets, layer_states

    def forward(
        self, x: Tensor, offsets: Tensor, layer_states: list[list[Tensor]]
    ) -> tuple[Tensor, Tensor, list[list[Tensor]]]:
        """
        Inputs:
            x: [B, T, C]
            offsets: optional offsets
            layer_states: list of self-attention layer states (offset + MHA)
        Returns:
            x_out, new_offsets, new_layer_states,
        """
        x = x.transpose(1, 2)
        new_layer_states = list()
        for layer_idx, layer in enumerate(self.layers):
            x, new_layer_state = layer(x, layer_states[layer_idx])
            new_layer_states.append(new_layer_state)

        new_offsets = offsets + x.shape[1]
        x = x.transpose(1, 2).to(x.dtype)

        return (x, new_offsets, new_layer_states)
