# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Please refer to README.md in the same folder for more information.

import argparse

from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple
import time

import torch
import torch.nn.functional as F

from torch import nn


# ======================== Stock Implementation ========================


def apply_scaling(freqs: torch.Tensor, scale_factor: int):
    # Values obtained from grid search
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    use_scaled: bool = False,
    scale_factor: Optional[int] = None,
):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device="cpu")[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # pyre-ignore
    if use_scaled:
        assert scale_factor is not None
        freqs = apply_scaling(freqs, scale_factor)  # pyre-ignore
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    freqs_cis_ndim = freqs_cis.ndim
    if freqs_cis_ndim == 3:
        # freqs_cis: (seq_len, n_heads, head_dim // 2)
        assert freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1])
        shape = [
            d if (i == ndim - 3 or i == ndim - 2 or i == ndim - 1) else 1
            for i, d in enumerate(x.shape)
        ]
    else:
        # freqs_cis: (seq_len, head_dim // 2)
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_to_k(
    xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xk_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xk_r)

    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xk_out.type_as(xk)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        return xq_out, xk_out


# ======================= HuggingFace Implementation ========================


# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L77
def hf_precompute_freqs_cis(dim: int, end: int, theta: float):
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device="cpu", dtype=torch.int64).float() / dim)
    )
    # pyre-ignore Undefined attribute [16]: `float` has no attribute `device`.
    t = torch.arange(end, device=freqs.device, dtype=torch.int64).type_as(
        freqs  # pyre-ignore
    )
    freqs = torch.outer(t, freqs).float()  # pyre-ignore
    emb = torch.cat((freqs, freqs), dim=-1)
    freqs_cos = torch.cos(emb)
    freqs_sin = torch.sin(emb)
    return freqs_cos, freqs_sin


# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L135
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def hf_apply_rotary_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def hf_apply_rotary_emb_to_k(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the key tensors.

    Args:
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of k. Similarly, if k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `torch.Tensor` the key tensor rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    hidden_dim: Optional[int] = None
    head_dim: Optional[int] = None  # Optional customized head_dim
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    use_kv_cache: bool = False  # Use key/value cache
    use_sdpa_with_kv_cache_op: bool = (
        False  # Use custom sdpa op that updates kv cache in-place
    )
    # Generate logits for all inputs. When it's True, it would take big memory usage
    # at runtime. Enable it only necessary (e.g., use perplexity tools that requires
    # logits for all input tokens.)
    generate_full_logits: bool = False
    enable_dynamic_shape: bool = False  # export model with dynamic shape support
    # A dictionary mapping from pruned token-id to original token-id
    input_prune_map: Optional[Dict[int, int]] = None
    # A dictionary mapping from pruned token-id to original token-id
    output_prune_map: Optional[Dict[int, int]] = None
    use_hf_rope: bool = False  # Use HuggingFace's RoPE implementation
    rope_theta: Optional[float] = (
        None  # The official name to override self.rope_freq_base.
    )
    rope_freq_base: float = 10000.0  # The base frequency for RoPE. Keep it for BC.
    use_scaled_rope: bool = False  # Use scaled RoPE, introduced in llama3.1.
    rope_scale_factor: int = 8
    # Additional Model Metadata needed at runtime
    bos_idx: int = 1
    eos_idx: int = 3
    bos_count: int = -1  # i.e., a single EOS is used as BOS
    eos_count: int = 2

    quantization_args: Optional[dict] = None
    lora_args: Optional[dict] = None

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # rope_theta overrides rope_freq_base since it's the official name.
        if self.rope_theta is not None:
            self.rope_freq_base = self.rope_theta

        if self.use_sdpa_with_kv_cache_op:
            assert self.use_kv_cache, "use_sdpa_with_kv_cache_op requires use_kv_cache"

        if self.hidden_dim is None:
            # If hidden_dim is not explicitly set in the ModelArgs,
            # then calculate implicitly based on dim and also multiple of `args.multiple_of`
            multiple_of = self.multiple_of
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            if self.ffn_dim_multiplier is not None:
                hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
            self.hidden_dim = find_multiple(hidden_dim, multiple_of)

        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads


class Rope(torch.nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        if self.params.use_hf_rope:
            self.precompute_freqs_cis = hf_precompute_freqs_cis
        else:
            self.precompute_freqs_cis = partial(
                precompute_freqs_cis,
                use_scaled=self.params.use_scaled_rope,
                scale_factor=self.params.rope_scale_factor,
            )
        freqs_cos, freqs_sin = self.precompute_freqs_cis(
            self.params.head_dim,
            (
                self.params.max_seq_len  # Normal llama2.
                if self.params.ffn_dim_multiplier is None
                else self.params.max_seq_len * 2  # Sharded checkpoint.
            ),
            self.params.rope_freq_base,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        if self.params.use_hf_rope:
            self.apply_rotary_emb = hf_apply_rotary_emb
        else:
            self.apply_rotary_emb = RotaryEmbedding()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        return self.apply_rotary_emb(q, k, freqs_cos, freqs_sin)

    def get_freqs(self, input_pos: Optional[torch.Tensor], seq_len: int):
        """
        Get the precomputed frequencies for the given input position and sequence length.

        Args:
            input_pos (torch.Tensor): The input position tensor.
            seq_len (int): The sequence length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The precomputed frequencies for the given input position and sequence length.
        """
        if self.params.use_kv_cache:
            assert (
                input_pos is not None
            ), "input_pos must be provided when use_kv_cache is True"

            if self.params.enable_dynamic_shape:
                # when KV cache is used, seqlen is most likely 1. We want to slice from the start_pos.
                input_pos_item = input_pos[-1].item()
                torch._check_is_size(input_pos_item)
                torch._check(input_pos_item < self.params.max_seq_len)
                # pyre-ignore: Incompatible parameter type [6]: torch.narrow does expect int or Tensor
                freqs_cos = self.freqs_cos.narrow(0, input_pos_item, seq_len)
                # pyre-ignore: Incompatible parameter type [6]
                freqs_sin = self.freqs_sin.narrow(0, input_pos_item, seq_len)
            else:
                # When not using dynamic shape, use of the .item results in
                # symints, due to querying the data from tensor.
                # this path avoids that for mps backend, although probably mps backend
                # can support dynamic shape?
                freqs_cos = self.freqs_cos[input_pos]
                freqs_sin = self.freqs_sin[input_pos]

        else:
            assert input_pos is None, "input_pos is unused when use_kv_cache is False"
            freqs_cos = self.freqs_cos[:seq_len]
            freqs_sin = self.freqs_sin[:seq_len]
        return freqs_cos, freqs_sin


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        transpose_cache: bool,
        enable_dynamic_shape: bool,
        dtype=torch.float32,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.is_transposed = transpose_cache
        if transpose_cache:
            cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        else:
            cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)

        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.transpose_cache = transpose_cache
        self.enable_dynamic_shape = enable_dynamic_shape
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype, device="cpu")
        )

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_pos: [S], k_val: [B, H, S, D] or [B, S, H, D] depending on transpose_cache
        if self.enable_dynamic_shape:
            start_pos = input_pos[0].item()
            torch._check_is_size(start_pos)
            torch._check(start_pos < self.max_seq_length)
            dim_to_slice = 2 if self.transpose_cache else 1
            seq_length = k_val.size(dim_to_slice)
            # Replace the entry in the cache for this token
            # The following lines are equivalent to:
            # cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            # cache_v[:bsz, start_pos : start_pos + seqlen] = xv
            # when dim_to_slice is 1
            # We use .narrow() here to make the compiler happy
            # pyre-ignore: Incompatible parameter type [6]
            narrowed_k = self.k_cache.narrow(dim_to_slice, start_pos, seq_length)
            # pyre-ignore: Incompatible parameter type [6]
            narrowed_v = self.v_cache.narrow(dim_to_slice, start_pos, seq_length)

            narrowed_k.copy_(k_val)
            narrowed_v.copy_(v_val)
            return self.k_cache, self.v_cache
        else:
            k_out = self.k_cache
            v_out = self.v_cache
            if self.transpose_cache:
                k_out[:, :, input_pos] = k_val
                v_out[:, :, input_pos] = v_val
            else:
                k_out[:, input_pos] = k_val
                v_out[:, input_pos] = v_val

            return k_out, v_out


class SDPA(nn.Module):
    def __init__(
        self,
        kv_cache: KVCache,
        dim: int,
        head_dim: int,
        n_rep: int,
        max_seq_len: int,
        enable_dynamic_shape: bool,
    ):
        super().__init__()
        self.kv_cache = kv_cache
        self.dim = dim
        self.head_dim = head_dim
        self.n_rep = n_rep
        self.max_seq_len = max_seq_len
        self.enable_dynamic_shape = enable_dynamic_shape

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,  # Already have rotary embeddings. (bs, seqlen, n_local_heads, head_dim)
        k: torch.Tensor,  # Already have rotary embeddings. (bs, seqlen, n_local_kv_heads, head_dim)
        v: torch.Tensor,  # (bs, seqlen, n_local_kv_heads, head_dim)
        bsz,
        seqlen,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k, v = self.kv_cache.update(input_pos, k, v)
        if self.enable_dynamic_shape:
            start_pos = input_pos[-1].item()
            torch._check_is_size(start_pos)
            torch._check(start_pos < self.max_seq_len)
            seq_length = q.size(2)
            # pyre-ignore: Incompatible parameter type [6]
            attn_mask = mask.narrow(0, start_pos, seq_length)
        else:
            attn_mask = mask[None, None, input_pos]

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)

        return y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int, rope: Rope):
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads
        self.n_kv_heads = self.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = self.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.layer_id = layer_id

        self.rope = rope

        causal_mask = torch.tril(
            torch.ones(
                self.max_seq_len,
                self.max_seq_len,
                dtype=torch.bool,
                device="cpu",
            )
        )
        self.register_buffer("mask", causal_mask, persistent=False)

        if self.use_kv_cache:
            self.kv_cache = KVCache(
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
                not args.use_sdpa_with_kv_cache_op,  # if we are using the custom op don't transpose the cache. Expect untransposed q k v
                args.enable_dynamic_shape,
            )
            self.SDPA = SDPA(
                kv_cache=self.kv_cache,
                dim=self.n_local_heads * self.head_dim,
                head_dim=self.head_dim,
                n_rep=self.n_rep,
                max_seq_len=self.max_seq_len,
                enable_dynamic_shape=args.enable_dynamic_shape,
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # We need view_copy elimination
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        q, k = self.rope.forward(q, k, freqs_cos, freqs_sin)

        if self.use_kv_cache:
            assert input_pos is not None
            output = self.SDPA(input_pos, q, k, v, bsz, seqlen, self.mask)
            return self.wo(output)

        q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # grouped multiquery attention: expand out keys and values
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        assert hasattr(self, "mask")

        mask = self.mask[:seqlen, :seqlen]

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.hidden_dim is not None
        hidden_dim: int = args.hidden_dim
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, rope: Rope):
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.attention = Attention(args, layer_id, rope)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, input_pos=None):  # x: 1xN
        h = self.attention.forward(
            self.attention_norm(x), freqs_cos, freqs_sin, input_pos
        )

        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.rope = Rope(params)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, self.rope))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.use_kv_cache = params.use_kv_cache
        self.generate_full_logits = params.generate_full_logits
        self.max_seq_len = params.max_seq_len
        self.input_prune_map = params.input_prune_map
        self.output_prune_map = params.output_prune_map

    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,  # tokens
        input_pos: Optional[
            torch.LongTensor
        ] = None,  # Scalar tensor indicating size of window of the caches
        h: Optional[torch.FloatTensor] = None,  # embeddings
    ) -> torch.Tensor:
        if (tokens is None) ^ (h is not None):
            raise ValueError(
                "You cannot specify both tokens and h at the same time, and must specify either one"
            )
        if tokens is not None and h is None:
            h = self.tok_embeddings(tokens)
        seqlen = h.shape[1]
        freqs_cos, freqs_sin = self.rope.get_freqs(input_pos, seqlen)

        for layer in self.layers:
            h = layer(
                h,
                freqs_cos,
                freqs_sin,
                input_pos,
            )

        if not self.generate_full_logits:
            # Only the last logit is used for the new generated token
            h = h[:, -1, :]

        h = self.norm(h)

        logits = self.output(h)

        if self.output_prune_map is not None:
            # expand to original size so that downstream applications can use the logits as-is.
            if self.generate_full_logits:
                # (1, seq_len, pruned_size) -> (1, seq_len, original_size)
                expanded_logits = torch.full(
                    [logits.shape[0], logits.shape[1], self.vocab_size],
                    float("-inf"),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                expanded_logits[:, :, list(self.output_prune_map.values())] = logits
            else:
                # (1, pruned_size) -> (1, original_size)
                expanded_logits = torch.full(
                    [logits.shape[0], self.vocab_size],
                    float("-inf"),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                expanded_logits[:, list(self.output_prune_map.values())] = logits
            logits = expanded_logits

        return logits

from executorch.extension.export_util.utils import export_to_edge
from torch.nn.attention import SDPBackend
from torch._export import capture_pre_autograd_graph
from executorch import exir
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, ExecutorchBackendConfig, to_edge, to_edge_transform_and_lower


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshi-weights", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--relaxed", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(42)

    params = ModelArgs(
        head_dim=64,
        hidden_dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        norm_eps=1e-5,
        rope_theta=500000,
        vocab_size=128256,
    )
    model = Transformer(params).to(dtype=torch.bfloat16)
    if args.verbose:
        print(model)
    print("moshi model loaded")

    print("running the model")
    sample_codes = torch.zeros((1, 1), dtype=torch.int64).to(args.device)
    for i in range(5):
        start_time = time.time()
        out_codes = model(sample_codes)
        dt = time.time() - start_time
        # print(out_codes[0, 0, 0, :20])
        print(f"step {i} shape: {out_codes.shape} dt: {1000 * dt:.0f}ms")


    print("exporting to aten and edge")
    aten_dialect: ExportedProgram = capture_pre_autograd_graph(model, (sample_codes,))
    edge_config = exir.EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_type_promotion=False,
        _skip_dim_order=True,
    )
    edge_manager = export_to_edge(
        aten_dialect,
        (sample_codes,),
        dynamic_shapes=None,
        edge_constant_methods={},
        edge_compile_config=edge_config,
        verbose=False,
    )
    edge_manager = edge_manager.to_backend(XnnpackPartitioner())
    if args.verbose:
        print(edge_manager.exported_program().graph)
    executorch_program = edge_manager.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[
                # If there are Linear operations left in the graph, let's execute
                # them with the optimized op_linear rather than materializing a
                # transpose followed by a regular op_mm.
                ConvertToLinearPass(),
                QuantFusionPass(),
            ],
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )

    filename = "llama-lm.pte"
    print(f"writing {filename}")
    with open(filename, "wb") as file:
        file.write(executorch_program.buffer)

with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
    main()

