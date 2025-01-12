# python scripts/llama3/import_llama.py --checkpoint /Users/laurent/tmp/llama-3.2/consolidated.00.pth --params /Users/laurent/tmp/llama-3.2/params.json -kv -X -d bf16 --output_name foo.pte
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

# Example script for exporting Llama2 to flatbuffer

import argparse
import copy
import json
import logging
import shlex
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Callable, List, Optional, Union

import pkg_resources

import torch

from executorch.devtools.etrecord import generate_etrecord

from executorch.extension.llm.export.builder import DType, LLMEdgeManager

from executorch.extension.llm.export.partitioner_lib import (
    get_coreml_partitioner,
    get_mps_partitioner,
    get_qnn_partitioner,
    get_vulkan_partitioner,
    get_xnnpack_partitioner,
)

from executorch.extension.llm.export.quantizer_lib import (
    get_coreml_quantizer,
    get_pt2e_quantization_params,
    get_pt2e_quantizers,
    get_qnn_quantizer,
)
from executorch.util.activation_memory_profiler import generate_memory_trace

import importlib
import os
from typing import Any, Tuple

import torch

import json
import os
from pathlib import Path

import torch

from abc import ABC, abstractmethod

import torch
import argparse
import copy

from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple
import time

import torch
import torch.nn.functional as F

from torch import nn


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
):
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device="cpu", dtype=torch.int64).float() / dim)
    )
    t = torch.arange(end, device=freqs.device, dtype=torch.int64).type_as(
        freqs
    )
    freqs = torch.outer(t, freqs).float()
    emb = torch.cat((freqs, freqs), dim=-1)
    freqs_cos = torch.cos(emb)
    freqs_sin = torch.sin(emb)
    return freqs_cos, freqs_sin




def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(q_, k_, cos, sin, unsqueeze_dim=1):
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
    # Convert to float as executorch 0.5 does not yet support the neg operation on bfloat16 tensors.
    q, k = q_.float(), k_.float()
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Interleave them instead of usual shape
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(dtype=q_.dtype), k_embed.to(dtype=k_.dtype)

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


class HeliumRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

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
    max_batch_size: int = 1
    max_seq_len: int = 2048
    use_kv_cache: bool = False  # Use key/value cache
    use_sdpa_with_kv_cache_op: bool = (
        False  # Use custom sdpa op that updates kv cache in-place
    )
    enable_dynamic_shape: bool = False  # export model with dynamic shape support
    rope_theta: Optional[float] = (
        None  # The official name to override self.rope_freq_base.
    )
    rope_freq_base: float = 10000.0  # The base frequency for RoPE. Keep it for BC.
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
        freqs_cos, freqs_sin = precompute_freqs_cis(
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
        self.attention_norm = HeliumRMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = HeliumRMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, input_pos=None):  # x: 1xN
        h = self.attention.forward(
            self.attention_norm(x), freqs_cos, freqs_sin, input_pos
        )

        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Embedding(nn.Embedding):
    def __init__(self, *args, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        y = torch.where(is_zero[..., None], zero, y)
        return y

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = Embedding(params.vocab_size, params.dim)
        self.rope = Rope(params)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, self.rope))
        self.norm = HeliumRMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.use_kv_cache = params.use_kv_cache
        self.max_seq_len = params.max_seq_len

    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,  # tokens
        input_pos: Optional[
            torch.LongTensor
        ] = None,  # Scalar tensor indicating size of window of the caches
    ) -> torch.Tensor:
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

        h = h[:, -1, :]
        h = self.norm(h)
        logits = self.output(h)
        return logits



class EagerModelBase(ABC):
    """
    Abstract base class for eager mode models.

    This abstract class defines the interface that eager mode model classes should adhere to.
    Eager mode models inherit from this class to ensure consistent behavior and structure.
    """

    @abstractmethod
    def __init__(self):
        """
        Constructor for EagerModelBase.

        This initializer may be overridden in derived classes to provide additional setup if needed.
        """
        pass

    @abstractmethod
    def get_eager_model(self) -> torch.nn.Module:
        """
        Abstract method to return an eager PyTorch model instance.

        Returns:
            nn.Module: An instance of a PyTorch model, suitable for eager execution.
        """
        raise NotImplementedError("get_eager_model")

    @abstractmethod
    def get_example_inputs(self):
        """
        Abstract method to provide example inputs for the model.

        Returns:
            Any: Example inputs that can be used for testing and tracing.
        """
        raise NotImplementedError("get_example_inputs")

class Llama2Model(EagerModelBase):
    def __init__(self, **kwargs):
        import pkg_resources

        # default path to the resource file
        # It currently supports 3 ways of specifying the checkpoint location:
        # 1. Using default path locates in examples/models/llama2/params
        # 2. Passing in the checkpoint path and params via kwargs
        # 3. Using the path from pkg_resources, only works with buck2
        try:
            # The 3rd way, if we can import this path, we are running with buck2, all resources can be accessed with pkg_resources.resource_filename
            # pyre-ignore
            # from executorch.examples.models.llama2 import params

            ckpt_dir = Path(
                pkg_resources.resource_filename(
                    "executorch.examples.models.llama2", "params"
                )
            )
        except:
            # The 1st way
            ckpt_dir = Path(__file__).absolute().parent / "params"

        # Check if checkpoint_dir was provided for a sharded checkpoint.
        checkpoint_dir = kwargs.get("checkpoint_dir", None)

        # Use single checkpoint file.
        checkpoint_path = kwargs.get("checkpoint", ckpt_dir / "demo_rand_params.pth")

        params_path = kwargs.get("params", ckpt_dir / "demo_config.json")

        self.use_kv_cache = kwargs.get("use_kv_cache", False)
        self.use_sdpa_with_kv_cache_op = kwargs.get("use_sdpa_with_kv_cache", False)
        self.generate_full_logits = kwargs.get("generate_full_logits", False)
        self.enable_dynamic_shape = kwargs.get("enable_dynamic_shape", False)

        self.max_seq_len = kwargs.get("max_seq_len", 128)
        self.args = kwargs.get("args", None)
        # The example is using a dummy small model with random weights for demo purpose only.
        # Follow the instruction in https://github.com/facebookresearch/llama to download the model
        device = "cpu"
        # flake8: noqa: TOR102
        cps = []
        if checkpoint_dir is not None:
            # Load multiple checkpoint; ignore the single path.
            checkpoint_path = None
            for i in range(4):
                cp_name = f"consolidated.{i}.pth"
                print(f"Loading {cp_name}")
                cps.append(
                    torch.load(
                        os.path.join(checkpoint_dir, cp_name),
                        map_location=device,
                        mmap=True,
                    )
                )
            checkpoint = {}
            for key in cps[0].keys():
                if not torch.allclose(cps[0][key], cps[1][key]):
                    values = (cps[0][key], cps[1][key], cps[2][key], cps[3][key])
                    if "wo" in key or "w2" in key:
                        # Concat on dim=1 for "wo" and "w2".
                        checkpoint[key] = torch.cat(values, dim=1)
                    else:
                        # Concat on dim=0 for everything else.
                        checkpoint[key] = torch.cat(values, dim=0)
                else:
                    # Do not duplicate layers shared between each checkpoint.
                    checkpoint[key] = cps[0][key]
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)
        fairseq2_checkpoint = kwargs.get("fairseq2", False)
        if fairseq2_checkpoint:
            print("Using fairseq2 checkpoint")
            checkpoint = convert_to_llama_checkpoint(checkpoint=checkpoint)
        if "model" in checkpoint:
            # NB: some checkpoint contains a "model" field, which is the actual weights dict
            checkpoint = checkpoint["model"]

        if (not fairseq2_checkpoint) and checkpoint.get(
            "final_proj.weight", None
        ) is not None:
            print(
                """

************************************************************
This looks like a Fairseq2 checkpoint (based on the presence
of `final_proj.weight`.

You can import Fairseq2 checkpoints using the --fairseq2
option, but --fairseq2 was not specified.  Please verify
the checkpoint format to avoid generating faulty models.
************************************************************
"""
            )

        # get checkpoint dtype
        self.dtype = None
        if len(checkpoint) > 0:
            first_key = next(iter(checkpoint))
            first = checkpoint[first_key]
            self.dtype = first.dtype
            mismatched_dtypes = [
                (key, value.dtype)
                for key, value in checkpoint.items()
                if value.dtype != self.dtype
            ]
            if len(mismatched_dtypes) > 0:
                print(
                    f"Mixed dtype model. Dtype of {first_key}: {first.dtype}. Mismatches in the checkpoint: {mismatched_dtypes}"
                )
        with open(params_path, "r") as f:
            params = json.loads(f.read())
            del params["use_scaled_rope"]
        max_seq_len = self.max_seq_len
        max_batch_size = 1
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            use_kv_cache=self.use_kv_cache,
            use_sdpa_with_kv_cache_op=self.use_sdpa_with_kv_cache_op,
            # generate_full_logits=self.generate_full_logits,
            enable_dynamic_shape=self.enable_dynamic_shape,
            **params,
        )
        if kwargs.get("fairseq2", False):
            print("Using fairseq2 checkpoint")
            checkpoint = convert_to_llama_checkpoint(checkpoint=checkpoint)
        if kwargs.get("verbose", False):
            print("============= weights ================")
            print("{key} : {weights.numel()} : {weights.size()}")
            for key, weights in checkpoint.items():
                print(f"{key} : {weights.numel()} : {weights.size()}")
            print("============= /weights ================")

        # Within the device="meta" context, tensors that are created do not carry data.
        # They possess all other metadata a tensor carries such as size, stride, requires_grad.
        with torch.device("meta"):
            self.model_ = Transformer(model_args)

        if "int8" in str(checkpoint_path):
            print("Using int8 weight-only quantization!")
            # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.examples.models.source_transformation.quantize`
            from ..source_transformation.quantize import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(self.model_)
            self.model_ = simple_quantizer.convert_for_runtime()
        elif "8da4w" in str(checkpoint_path):
            print("Using int4 weight and int8 dynamic activation quantization!")
            from torchao.quantization.quant_api import Int8DynActInt4WeightQuantizer

            self.model_ = Int8DynActInt4WeightQuantizer()._convert_for_runtime(
                self.model_
            )
        elif hasattr(self.args, "use_spin_quant") and self.args.use_spin_quant:
            print("Using SPIN quantization.")
            assert hasattr(self.args, "group_size"), "group_size must be specified"
            assert hasattr(
                self.args, "quantization_mode"
            ), "quantization_mode must be specified"
            assert hasattr(
                self.args, "dtype_override"
            ), "dtype_override must be specified"
            from .source_transformation.spin_quant import (
                sanitize_checkpoint_from_spinquant,
                transform_for_spinquant,
            )

            mapping = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }

            self.model_ = transform_for_spinquant(
                self.model_,
                checkpoint,
                self.args.group_size,
                self.args.quantization_mode,
                mapping[self.args.dtype_override],
            )

            sanitize_checkpoint_from_spinquant(
                checkpoint,
                self.args.group_size,
            )

        # assign=True: load params/buffers by assignment instead of performing an in-place copy.
        # Because we are using device="meta", tensors do not have memory associated with them
        # and an in-place copy is a no-op. Use assign=True in load_state_dict for this scenario.
        missing, unexpected = self.model_.load_state_dict(
            checkpoint,
            strict=False,
            assign=True,
        )  # self.model_ = Transformer(gptconf)
        if kwargs.get("verbose", False):
            print("============= missing keys ================")
            print(missing)
            print("============= /missing ================")
            print("============= unexpected keys ================")
            print(unexpected)
            print("============= /unexpected ================")

    def get_eager_model(self):
        if self.dtype:
            # convert to the type of the provided checkpoint
            # input and output are torch.long, so signature unchanged
            return self.model_.to(self.dtype)
        else:
            # int8 quantization code has some bf16,
            # switch all to FP32
            return self.model_.to(torch.float32)

    def get_example_inputs(self):
        if self.use_kv_cache:
            return self.get_example_inputs_kvcache_sdpa()
        else:
            return (
                torch.tensor(
                    [[1, 2, 3]], dtype=torch.long
                ),  # tokens, with kv cache our input token length is always just 1 token.
            )

    # assumption is the custom op doesnt support dynamic shape right now. It might but its untested so lets first get static shape working
    def get_example_inputs_kvcache_sdpa(self):
        if self.enable_dynamic_shape:
            return (
                torch.tensor([[2, 3, 4]], dtype=torch.long),
                torch.tensor([0], dtype=torch.long),
            )
        else:
            return (
                torch.tensor(
                    [[1]], dtype=torch.long
                ),  # tokens, with kv cache our input token length is always just 1 token.
                torch.tensor(
                    [0], dtype=torch.long
                ),  # start_pos, what token of output are we on.
            )
class EagerModelFactory:
    """
    A factory class for dynamically creating instances of classes implementing EagerModelBase.
    """

    @staticmethod
    def create_model(
        module_name, model_class_name, **kwargs
    ) -> Tuple[torch.nn.Module, Any, Any]:
        """
        Create an instance of a model class that implements EagerModelBase and retrieve related data.

        Args:
            module_name (str): The name of the module containing the model class.
            model_class_name (str): The name of the model class to create an instance of.

        Returns:
            Tuple[nn.Module, Any]: A tuple containing the eager PyTorch model instance and example inputs,
              and any dynamic shape information for those inputs.

        Raises:
            ValueError: If the provided model class is not found in the module.
        """
        package_prefix = "executorch." if not os.getcwd().endswith("executorch") else ""
        module = importlib.import_module(
            f"{package_prefix}examples.models.{module_name}"
        )
        print(">>>", module, model_class_name)
        model = Llama2Model(**kwargs)
        if hasattr(model, "get_dynamic_shapes"):
            return (
                model.get_eager_model(),
                model.get_example_inputs(),
                model.get_dynamic_shapes(),
            )
        else:
            return model.get_eager_model(), model.get_example_inputs(), None

IS_FBCODE = True  #  os.environ.get("FBCODE_PLATFORM", False)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

pkg_name = __name__
verbosity_setting = None


class WeightType(Enum):
    LLAMA = "LLAMA"
    FAIRSEQ2 = "FAIRSEQ2"


def set_pkg_name(name: str) -> None:
    global pkg_name
    pkg_name = name


def get_resource_path(resource_name) -> str:
    return pkg_resources.resource_filename(pkg_name, resource_name)


def set_verbosity(val):
    global verbosity_setting
    verbosity_setting = val


def verbose_export():
    return verbosity_setting


def build_model(
    modelname: str = "model",
    extra_opts: str = "",
    *,
    par_local_output: bool = False,
    resource_pkg_name: str = __name__,
) -> str:
    if False:  # par_local_output:
        output_dir_path = "par:."
    else:
        output_dir_path = "."

    argString = f"--checkpoint par:{modelname}_ckpt.pt --params par:{modelname}_params.json {extra_opts} --output-dir {output_dir_path}"
    parser = build_args_parser()
    args = parser.parse_args(shlex.split(argString))
    # pkg_name = resource_pkg_name
    return export_llama(modelname, args)


def build_args_parser() -> argparse.ArgumentParser:
    ckpt_dir = f"{Path(__file__).absolute().parent.as_posix()}"
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", default=".", help="output directory")
    # parser.add_argument(
    #     "-q", "--quantized_ckpt", default=None, help="quantized checkpoint file"
    # )
    parser.add_argument(
        "-E",
        "--embedding-quantize",
        default=None,
        type=str,
        help="type of embedding quantization, '<bitwidth>,<groupsize>', e.g., '8,1024'.",
    )
    parser.add_argument(
        "--pt2e_quantize",
        default=None,
        choices=[
            "xnnpack_dynamic",
            "xnnpack_dynamic_qc4",
            "qnn_8a8w",
            "qnn_16a16w",
            "qnn_16a4w",
            "coreml_c4w",
            "coreml_8a_c8w",
            "coreml_8a_c4w",
            "coreml_baseline_8a_c8w",
            "coreml_baseline_8a_c4w",
        ],
        help="Use PT2E quantization. Comma separated options. e.g. xnnpack_dynamic (for per channel 8 bit weight), xnnpack_dynamic_qc4 (for per channel 4 bit weight), embedding.",
    )
    parser.add_argument(
        "-qmode",
        "--quantization_mode",
        type=str,
        default=None,
        choices=["int8", "8da4w", "8da4w-gptq"],
        help="type of quantization",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        default=f"{ckpt_dir}/params/demo_rand_params.pth",
        help="checkpoint path",
    )

    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="checkpoint directory. Use with a sharded checkpoint, not for the standard llama2 model. Note, checkpoint_dir takes precedence over checkpoint if both are set.",
    )

    parser.add_argument(
        "--calibration_tasks",
        nargs="+",
        type=str,
        default=None,
        help="Tasks for GPTQ calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=None,
        help="number of samples used for calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_seq_length",
        type=int,
        default=None,
        help="Sequence length for GPTQ calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_data",
        type=str,
        default="Once upon a time",
        help="Calibration prompts from users",
    )
    parser.add_argument(
        "-t",
        "--tokenizer_path",
        default=None,
        help="tokenizer path (Note: .model not .bin)",
    )
    parser.add_argument(
        "-kv",
        "--use_kv_cache",
        default=False,
        action="store_true",
        help="Whether or not to export a model using kv cache",
    )
    parser.add_argument(
        "--num_sharding",
        type=int,
        default=0,
        help="Specify the number of splits by inserting the fallback custom op. The graph will be split evenly by layers.",
    )
    parser.add_argument(
        "--use_sdpa_with_kv_cache",
        default=False,
        action="store_true",
        help="Whether to use sdpa_with_kv_cache update op when using kv cache",
    )
    parser.add_argument(
        "--disable_dynamic_shape",
        dest="enable_dynamic_shape",
        default=True,  # Enable this by default
        action="store_false",
        help="Enable dynamic shape along seq dim. Used for faster prefill",
    )
    parser.add_argument(
        "-p",
        "--params",
        default=f"{ckpt_dir}/params/demo_config.json",
        help="config.json",
    )
    parser.add_argument(
        "--optimized_rotation_path",
        default=None,
        required=False,
        help="[QNN backend] Optimized rotation checkpoint path. Just apply R1/R2 here."
        "You can download the optimized rotation matrices from https://github.com/facebookresearch/SpinQuant/tree/main",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        default=None,
        help='metadata string in json format. Example {"key": 1, "key2": "value2"}',
    )
    parser.add_argument(
        "-s",
        "--so_library",
        default=None,
        required=False,
        help="shared library for quantized operators",
    )
    parser.add_argument(
        "--profile_memory",
        required=False,
        action="store_true",
        help="Generate chrome trace of activation memory for intermediate tensors.",
    )
    parser.add_argument(
        "-prof",
        "--profile_path",
        default=None,
        help="Use cProfile to profile model export. Results saved to profile_path as a html file.",
    )
    parser.add_argument(
        "-G",
        "--group_size",
        type=int,
        default=None,
        help="group_size for weight quantization",
    )

    parser.add_argument(
        "-d",
        "--dtype-override",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        help="Override the dtype of the model (default is the checkpoint dtype)."
        "Options: fp32, fp16, bf16. Please be aware that only some backends support fp16 and bf16.",
    )

    parser.add_argument(
        "-n",
        "--output_name",
        default=None,
        help="Override the output filename of the saved pte model file.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="maximum length sequence to evaluate",
    )

    parser.add_argument("-2", "--fairseq2", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-X", "--xnnpack", action="store_true")
    parser.add_argument("-V", "--vulkan", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--coreml", action="store_true")
    parser.add_argument(
        "--coreml-enable-state",
        action="store_true",
        help="This option is only for coreml, and is only supported for MacOS15+/iOS18+",
    )
    parser.add_argument(
        "--coreml-preserve-sdpa",
        action="store_true",
        help="This option is only for coreml: Preserve sdpa in torch edge program to use coreml iOS18.sdpa op",
    )
    parser.add_argument(
        "--coreml-quantize",
        default=None,
        choices=["b4w"],
        help="This option is only for coreml: Use coreml quantization, e.g. b4w (for blockwise 4 bit weight)",
    )
    parser.add_argument(
        "--coreml-ios",
        type=int,
        default=15,
        choices=(15, 16, 17, 18),
        help="This option is only for coreml: The minimum iOS version to deploy",
    )
    parser.add_argument(
        "--qnn",
        action="store_true",
        help="Delegate llama2 to qnn backend (Qualcomm), please use it --kv_cahce=True",
    )

    parser.add_argument(
        "--expand_rope_table",
        default=False,
        action="store_true",
        help="[Temp workaround] Expand sin/cos table in head dim to take vectorized path in optimized kernels.",
    )

    parser.add_argument(
        "--generate_etrecord",
        action="store_true",
        required=False,
        default=False,
        help="Generate the ETRecord debug artifact.",
    )

    parser.add_argument(
        "--generate_full_logits",
        action="store_true",
        required=False,
        default=False,
        help="Generate logits for all inputs.",
    )

    parser.add_argument(
        "--soc_model",
        help="[QNN backend] SoC model of current device. e.g. 'SM8650' for Snapdragon 8 Gen 3.",
        type=str,
        required=False,
        default="SM8650",
    )

    parser.add_argument(
        "-sq",
        "--use_spin_quant",
        type=str,
        default=None,
        choices=["cuda", "native"],
        help="Use SpinQuant for better quantization performance. Only support cuda and native.",
    )
    return parser


def canonical_path(path: Union[str, Path], *, dir: bool = False) -> str:
    path = str(path)

    if verbose_export():
        print(f"creating canonical path for {path}")

    if not path.startswith("par:"):
        return path

    if not IS_FBCODE:
        print("not FBCODE")
        return path[4:]
    else:
        return_val = pkg_resources.resource_filename(pkg_name, path[4:])
        if verbose_export():
            print(f"canonical name is: {return_val}")
        return return_val


def export_llama(modelname, args) -> str:
    if args.profile_path is not None:
        try:
            from executorch.util.python_profiler import CProfilerFlameGraph

            with CProfilerFlameGraph(args.profile_path):
                builder = _export_llama(modelname, args)
                assert (
                    filename := builder.get_saved_pte_filename()
                ) is not None, "Fail to get file name from builder"
                return filename
        except ImportError:
            print(
                "Please run `pip install snakeviz` to install required dependencies for cProfiler flamegraph."
            )
            return ""
    else:
        builder = _export_llama(modelname, args)
        assert (
            filename := builder.get_saved_pte_filename()
        ) is not None, "Fail to get file name from builder"
        return filename


def _prepare_for_llama_export(modelname: str, args) -> LLMEdgeManager:
    """
    Helper function for export_llama. Loads the model from checkpoint and params,
    and sets up a LLMEdgeManager with initial transforms and dtype conversion.

    Returns a LLMEdgeManager prior to calling export_to_edge with quantizers
    """

    print("HERE")
    # load model from checkpoint and params.json
    checkpoint_path = canonical_path(args.checkpoint) if args.checkpoint else None
    checkpoint_dir = (
        canonical_path(args.checkpoint_dir) if args.checkpoint_dir else None
    )
    params_path = canonical_path(args.params)
    output_dir_path = canonical_path(args.output_dir, dir=True)
    weight_type = WeightType.FAIRSEQ2 if args.fairseq2 else WeightType.LLAMA

    # dtype override
    if args.dtype_override is not None:
        dtype_override = DType[args.dtype_override]
    elif args.quantization_mode in ["8da4w", "8da4w-gptq"]:
        dtype_override = DType["fp16"]
    else:
        dtype_override = None

    model = (
        _load_llama_model(
            modelname=modelname,
            checkpoint=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            params_path=params_path,
            use_kv_cache=args.use_kv_cache,
            use_sdpa_with_kv_cache=args.use_sdpa_with_kv_cache,
            generate_full_logits=args.generate_full_logits,
            weight_type=weight_type,
            enable_dynamic_shape=args.enable_dynamic_shape,
            calibration_tasks=args.calibration_tasks,
            calibration_limit=args.calibration_limit,
            calibration_seq_length=args.calibration_seq_length,
            calibration_data=args.calibration_data,
            tokenizer_path=args.tokenizer_path,
            verbose=args.verbose,
            max_seq_len=args.max_seq_length,
            metadata_str=args.metadata,
            args=args,
        )
        .set_output_dir(output_dir_path)
        .to_dtype(dtype_override)
        # .source_transform(_get_source_transforms(modelname, dtype_override, args))
    )
    print(model)
    return model


def get_quantizer_and_quant_params(args):
    pt2e_quant_params = get_pt2e_quantization_params(
        args.pt2e_quantize, args.quantization_mode
    )
    quantizers = get_pt2e_quantizers(pt2e_quant_params, args.so_library)
    quant_dtype = None
    if args.qnn and args.pt2e_quantize:
        assert len(quantizers) == 0, "Should not enable both xnnpack and qnn"
        qnn_quantizer, quant_dtype = get_qnn_quantizer(
            args.pt2e_quantize, args.quantization_mode
        )
        quantizers.append(qnn_quantizer)
    if args.coreml and args.pt2e_quantize:
        assert len(quantizers) == 0, "Should not enable both xnnpack / qnn and coreml"
        coreml_quantizer = get_coreml_quantizer(args.pt2e_quantize)
        quantizers.append(coreml_quantizer)
    logging.info(f"Applying quantizers: {quantizers}")
    return pt2e_quant_params, quantizers, quant_dtype


def _validate_args(args):
    """
    TODO: Combine all the backends under --backend args
    """
    if args.enable_dynamic_shape and (args.coreml or args.mps or args.qnn):
        raise ValueError(
            "Dynamic shape is not supported with coreml, MPS or qnn backends."
            " Please use --disable_dynamic_shape."
        )

    if args.num_sharding > 0 and not args.qnn:
        raise ValueError("Model shard is only supported with qnn backend now.")


def _export_llama(modelname, args) -> LLMEdgeManager:  # noqa: C901
    _validate_args(args)
    pt2e_quant_params, quantizers, quant_dtype = get_quantizer_and_quant_params(args)

    # export_to_edge
    builder_exported_to_edge = (
        _prepare_for_llama_export(modelname, args)
        .capture_pre_autograd_graph()
        #.pt2e_quantize(quantizers)
        .export_to_edge()
    )

    modelname = builder_exported_to_edge.modelname

    # to_backend
    partitioners = []
    if pt2e_quant_params is not None and pt2e_quant_params.quantize_linear is not None:
        partitioners.append(get_xnnpack_partitioner())
        modelname = f"xnnpack_dq_{modelname}"

    if args.xnnpack:
        partitioners.append(get_xnnpack_partitioner())
        modelname = f"xnnpack_{modelname}"

    if args.vulkan:
        partitioners.append(
            get_vulkan_partitioner(
                args.dtype_override,
                args.quantization_mode,
            )
        )
        modelname = f"vulkan_{modelname}"

    if args.mps:
        partitioners.append(get_mps_partitioner(args.use_kv_cache))
        modelname = f"mps_{modelname}"

    if args.coreml:
        coreml_partitioner = get_coreml_partitioner(
            args.coreml_ios,
            args.embedding_quantize,
            args.pt2e_quantize,
            args.coreml_quantize,
        )
        partitioners.append(coreml_partitioner)
        modelname = f"coreml_{modelname}"

    if args.qnn:
        from executorch.extension.llm.custom_ops import model_sharding

        partitioners.append(
            get_qnn_partitioner(
                args.use_kv_cache, args.pt2e_quantize, args.num_sharding, args.soc_model
            )
        )
        # pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.backends.qualcomm.utils.utils`
        from executorch.backends.qualcomm.utils.utils import _transform

        # pyre-ignore: Undefined attribute [16]: Module `executorch.backends` has no attribute `qualcomm`, Optional type has no attribute `exported_program`
        _transform(builder_exported_to_edge.edge_manager.exported_program())

        if args.num_sharding > 0:
            model_sharding.split_graph(
                builder_exported_to_edge.edge_manager.exported_program(),
                builder_exported_to_edge.metadata["get_n_layers"],
                shares=args.num_sharding,
            )

    if args.generate_etrecord:
        if not builder_exported_to_edge.edge_manager:
            raise ValueError("Unable to generate etrecord due to missing edge manager.")

        logging.info("Generating etrecord")
        # Copy the edge manager which will be serialized into etrecord. This is memory-wise expensive.
        edge_manager_copy = copy.deepcopy(builder_exported_to_edge.edge_manager)
        print("PART2", partitioners)
        builder = builder_exported_to_edge.to_backend(partitioners)
        if args.num_sharding > 0 and args.qnn:
            from executorch.backends.qualcomm.utils.utils import canonicalize_program

            canonicalize_program(builder.edge_manager.exported_program())

        builder = builder.to_executorch()

        # Generate ETRecord
        if edge_manager_copy:
            generate_etrecord(
                et_record="etrecord.bin",
                edge_dialect_program=edge_manager_copy,
                executorch_program=builder.export_program,
            )
            logging.info("Generated etrecord.bin")
    else:
        print("PART", partitioners)
        builder = builder_exported_to_edge.to_backend(partitioners)
        if args.num_sharding > 0 and args.qnn:
            from executorch.backends.qualcomm.utils.utils import canonicalize_program

            canonicalize_program(builder.edge_manager.exported_program())

        builder = builder.to_executorch()

    if args.profile_memory:
        generate_memory_trace(builder.export_program, "memory_profile.json")

    if builder.dtype == DType.fp16:
        modelname = f"{modelname}_h"

    if args.output_name:
        modelname = args.output_name
        if modelname.endswith(".pte"):
            output_file = modelname
            modelname = modelname[:-4]
            print(f"modelname: {modelname}")
            print(f"output_file: {output_file}")
        else:
            output_file = f"{builder.output_dir}/{modelname}.pte"
            print(f"modelname: {modelname}")
            print(f"output_file: {output_file}")
    else:
        output_file = f"{builder.output_dir}/{modelname}.pte"

    builder.save_to_pte(output_file)

    return builder


def _load_llama_model_metadata(
    weight_type: WeightType,
    use_kv_cache: bool,
    use_sdpa_with_kv_cache: bool,
    enable_dynamic_shape: bool,
    model_args: ModelArgs,
    metadata_str: Optional[str] = None,
):
    is_fairseq2 = weight_type == WeightType.FAIRSEQ2
    metadata = {
        "append_eos_to_prompt": is_fairseq2,  # For language llama, tell the runtime to always append EOS token(s) to prompt.
        "get_bos_id": 3 if is_fairseq2 else 1,
        "get_eos_ids": [3] if is_fairseq2 else [2],
        "get_max_seq_len": model_args.max_seq_len,
        "get_n_bos": 1,
        "get_n_eos": 2 if is_fairseq2 else 1,
        "get_n_layers": model_args.n_layers,
        "get_vocab_size": model_args.vocab_size,
        "use_kv_cache": use_kv_cache,
        "use_sdpa_with_kv_cache": use_sdpa_with_kv_cache,
        "enable_dynamic_shape": enable_dynamic_shape,
    }
    if metadata_str:
        try:
            extra = json.loads(metadata_str)
            for k, v in extra.items():
                metadata[k] = v
        except JSONDecodeError:
            logging.error("Invalid metadata, should be a valid JSON string")
    return metadata


def _load_llama_model(
    *,
    modelname: str = "llama2",
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    params_path: str,
    use_kv_cache: bool = False,
    use_sdpa_with_kv_cache: bool = False,
    generate_full_logits: bool = False,
    weight_type: WeightType = WeightType.LLAMA,
    enable_dynamic_shape: bool = False,
    calibration_tasks: Optional[List[str]] = None,
    calibration_limit: Optional[int] = None,
    calibration_seq_length: Optional[int] = None,
    calibration_data: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    verbose: bool = False,
    max_seq_len: int = 128,
    metadata_str: Optional[str] = None,
    args,
) -> "LLMEdgeManager":
    """
    A helper util that builds a Llama2 model. It returns a LLMEdgeManager that
    can help further lower the model to ExecuTorch.
    Returns:
        An instance of LLMEdgeManager which contains the eager mode model.
    """
    assert (
        checkpoint or checkpoint_dir
    ) and params_path, "Both checkpoint/checkpoint_dir and params can't be empty"
    logging.info(
        f"Loading model with checkpoint={checkpoint}, params={params_path}, use_kv_cache={use_kv_cache}, weight_type={weight_type}"
    )
    model, example_inputs, _ = EagerModelFactory.create_model(
        "llama2",
        "Llama2Model",
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        params=params_path,
        use_kv_cache=use_kv_cache,
        use_sdpa_with_kv_cache=use_sdpa_with_kv_cache,
        generate_full_logits=generate_full_logits,
        fairseq2=weight_type == WeightType.FAIRSEQ2,
        max_seq_len=max_seq_len,
        enable_dynamic_shape=enable_dynamic_shape,
        args=args,
    )
    state_dict = model.state_dict()
    dtype = state_dict[next(iter(state_dict))].dtype
    assert dtype in [
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ], f"Only support bfloat16, fp16 or fp32 got {dtype}"
    logging.info(f"Loaded model with dtype={dtype}")
    print(model)

    if dtype == torch.bfloat16:
        dtype = DType.bf16
    elif dtype == torch.float16:
        dtype = DType.fp16
    elif dtype == torch.float32:
        dtype = DType.fp32
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    return LLMEdgeManager(
        model=model,
        modelname=modelname,
        max_seq_len=model.params.max_seq_len,
        dtype=dtype,
        use_kv_cache=use_kv_cache,
        generate_full_logits=generate_full_logits,
        example_inputs=example_inputs,
        enable_dynamic_shape=enable_dynamic_shape,
        calibration_tasks=calibration_tasks,
        calibration_limit=calibration_limit,
        calibration_seq_length=calibration_seq_length,
        calibration_data=calibration_data,
        tokenizer_path=tokenizer_path,
        verbose=verbose,
        metadata=_load_llama_model_metadata(
            weight_type,
            use_kv_cache,
            use_sdpa_with_kv_cache,
            enable_dynamic_shape,
            model.params,
            metadata_str,
        ),
        args=args,
    )

import logging

import torch

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    modelname = "llama2"
    parser = build_args_parser()
    args = parser.parse_args()
    export_llama(modelname, args)


if __name__ == "__main__":
    main()  # pragma: no cover
