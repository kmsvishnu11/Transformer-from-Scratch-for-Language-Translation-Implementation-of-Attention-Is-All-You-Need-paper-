from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention from Section 3.2.1 of
    "Attention Is All You Need" (Vaswani et al., 2017).

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query: Tensor of shape [batch_size, num_heads, query_len, head_dim]
            key: Tensor of shape [batch_size, num_heads, key_len, head_dim]
            value: Tensor of shape [batch_size, num_heads, key_len, head_dim]
            mask: Optional boolean tensor broadcastable to
                [batch_size, num_heads, query_len, key_len].
                True denotes a valid position and False denotes a masked position.

        Returns:
            output: Tensor of shape [batch_size, num_heads, query_len, head_dim]
            attention_weights: Tensor of shape
                [batch_size, num_heads, query_len, key_len]
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention from Section 3.2.2 of the paper.

    The inputs are linearly projected h times into smaller subspaces,
    attention is computed in parallel, and the resulting heads are
    concatenated and projected back to d_model.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query: [batch_size, query_len, d_model]
            key: [batch_size, key_len, d_model]
            value: [batch_size, key_len, d_model]
            mask: Optional mask broadcastable to
                [batch_size, num_heads, query_len, key_len]

        Returns:
            attention_output: [batch_size, query_len, d_model]
            attention_weights: [batch_size, num_heads, query_len, key_len]
        """
        query = self._split_heads(self.query_proj(query))
        key = self._split_heads(self.key_proj(key))
        value = self._split_heads(self.value_proj(value))

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        attention_output, attention_weights = self.attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
        )
        attention_output = self._combine_heads(attention_output)
        attention_output = self.out_proj(attention_output)
        return attention_output, attention_weights
