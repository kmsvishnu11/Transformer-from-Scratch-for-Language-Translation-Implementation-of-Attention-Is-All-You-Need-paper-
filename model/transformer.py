from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from model.attention import MultiHeadAttention
from model.positional_encoding import PositionalEncoding


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network from Section 3.3 of the paper.

    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Encoder layer from Section 3.1.

    Note: This implementation follows the original paper's post-norm layout:
        LayerNorm(x + Sublayer(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        src_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        attn_output, attn_weights = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x, attn_weights if return_attn_weights else None


class TransformerDecoderLayer(nn.Module):
    """
    Decoder layer from Section 3.1.

    Contains:
      1. Masked self-attention
      2. Encoder-decoder attention
      3. Position-wise feed-forward network
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        cross_attn_output, cross_attn_weights = self.cross_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        if return_attn_weights:
            return x, self_attn_weights, cross_attn_weights
        return x, None, None


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        src_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, List[Tensor]]:
        attention_maps: List[Tensor] = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask=src_mask, return_attn_weights=return_attn_weights)
            if attn_weights is not None:
                attention_maps.append(attn_weights)
        return x, attention_maps


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        self_attention_maps: List[Tensor] = []
        cross_attention_maps: List[Tensor] = []

        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x=x,
                memory=memory,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                return_attn_weights=return_attn_weights,
            )
            if self_attn_weights is not None:
                self_attention_maps.append(self_attn_weights)
            if cross_attn_weights is not None:
                cross_attention_maps.append(cross_attn_weights)

        return x, self_attention_maps, cross_attention_maps


class Transformer(nn.Module):
    """
    Full encoder-decoder Transformer from Sections 3.1-3.4 of the paper.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        tie_output_projection: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len,
        )

        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)

        if tie_output_projection:
            self.generator.weight = self.tgt_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

        if self.src_embedding.padding_idx is not None:
            with torch.no_grad():
                self.src_embedding.weight[self.src_embedding.padding_idx].zero_()
        if self.tgt_embedding.padding_idx is not None:
            with torch.no_grad():
                self.tgt_embedding.weight[self.tgt_embedding.padding_idx].zero_()

    def make_src_mask(self, src_tokens: Tensor) -> Tensor:
        """
        Returns a mask of shape [batch_size, 1, 1, src_len]
        where True indicates valid (non-pad) source positions.
        """
        return (src_tokens != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt_tokens: Tensor) -> Tensor:
        """
        Returns a combined padding + causal mask of shape
        [batch_size, 1, tgt_len, tgt_len].
        """
        batch_size, tgt_len = tgt_tokens.shape
        pad_mask = (tgt_tokens != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt_tokens.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(1)
        return pad_mask & causal_mask.expand(batch_size, -1, -1, -1)

    def encode(
        self,
        src_tokens: Tensor,
        src_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, List[Tensor]]:
        if src_mask is None:
            src_mask = self.make_src_mask(src_tokens)

        x = self.src_embedding(src_tokens) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.encoder(x, src_mask=src_mask, return_attn_weights=return_attn_weights)

    def decode(
        self,
        tgt_tokens: Tensor,
        memory: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt_tokens)

        x = self.tgt_embedding(tgt_tokens) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        return self.decoder(
            x=x,
            memory=memory,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            return_attn_weights=return_attn_weights,
        )

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, Dict[str, List[Tensor]]]:
        if src_mask is None:
            src_mask = self.make_src_mask(src_tokens)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt_tokens)

        memory, encoder_attentions = self.encode(
            src_tokens=src_tokens,
            src_mask=src_mask,
            return_attn_weights=return_attn_weights,
        )
        decoder_output, decoder_self_attentions, decoder_cross_attentions = self.decode(
            tgt_tokens=tgt_tokens,
            memory=memory,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            return_attn_weights=return_attn_weights,
        )
        logits = self.generator(decoder_output)

        attention_dict = {
            "encoder_self_attentions": encoder_attentions,
            "decoder_self_attentions": decoder_self_attentions,
            "decoder_cross_attentions": decoder_cross_attentions,
        }
        return logits, attention_dict
