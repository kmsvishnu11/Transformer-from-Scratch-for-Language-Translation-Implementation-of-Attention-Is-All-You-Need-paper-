from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_preference: str = "cuda") -> torch.device:
    if device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@dataclass
class AverageMeter:
    value: float = 0.0
    total: float = 0.0
    count: int = 0

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.total += value * n
        self.count += n


class NoamScheduler:
    """
    Learning-rate schedule from Section 5.3 of the paper:

        lr = factor * d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def rate(self, step: int | None = None) -> float:
        step = self._step if step is None else step
        step = max(step, 1)
        return self.factor * (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5),
        )

    def step(self) -> float:
        self._step += 1
        learning_rate = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate
        return learning_rate

    def state_dict(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "factor": self.factor,
            "step": self._step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.d_model = int(state_dict["d_model"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.factor = float(state_dict["factor"])
        self._step = int(state_dict["step"])
        current_lr = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as used in Section 5.4 of the paper.

    This implementation follows the common Transformer formulation where the
    smoothed target distribution excludes the padding token.
    """

    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1).")
        if vocab_size <= 2:
            raise ValueError("vocab_size must be greater than 2 for label smoothing.")

        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        target = target.long()

        with torch.no_grad():
            true_dist = torch.full_like(
                log_probs,
                fill_value=self.smoothing / (self.vocab_size - 2),
            )
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0.0

            padding_positions = target.eq(self.padding_idx)
            true_dist.masked_fill_(padding_positions.unsqueeze(1), 0.0)

        loss = F.kl_div(log_probs, true_dist, reduction="sum")
        normalizer = target.ne(self.padding_idx).sum().clamp_min(1)
        return loss / normalizer


def save_checkpoint(path: str | Path, checkpoint: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@torch.inference_mode()
def greedy_decode(
    model: nn.Module,
    src_tokens: Tensor,
    bos_idx: int,
    eos_idx: int,
    max_len: int,
) -> Tensor:
    model.eval()

    src_mask = model.make_src_mask(src_tokens)
    memory, _ = model.encode(src_tokens, src_mask=src_mask, return_attn_weights=False)

    generated = torch.full(
        (src_tokens.size(0), 1),
        fill_value=bos_idx,
        dtype=torch.long,
        device=src_tokens.device,
    )
    finished = torch.zeros(src_tokens.size(0), dtype=torch.bool, device=src_tokens.device)

    for _ in range(max_len - 1):
        tgt_mask = model.make_tgt_mask(generated)
        decoder_output, _, _ = model.decode(
            tgt_tokens=generated,
            memory=memory,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            return_attn_weights=False,
        )
        next_token_logits = model.generator(decoder_output[:, -1, :])
        next_tokens = next_token_logits.argmax(dim=-1)

        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, eos_idx),
            next_tokens,
        )
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        finished = finished | next_tokens.eq(eos_idx)

        if finished.all():
            break

    return generated


def sequence_to_text(token_ids: Sequence[int], vocab) -> str:
    return " ".join(vocab.decode_ids(token_ids, remove_special_tokens=True))


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return float(bleu.score)


def plot_attention_weights(
    attention_weights: Tensor,
    src_tokens: Sequence[str],
    tgt_tokens: Sequence[str],
    save_path: str | Path,
    title: str = "Attention Weights",
) -> None:
    """
    Visualizes attention weights.

    Args:
        attention_weights:
            Either [num_heads, tgt_len, src_len] or [tgt_len, src_len].
    """
    attention = attention_weights.detach().cpu()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if attention.dim() == 2:
        attention = attention.unsqueeze(0)

    num_heads = attention.size(0)
    num_cols = min(4, num_heads)
    num_rows = math.ceil(num_heads / num_cols)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4.5 * num_cols, 4.0 * num_rows),
        squeeze=False,
    )
    fig.suptitle(title)

    for head_index in range(num_rows * num_cols):
        axis = axes[head_index // num_cols][head_index % num_cols]
        if head_index >= num_heads:
            axis.axis("off")
            continue

        image = axis.imshow(attention[head_index].numpy(), aspect="auto", cmap="viridis")
        axis.set_title(f"Head {head_index}")
        axis.set_xticks(range(len(src_tokens)))
        axis.set_xticklabels(src_tokens, rotation=90)
        axis.set_yticks(range(len(tgt_tokens)))
        axis.set_yticklabels(tgt_tokens)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
