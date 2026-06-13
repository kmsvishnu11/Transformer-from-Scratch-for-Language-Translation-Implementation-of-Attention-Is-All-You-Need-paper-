from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from config import ProjectConfig, get_default_config
from data.data_utils import Vocabulary, build_dataloaders
from model.positional_encoding import PositionalEncoding
from model.transformer import Transformer
from utils import (
    AverageMeter,
    LabelSmoothingLoss,
    NoamScheduler,
    compute_bleu,
    count_parameters,
    get_device,
    greedy_decode,
    save_checkpoint,
    save_json,
    sequence_to_text,
    set_seed,
)


class TorchTransformerBaseline(nn.Module):
    """
    Thin wrapper around torch.nn.Transformer for apples-to-apples comparison.
    This is not used by the main from-scratch implementation.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        tie_output_projection: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)

        if tie_output_projection:
            self.generator.weight = self.tgt_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.src_embedding.weight[self.src_pad_idx].zero_()
            self.tgt_embedding.weight[self.tgt_pad_idx].zero_()

    def make_src_mask(self, src_tokens: Tensor) -> Tensor:
        return (src_tokens != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt_tokens: Tensor) -> Tensor:
        batch_size, tgt_len = tgt_tokens.shape
        pad_mask = (tgt_tokens != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt_tokens.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(1)
        return pad_mask & causal_mask.expand(batch_size, -1, -1, -1)

    def _causal_mask_for_torch(self, tgt_len: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones((tgt_len, tgt_len), device=device, dtype=torch.bool),
            diagonal=1,
        )

    def encode(
        self,
        src_tokens: Tensor,
        src_mask: Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, list]:
        del return_attn_weights
        src_key_padding_mask = src_tokens.eq(self.src_pad_idx)
        src_embeddings = self.positional_encoding(self.src_embedding(src_tokens) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
        return memory, []

    def decode(
        self,
        tgt_tokens: Tensor,
        memory: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, list, list]:
        del tgt_mask, return_attn_weights
        tgt_embeddings = self.positional_encoding(self.tgt_embedding(tgt_tokens) * math.sqrt(self.d_model))
        tgt_key_padding_mask = tgt_tokens.eq(self.tgt_pad_idx)

        memory_key_padding_mask = None
        if src_mask is not None:
            memory_key_padding_mask = ~src_mask.squeeze(1).squeeze(1)
        causal_mask = self._causal_mask_for_torch(tgt_tokens.size(1), tgt_tokens.device)

        output = self.transformer.decoder(
            tgt=tgt_embeddings,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return output, [], []

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, Dict[str, list]]:
        memory, _ = self.encode(src_tokens, src_mask=src_mask, return_attn_weights=return_attn_weights)
        decoder_output, _, _ = self.decode(
            tgt_tokens=tgt_tokens,
            memory=memory,
            src_mask=src_mask if src_mask is not None else self.make_src_mask(src_tokens),
            tgt_mask=tgt_mask,
            return_attn_weights=return_attn_weights,
        )
        logits = self.generator(decoder_output)
        return logits, {
            "encoder_self_attentions": [],
            "decoder_self_attentions": [],
            "decoder_cross_attentions": [],
        }


def build_model(
    config: ProjectConfig,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
) -> nn.Module:
    common_kwargs = dict(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_idx=src_vocab.pad_index,
        tgt_pad_idx=tgt_vocab.pad_index,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        max_len=config.model.max_position_embeddings,
        tie_output_projection=config.model.tie_output_projection,
    )

    if config.train.model_impl == "scratch":
        return Transformer(**common_kwargs)
    if config.train.model_impl == "torch":
        return TorchTransformerBaseline(**common_kwargs)

    raise ValueError("config.train.model_impl must be one of {'scratch', 'torch'}")


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    clip_grad_norm: float,
    log_every: int,
    pad_idx: int,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    token_meter = AverageMeter()

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for step, (src_tokens, tgt_tokens) in enumerate(progress_bar, start=1):
        src_tokens = src_tokens.to(device, non_blocking=True)
        tgt_tokens = tgt_tokens.to(device, non_blocking=True)

        decoder_inputs = tgt_tokens[:, :-1]
        labels = tgt_tokens[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(src_tokens, decoder_inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        learning_rate = scheduler.step()

        non_pad_tokens = int(labels.ne(pad_idx).sum().item())
        loss_meter.update(loss.item(), n=non_pad_tokens)
        token_meter.update(float(non_pad_tokens), n=1)

        if step % log_every == 0 or step == len(dataloader):
            progress_bar.set_postfix(
                loss=f"{loss_meter.average:.4f}",
                lr=f"{learning_rate:.6f}",
                grad_norm=f"{float(grad_norm):.2f}",
            )

    return {
        "loss": loss_meter.average,
        "tokens_per_batch": token_meter.average,
    }


@torch.inference_mode()
def evaluate_loss(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int,
) -> float:
    model.eval()
    loss_meter = AverageMeter()

    for src_tokens, tgt_tokens in tqdm(dataloader, desc="Validation loss", leave=False):
        src_tokens = src_tokens.to(device, non_blocking=True)
        tgt_tokens = tgt_tokens.to(device, non_blocking=True)

        decoder_inputs = tgt_tokens[:, :-1]
        labels = tgt_tokens[:, 1:]
        logits, _ = model(src_tokens, decoder_inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        non_pad_tokens = int(labels.ne(pad_idx).sum().item())
        loss_meter.update(loss.item(), n=non_pad_tokens)

    return loss_meter.average


@torch.inference_mode()
def evaluate_bleu(
    model: nn.Module,
    dataloader,
    tgt_vocab: Vocabulary,
    device: torch.device,
    max_decode_len: int,
) -> float:
    model.eval()
    predictions = []
    references = []

    for src_tokens, tgt_tokens in tqdm(dataloader, desc="BLEU", leave=False):
        src_tokens = src_tokens.to(device, non_blocking=True)
        generated = greedy_decode(
            model=model,
            src_tokens=src_tokens,
            bos_idx=tgt_vocab.bos_index,
            eos_idx=tgt_vocab.eos_index,
            max_len=max_decode_len,
        )

        for predicted_ids, reference_ids in zip(generated.tolist(), tgt_tokens.tolist()):
            predictions.append(sequence_to_text(predicted_ids, tgt_vocab))
            references.append(sequence_to_text(reference_ids, tgt_vocab))

    return compute_bleu(predictions, references)


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: ProjectConfig,
) -> Tuple[torch.optim.Optimizer, NoamScheduler]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        eps=config.train.adam_eps,
    )
    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=config.model.d_model,
        warmup_steps=config.train.warmup_steps,
        factor=config.train.lr_factor,
    )
    return optimizer, scheduler


def run_experiment(config: ProjectConfig) -> Dict[str, Any]:
    set_seed(config.train.seed)
    device = get_device(config.train.device)
    use_amp = config.train.amp and device.type == "cuda"

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(config.data)
    model = build_model(config, src_vocab=src_vocab, tgt_vocab=tgt_vocab).to(device)

    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        padding_idx=tgt_vocab.pad_index,
        smoothing=config.train.label_smoothing,
    )

    experiment_dir = Path(config.train.save_dir) / config.train.model_impl
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config.save(experiment_dir / f"config_{config.train.model_impl}.json")

    print(f"Device: {device}")
    print(f"Model implementation: {config.train.model_impl}")
    print(f"Train examples: {len(train_loader.dataset):,}")
    print(f"Validation examples: {len(val_loader.dataset):,}")
    print(f"Test examples: {len(test_loader.dataset):,}")
    print(f"Source vocab size: {len(src_vocab):,}")
    print(f"Target vocab size: {len(tgt_vocab):,}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    best_val_bleu = float("-inf")
    best_checkpoint_path = experiment_dir / config.train.checkpoint_name
    history = []

    for epoch in range(1, config.train.epochs + 1):
        print(f"\nEpoch {epoch}/{config.train.epochs}")
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            clip_grad_norm=config.train.clip_grad_norm,
            log_every=config.train.log_every,
            pad_idx=tgt_vocab.pad_index,
            use_amp=use_amp,
        )
        val_loss = evaluate_loss(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            pad_idx=tgt_vocab.pad_index,
        )

        epoch_metrics: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        if epoch % config.train.eval_every == 0:
            val_bleu = evaluate_bleu(
                model=model,
                dataloader=val_loader,
                tgt_vocab=tgt_vocab,
                device=device,
                max_decode_len=config.train.max_decode_len,
            )
            epoch_metrics["val_bleu"] = val_bleu
            print(
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_loss:.4f} | val_bleu={val_bleu:.2f}"
            )

            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                save_checkpoint(
                    best_checkpoint_path,
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": config.to_dict(),
                        "src_vocab": src_vocab.to_dict(),
                        "tgt_vocab": tgt_vocab.to_dict(),
                        "metrics": epoch_metrics,
                        "model_impl": config.train.model_impl,
                    },
                )
        else:
            print(f"train_loss={train_metrics['loss']:.4f} | val_loss={val_loss:.4f}")

        history.append(epoch_metrics)
        save_json(experiment_dir / f"history_{config.train.model_impl}.json", {"history": history})

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_bleu = evaluate_bleu(
        model=model,
        dataloader=test_loader,
        tgt_vocab=tgt_vocab,
        device=device,
        max_decode_len=config.train.max_decode_len,
    )
    test_loss = evaluate_loss(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        pad_idx=tgt_vocab.pad_index,
    )

    results = {
        "model_impl": config.train.model_impl,
        "checkpoint_path": str(best_checkpoint_path),
        "best_val_bleu": best_val_bleu,
        "test_bleu": test_bleu,
        "test_loss": test_loss,
        "num_parameters": count_parameters(model),
    }
    save_json(experiment_dir / f"results_{config.train.model_impl}.json", results)

    print(
        f"\nFinished {config.train.model_impl} run | "
        f"best_val_bleu={best_val_bleu:.2f} | test_bleu={test_bleu:.2f} | test_loss={test_loss:.4f}"
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer from scratch for translation.")
    parser.add_argument("--model-impl", choices=["scratch", "torch"], default="scratch")
    parser.add_argument("--compare-with-torch-transformer", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--src-lang", type=str, default=None)
    parser.add_argument("--tgt-lang", type=str, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-encoder-layers", type=int, default=None)
    parser.add_argument("--num-decoder-layers", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-decode-len", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def apply_overrides(config: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    config.train.model_impl = args.model_impl
    config.train.compare_with_torch_transformer = args.compare_with_torch_transformer

    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.dataset_name is not None:
        config.data.dataset_name = args.dataset_name
    if args.src_lang is not None:
        config.data.src_lang = args.src_lang
    if args.tgt_lang is not None:
        config.data.tgt_lang = args.tgt_lang
    if args.d_model is not None:
        config.model.d_model = args.d_model
    if args.num_heads is not None:
        config.model.num_heads = args.num_heads
    if args.num_encoder_layers is not None:
        config.model.num_encoder_layers = args.num_encoder_layers
    if args.num_decoder_layers is not None:
        config.model.num_decoder_layers = args.num_decoder_layers
    if args.d_ff is not None:
        config.model.d_ff = args.d_ff
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.warmup_steps is not None:
        config.train.warmup_steps = args.warmup_steps
    if args.label_smoothing is not None:
        config.train.label_smoothing = args.label_smoothing
    if args.max_length is not None:
        config.data.max_length = args.max_length
    if args.max_decode_len is not None:
        config.train.max_decode_len = args.max_decode_len
    if args.save_dir is not None:
        config.train.save_dir = args.save_dir
    if args.checkpoint_name is not None:
        config.train.checkpoint_name = args.checkpoint_name
    if args.seed is not None:
        config.train.seed = args.seed
    if args.device is not None:
        config.train.device = args.device
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.no_amp:
        config.train.amp = False

    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(get_default_config(), args)
    primary_results = run_experiment(config)

    if args.compare_with_torch_transformer and config.train.model_impl == "scratch":
        baseline_config = copy.deepcopy(config)
        baseline_config.train.model_impl = "torch"
        baseline_config.train.checkpoint_name = "transformer_torch_baseline.pt"
        baseline_results = run_experiment(baseline_config)

        comparison = {
            "scratch": primary_results,
            "torch": baseline_results,
        }
        save_json(Path(config.train.save_dir) / "comparison.json", comparison)
        print("\nSaved comparison results to comparison.json")


if __name__ == "__main__":
    main()
