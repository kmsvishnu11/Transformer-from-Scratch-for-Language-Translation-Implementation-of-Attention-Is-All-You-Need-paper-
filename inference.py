from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from config import ProjectConfig
from data.data_utils import Vocabulary, get_tokenizer, numericalize_sentence
from train import build_model
from utils import get_device, greedy_decode, load_checkpoint, plot_attention_weights


def load_model_and_assets(
    checkpoint_path: str | Path,
    device: torch.device,
):
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = ProjectConfig.from_dict(checkpoint["config"])
    config.train.model_impl = checkpoint.get("model_impl", config.train.model_impl)

    src_vocab = Vocabulary.from_dict(checkpoint["src_vocab"])
    tgt_vocab = Vocabulary.from_dict(checkpoint["tgt_vocab"])
    model = build_model(config, src_vocab=src_vocab, tgt_vocab=tgt_vocab).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, src_vocab, tgt_vocab


@torch.inference_mode()
def translate_sentence(
    sentence: str,
    model,
    config: ProjectConfig,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    max_len: int,
) -> tuple[str, torch.Tensor, List[str], List[str]]:
    src_tokenizer = get_tokenizer(config.data.src_lang, lowercase=config.data.lowercase)
    src_ids = numericalize_sentence(sentence, vocab=src_vocab, tokenizer=src_tokenizer)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    generated = greedy_decode(
        model=model,
        src_tokens=src_tensor,
        bos_idx=tgt_vocab.bos_index,
        eos_idx=tgt_vocab.eos_index,
        max_len=max_len,
    )
    prediction_ids = generated[0].tolist()
    translation_tokens = tgt_vocab.decode_ids(prediction_ids, remove_special_tokens=True)
    translation = " ".join(translation_tokens)

    src_display_tokens = src_vocab.decode_ids(src_ids, remove_special_tokens=False)
    decoder_inputs = generated[:, :-1]
    _, attention_dict = model(src_tensor, decoder_inputs, return_attn_weights=True)

    tgt_display_tokens = tgt_vocab.decode_ids(decoder_inputs[0].tolist(), remove_special_tokens=False)
    cross_attention = None
    if attention_dict["decoder_cross_attentions"]:
        cross_attention = attention_dict["decoder_cross_attentions"][-1].squeeze(0)

    return translation, cross_attention, src_display_tokens, tgt_display_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained Transformer checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved model checkpoint.")
    parser.add_argument("--sentence", type=str, default=None, help="Single source sentence to translate.")
    parser.add_argument("--input-file", type=str, default=None, help="Text file with one source sentence per line.")
    parser.add_argument("--output-attention", type=str, default=None, help="Optional path to save an attention plot for a single sentence.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-len", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sentence is None and args.input_file is None:
        raise ValueError("Provide either --sentence or --input-file.")

    device = get_device(args.device)
    model, config, src_vocab, tgt_vocab = load_model_and_assets(args.checkpoint, device=device)

    if args.sentence is not None:
        translation, cross_attention, src_tokens, tgt_tokens = translate_sentence(
            sentence=args.sentence,
            model=model,
            config=config,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            max_len=args.max_len,
        )
        print(f"Source ({config.data.src_lang}): {args.sentence}")
        print(f"Prediction ({config.data.tgt_lang}): {translation}")

        if args.output_attention is not None and cross_attention is not None:
            plot_attention_weights(
                attention_weights=cross_attention,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
                save_path=args.output_attention,
                title="Decoder Cross-Attention (Last Layer)",
            )
            print(f"Saved attention visualization to {args.output_attention}")

    if args.input_file is not None:
        input_path = Path(args.input_file)
        lines = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            translation, _, _, _ = translate_sentence(
                sentence=line,
                model=model,
                config=config,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                device=device,
                max_len=args.max_len,
            )
            print(f"SRC: {line}")
            print(f"TGT: {translation}")
            print("-" * 80)


if __name__ == "__main__":
    main()
