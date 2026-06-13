from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import spacy
import torch
from datasets import DatasetDict, load_dataset
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import DataConfig

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


@dataclass
class Vocabulary:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[Sequence[str]],
        min_freq: int = 1,
        max_size: int | None = None,
        specials: Sequence[str] = SPECIAL_TOKENS,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for tokens in token_sequences:
            counter.update(tokens)

        sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        itos = list(specials)
        for token, frequency in sorted_tokens:
            if token in specials:
                continue
            if frequency < min_freq:
                continue
            if max_size is not None and len(itos) >= max_size:
                break
            itos.append(token)

        stoi = {token: index for index, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def from_dict(cls, data: Dict[str, List[str] | Dict[str, int]]) -> "Vocabulary":
        return cls(stoi=dict(data["stoi"]), itos=list(data["itos"]))

    def to_dict(self) -> Dict[str, List[str] | Dict[str, int]]:
        return {"stoi": self.stoi, "itos": self.itos}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_index(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_index(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def bos_index(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_index(self) -> int:
        return self.stoi[EOS_TOKEN]

    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, self.unk_index)

    def id_to_token(self, index: int) -> str:
        return self.itos[index]

    def encode_tokens(self, tokens: Sequence[str], add_special_tokens: bool = True) -> List[int]:
        token_ids = [self.token_to_id(token) for token in tokens]
        if add_special_tokens:
            token_ids = [self.bos_index] + token_ids + [self.eos_index]
        return token_ids

    def decode_ids(self, token_ids: Sequence[int], remove_special_tokens: bool = True) -> List[str]:
        tokens: List[str] = []
        for index in token_ids:
            if index < 0 or index >= len(self.itos):
                continue
            token = self.id_to_token(index)
            if remove_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return tokens


class TranslationDataset(Dataset[Tuple[List[int], List[int]]]):
    def __init__(
        self,
        split,
        src_lang: str,
        tgt_lang: str,
        src_tokenizer: Callable[[str], List[str]],
        tgt_tokenizer: Callable[[str], List[str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_length: int | None = None,
    ) -> None:
        self.examples: List[Tuple[List[int], List[int]]] = []

        for example in split:
            src_text, tgt_text = extract_translation_pair(example, src_lang=src_lang, tgt_lang=tgt_lang)
            src_tokens = src_tokenizer(src_text)
            tgt_tokens = tgt_tokenizer(tgt_text)

            if max_length is not None:
                if len(src_tokens) + 2 > max_length or len(tgt_tokens) + 2 > max_length:
                    continue

            src_ids = src_vocab.encode_tokens(src_tokens, add_special_tokens=True)
            tgt_ids = tgt_vocab.encode_tokens(tgt_tokens, add_special_tokens=True)
            self.examples.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.examples[index]


def get_tokenizer(language: str, lowercase: bool = True) -> Callable[[str], List[str]]:
    """
    Uses spaCy's rule-based tokenizer. `spacy.blank` avoids downloading large
    language models while still providing strong tokenization behavior.
    """
    nlp = spacy.blank(language)

    def tokenize(text: str) -> List[str]:
        text = text.strip()
        if lowercase:
            text = text.lower()
        return [token.text for token in nlp.tokenizer(text)]

    return tokenize


def extract_translation_pair(example: Dict, src_lang: str, tgt_lang: str) -> Tuple[str, str]:
    if src_lang in example and tgt_lang in example:
        return str(example[src_lang]), str(example[tgt_lang])

    if "translation" in example:
        translation = example["translation"]
        return str(translation[src_lang]), str(translation[tgt_lang])

    raise KeyError(
        f"Could not find translation fields for languages '{src_lang}' and '{tgt_lang}'."
    )


def load_translation_splits(config: DataConfig) -> DatasetDict:
    dataset = load_dataset(config.dataset_name, cache_dir=config.cache_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected a Hugging Face DatasetDict with train/validation/test splits.")
    return dataset


def build_vocabularies(
    train_split,
    src_lang: str,
    tgt_lang: str,
    src_tokenizer: Callable[[str], List[str]],
    tgt_tokenizer: Callable[[str], List[str]],
    min_freq: int,
    max_size: int | None,
) -> Tuple[Vocabulary, Vocabulary]:
    src_token_sequences: List[List[str]] = []
    tgt_token_sequences: List[List[str]] = []

    for example in train_split:
        src_text, tgt_text = extract_translation_pair(example, src_lang=src_lang, tgt_lang=tgt_lang)
        src_token_sequences.append(src_tokenizer(src_text))
        tgt_token_sequences.append(tgt_tokenizer(tgt_text))

    src_vocab = Vocabulary.build(
        token_sequences=src_token_sequences,
        min_freq=min_freq,
        max_size=max_size,
    )
    tgt_vocab = Vocabulary.build(
        token_sequences=tgt_token_sequences,
        min_freq=min_freq,
        max_size=max_size,
    )
    return src_vocab, tgt_vocab


def collate_translation_batch(
    batch: Sequence[Tuple[List[int], List[int]]],
    src_pad_idx: int,
    tgt_pad_idx: int,
) -> Tuple[Tensor, Tensor]:
    src_batch = [torch.tensor(src_ids, dtype=torch.long) for src_ids, _ in batch]
    tgt_batch = [torch.tensor(tgt_ids, dtype=torch.long) for _, tgt_ids in batch]

    src_batch_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_batch_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
    return src_batch_padded, tgt_batch_padded


def build_dataloaders(
    config: DataConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    dataset = load_translation_splits(config)
    src_tokenizer = get_tokenizer(config.src_lang, lowercase=config.lowercase)
    tgt_tokenizer = get_tokenizer(config.tgt_lang, lowercase=config.lowercase)

    src_vocab, tgt_vocab = build_vocabularies(
        train_split=dataset["train"],
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        min_freq=config.min_freq,
        max_size=config.max_vocab_size,
    )

    train_dataset = TranslationDataset(
        split=dataset["train"],
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_length=config.max_length,
    )
    val_dataset = TranslationDataset(
        split=dataset[config.val_split],
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_length=config.max_length,
    )
    test_dataset = TranslationDataset(
        split=dataset[config.test_split],
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_length=config.max_length,
    )

    collate_fn = partial(
        collate_translation_batch,
        src_pad_idx=src_vocab.pad_index,
        tgt_pad_idx=tgt_vocab.pad_index,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


def numericalize_sentence(
    sentence: str,
    vocab: Vocabulary,
    tokenizer: Callable[[str], List[str]],
) -> List[int]:
    return vocab.encode_tokens(tokenizer(sentence), add_special_tokens=True)
