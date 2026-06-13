from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class DataConfig:
    dataset_name: str = "bentrevett/multi30k"
    src_lang: str = "de"
    tgt_lang: str = "en"
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    min_freq: int = 2
    max_vocab_size: int | None = 30000
    max_length: int = 128
    lowercase: bool = True
    val_split: str = "validation"
    test_split: str = "test"
    cache_dir: str | None = None


@dataclass
class ModelConfig:
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_position_embeddings: int = 5000
    tie_output_projection: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 20
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    label_smoothing: float = 0.1
    lr_factor: float = 1.0
    warmup_steps: int = 4000
    clip_grad_norm: float = 1.0
    amp: bool = True
    device: str = "cuda"
    log_every: int = 100
    eval_every: int = 1
    max_decode_len: int = 128
    save_dir: str = "checkpoints"
    checkpoint_name: str = "transformer_scratch.pt"
    model_impl: str = "scratch"  # one of {scratch, torch}
    compare_with_torch_transformer: bool = False


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectConfig":
        return cls(
            data=DataConfig(**data.get("data", {})),
            model=ModelConfig(**data.get("model", {})),
            train=TrainConfig(**data.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def get_default_config() -> ProjectConfig:
    return ProjectConfig()
