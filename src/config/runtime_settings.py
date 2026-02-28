from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.loader import PROJECT_ROOT, load_config


@dataclass(frozen=True)
class DataPaths:
    default_dataset: Path
    raw_dataset: Path
    splits_dir: Path
    train_file: Path
    validation_file: Path
    test_file: Path
    expanded_train_file: Path
    augmented_only_file: Path


@dataclass(frozen=True)
class EvaluationSettings:
    random_state: int
    test_size: float
    validation_size: float


@dataclass(frozen=True)
class AugmentationSettings:
    random_state: int
    max_variants_per_text: int
    target_strategy: str
    prefixes: tuple[str, ...]
    suffixes: tuple[str, ...]
    token_replacements: dict[str, list[str]]


@dataclass(frozen=True)
class RuntimeSettings:
    data: DataPaths
    evaluation: EvaluationSettings
    augmentation: AugmentationSettings


def get_runtime_settings() -> RuntimeSettings:
    config = load_config()
    paths_config = config.get("paths", {})
    data_config = config.get("data", {})
    eval_config = config.get("evaluation", {})
    aug_config = config.get("augmentation", {})

    splits_dir = PROJECT_ROOT / data_config.get("splits_dir", "data/splits")

    data = DataPaths(
        default_dataset=PROJECT_ROOT / paths_config.get("default_dataset", "data/intent-detection-train.jsonl"),
        raw_dataset=PROJECT_ROOT / data_config.get("raw_dataset", "data/intent-detection-train.jsonl"),
        splits_dir=splits_dir,
        train_file=splits_dir / data_config.get("train_file", "train.jsonl"),
        validation_file=splits_dir / data_config.get("validation_file", "validation.jsonl"),
        test_file=splits_dir / data_config.get("test_file", "test.jsonl"),
        expanded_train_file=splits_dir / data_config.get("expanded_train_file", "train_expanded.jsonl"),
        augmented_only_file=splits_dir / data_config.get("augmented_only_file", "train_augmented_only.jsonl"),
    )

    evaluation = EvaluationSettings(
        random_state=int(eval_config.get("random_state", 42)),
        test_size=float(eval_config.get("test_size", 0.2)),
        validation_size=float(data_config.get("validation_size", 0.1)),
    )

    augmentation = AugmentationSettings(
        random_state=int(aug_config.get("random_state", 42)),
        max_variants_per_text=int(aug_config.get("max_variants_per_text", 6)),
        target_strategy=str(aug_config.get("target_strategy", "balance_to_max")),
        prefixes=tuple(aug_config.get("prefixes", ["bonjour, ", "svp, ", "s'il vous plaît, ", "urgent: "])),
        suffixes=tuple(aug_config.get("suffixes", [" merci", " svp", " rapidement", " s'il vous plaît"])),
        token_replacements=dict(
            aug_config.get(
                "token_replacements",
                {
                    "vol": ["billet", "trajet"],
                    "hôtel": ["hotel", "hébergement"],
                    "bagage": ["valise"],
                    "traduire": ["traduction", "traduisez"],
                    "retard": ["décalage"],
                    "annulé": ["annule"],
                },
            )
        ),
    )

    return RuntimeSettings(data=data, evaluation=evaluation, augmentation=augmentation)
