from __future__ import annotations

import logging
import random
import re
from pathlib import Path

import pandas as pd

from src.config.loader import PROJECT_ROOT
from src.config.runtime_settings import get_runtime_settings
from src.models.registry import get_model_logger
from src.utils.dataset_utils import clean_text_label_dataframe, normalize_text, read_jsonl, write_jsonl
from src.utils.run_utils import format_run_message


RUNTIME_SETTINGS = get_runtime_settings()
RAW_DATASET = RUNTIME_SETTINGS.data.raw_dataset
TRAIN_FILE = RUNTIME_SETTINGS.data.train_file
TEST_FILE = RUNTIME_SETTINGS.data.test_file
EXPANDED_TRAIN_FILE = RUNTIME_SETTINGS.data.expanded_train_file
AUGMENTED_ONLY_FILE = RUNTIME_SETTINGS.data.augmented_only_file

RANDOM_STATE = RUNTIME_SETTINGS.augmentation.random_state
MAX_VARIANTS_PER_TEXT = RUNTIME_SETTINGS.augmentation.max_variants_per_text
TARGET_STRATEGY = RUNTIME_SETTINGS.augmentation.target_strategy

PREFIXES = RUNTIME_SETTINGS.augmentation.prefixes
SUFFIXES = RUNTIME_SETTINGS.augmentation.suffixes
TOKEN_REPLACEMENTS = RUNTIME_SETTINGS.augmentation.token_replacements


def _replace_known_tokens(text: str, rng: random.Random) -> str:
    output = text
    for token, candidates in TOKEN_REPLACEMENTS.items():
        pattern = re.compile(rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
        if pattern.search(output):
            replacement = rng.choice(candidates)
            output = pattern.sub(replacement, output, count=1)
    return output


def _inject_typo(text: str, rng: random.Random) -> str:
    vowels = [i for i, c in enumerate(text) if c.lower() in "aeiouyàâéèêëîïôùûü"]
    if not vowels:
        return text
    idx = rng.choice(vowels)
    return text[:idx + 1] + text[idx] + text[idx + 1 :]


def _variants_for_text(text: str, rng: random.Random) -> list[str]:
    base = text.strip()
    variants = {
        base.lower(),
        base.upper(),
        re.sub(r"\s+", " ", base).strip(),
        base.replace("'", " "),
        _replace_known_tokens(base, rng),
        _inject_typo(base, rng),
        f"{rng.choice(PREFIXES)}{base}",
        f"{base}{rng.choice(SUFFIXES)}",
        f"{base} !!!",
    }

    cleaned = [v.strip() for v in variants if isinstance(v, str) and len(v.strip()) > 3]
    rng.shuffle(cleaned)
    return cleaned[:MAX_VARIANTS_PER_TEXT]


def _load_source_dataset(dataset: str | None) -> tuple[pd.DataFrame, Path]:
    source_path = Path(dataset) if dataset else TRAIN_FILE
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path

    if not source_path.exists():
        source_path = RAW_DATASET

    data = read_jsonl(source_path)
    cleaned = clean_text_label_dataframe(data)
    return cleaned, source_path


def _target_count_per_label(counts: pd.Series) -> int:
    if TARGET_STRATEGY == "balance_to_p90":
        return int(counts.quantile(0.9))
    return int(counts.max())


def expand_training_dataset(dataset: str | None = None, run_id: str | None = None) -> dict[str, str]:
    logger: logging.Logger = get_model_logger("classic")
    rng = random.Random(RANDOM_STATE)

    base_df, source_path = _load_source_dataset(dataset)
    logger.info(format_run_message(run_id, f"Expanding training dataset from {source_path}"))

    counts = base_df["label"].value_counts()
    target_count = _target_count_per_label(counts)

    generated_rows: list[dict[str, str]] = []

    for label, count in counts.items():
        needed = max(0, target_count - int(count))
        if needed == 0:
            continue

        label_texts = base_df.loc[base_df["label"] == label, "text"].tolist()
        pool: list[str] = []
        for text in label_texts:
            pool.extend(_variants_for_text(text, rng))

        pool = [p for p in dict.fromkeys(pool) if p not in set(label_texts)]
        if not pool:
            continue

        rng.shuffle(pool)
        selected = pool[:needed]

        for text in selected:
            generated_rows.append(
                {
                    "text": text,
                    "label": str(label),
                    "source": "augmentation_rule",
                }
            )

    augmented_df = pd.DataFrame(generated_rows)
    if augmented_df.empty:
        expanded_df = base_df.copy()
        expanded_df["source"] = "original"
    else:
        original_df = base_df.copy()
        original_df["source"] = "original"
        expanded_df = pd.concat([original_df, augmented_df], ignore_index=True)
        expanded_df = expanded_df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    write_jsonl(augmented_df if not augmented_df.empty else pd.DataFrame(columns=["text", "label", "source"]), AUGMENTED_ONLY_FILE)
    write_jsonl(expanded_df, EXPANDED_TRAIN_FILE)

    if TEST_FILE.exists() and not augmented_df.empty:
        test_df = read_jsonl(TEST_FILE)
        test_set = set(test_df["text"].map(normalize_text))
        aug_set = set(augmented_df["text"].map(normalize_text))
        overlap_count = len(test_set.intersection(aug_set))
        logger.info(format_run_message(run_id, f"Leakage check (exact normalized overlap with test): {overlap_count}"))

    if not augmented_df.empty:
        label_counts = augmented_df["label"].value_counts().to_dict()
        logger.info(format_run_message(run_id, f"Generated samples by label: {label_counts}"))

    logger.info(
        format_run_message(
            run_id,
            f"Expanded dataset rows={len(expanded_df)} (original={len(base_df)}, generated={len(augmented_df)})",
        )
    )
    logger.info(format_run_message(run_id, f"Saved augmented-only dataset to {AUGMENTED_ONLY_FILE}"))
    logger.info(format_run_message(run_id, f"Saved expanded training dataset to {EXPANDED_TRAIN_FILE}"))

    return {
        "augmented_only": str(AUGMENTED_ONLY_FILE),
        "expanded_train": str(EXPANDED_TRAIN_FILE),
    }
