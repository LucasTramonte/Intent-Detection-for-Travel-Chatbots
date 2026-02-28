from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.loader import PROJECT_ROOT, load_config
from src.config.runtime_settings import get_runtime_settings
from src.models.registry import get_model_logger
from src.utils.dataset_utils import clean_text_label_dataframe, read_jsonl, write_jsonl
from src.utils.run_utils import format_run_message


CONFIG = load_config()
RUNTIME_SETTINGS = get_runtime_settings()

RANDOM_STATE = RUNTIME_SETTINGS.evaluation.random_state
TEST_SIZE = RUNTIME_SETTINGS.evaluation.test_size
VAL_SIZE = RUNTIME_SETTINGS.evaluation.validation_size

RAW_DATASET = RUNTIME_SETTINGS.data.raw_dataset
SPLITS_DIR = RUNTIME_SETTINGS.data.splits_dir
TRAIN_FILE = RUNTIME_SETTINGS.data.train_file
VAL_FILE = RUNTIME_SETTINGS.data.validation_file
TEST_FILE = RUNTIME_SETTINGS.data.test_file

def _log_split_stats(logger: logging.Logger, run_id: str | None, split_name: str, df: pd.DataFrame) -> None:
    logger.info(
        format_run_message(
            run_id,
            f"{split_name}: rows={len(df)} intents={df['label'].nunique()} min_class={df['label'].value_counts().min()} max_class={df['label'].value_counts().max()}",
        )
    )


def prepare_dataset_splits(source_dataset: str | None = None, run_id: str | None = None) -> dict[str, str]:
    logger = get_model_logger("classic")
    dataset_path = Path(source_dataset) if source_dataset else RAW_DATASET
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    logger.info(format_run_message(run_id, f"Preparing dataset splits from {dataset_path}"))

    data = read_jsonl(dataset_path)
    data = clean_text_label_dataframe(data)

    train_df, test_df = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    val_relative_size = VAL_SIZE / (1.0 - TEST_SIZE)
    class_count = int(train_df["label"].nunique())
    min_val_relative = class_count / max(len(train_df), 1)
    effective_val_relative = max(val_relative_size, min_val_relative)

    if effective_val_relative >= 0.5:
        logger.warning(
            format_run_message(
                run_id,
                f"Validation split too constrained for stratification (effective={effective_val_relative:.3f}); using non-stratified split with test_size={val_relative_size:.3f}",
            )
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_relative_size,
            random_state=RANDOM_STATE,
            stratify=None,
        )
    else:
        if effective_val_relative > val_relative_size:
            logger.warning(
                format_run_message(
                    run_id,
                    f"Adjusted validation split from {val_relative_size:.3f} to {effective_val_relative:.3f} to keep at least one sample per class.",
                )
            )
        train_df, val_df = train_test_split(
            train_df,
            test_size=effective_val_relative,
            random_state=RANDOM_STATE,
            stratify=train_df["label"],
        )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    write_jsonl(train_df, TRAIN_FILE)
    write_jsonl(val_df, VAL_FILE)
    write_jsonl(test_df, TEST_FILE)

    _log_split_stats(logger, run_id, "train", train_df)
    _log_split_stats(logger, run_id, "validation", val_df)
    _log_split_stats(logger, run_id, "test", test_df)

    logger.info(format_run_message(run_id, f"Saved splits to {SPLITS_DIR}"))

    return {
        "train": str(TRAIN_FILE),
        "validation": str(VAL_FILE),
        "test": str(TEST_FILE),
    }
