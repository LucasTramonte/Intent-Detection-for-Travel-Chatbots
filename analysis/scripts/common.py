from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "intent-detection-train.jsonl"
OUTPUTS_DIR = PROJECT_ROOT / "analysis" / "outputs"


def ensure_outputs_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUTS_DIR


def load_dataset(path: str | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path else DEFAULT_DATASET
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    return pd.read_json(dataset_path, lines=True)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
