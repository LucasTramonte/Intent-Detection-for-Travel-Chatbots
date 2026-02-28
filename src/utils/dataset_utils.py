from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def clean_text_label_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "label"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    cleaned = data[["text", "label"]].copy()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned["label"] = cleaned["label"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"].str.len() > 0]
    cleaned = cleaned.dropna(subset=["text", "label"])
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).lower().strip())
