from __future__ import annotations

from common import ensure_outputs_dir, load_dataset, write_json


def main() -> None:
    outputs_dir = ensure_outputs_dir()
    df = load_dataset()

    counts = df["label"].value_counts()
    chars = df["text"].astype(str).str.len()
    tokens = df["text"].astype(str).str.split().str.len()

    summary = {
        "rows": int(len(df)),
        "intents": int(df["label"].nunique()),
        "avg_samples_per_intent": float(counts.mean()),
        "min_samples_per_intent": int(counts.min()),
        "max_samples_per_intent": int(counts.max()),
        "duplicate_rate": float(df.duplicated().mean()),
        "avg_chars": float(chars.mean()),
        "avg_tokens": float(tokens.mean()),
    }
    write_json(summary, outputs_dir / "dataset_summary.json")

    intent_distribution = {str(label): int(value) for label, value in counts.to_dict().items()}
    write_json(intent_distribution, outputs_dir / "intent_distribution.json")


if __name__ == "__main__":
    main()
