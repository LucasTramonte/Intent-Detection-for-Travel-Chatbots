from __future__ import annotations

import re

import pandas as pd

from common import PROJECT_ROOT, ensure_outputs_dir, load_dataset, write_json


PERTURBED_DATASET_PATH = PROJECT_ROOT / "data" / "intent-detection-test-perturbed.jsonl"


def perturb_text(text: str) -> list[str]:
    variants = {
        text,
        text.lower(),
        text.upper(),
        re.sub(r"\\s+", " ", text).strip(),
        text.replace("'", " "),
        f"{text} !!!",
        f"{text} svp",
        f"{text} urgent",
    }
    typo_variant = re.sub(r"([aeiou])", r"\\1\\1", text, count=1, flags=re.IGNORECASE)
    variants.add(typo_variant)
    return [value for value in variants if isinstance(value, str) and value.strip()]


def main() -> None:
    outputs_dir = ensure_outputs_dir()
    df = load_dataset()

    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        for variant in perturb_text(str(row["text"])):
            rows.append({"text": variant, "label": str(row["label"]), "source": "rule_perturb"})

    stress_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    PERTURBED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    stress_df.to_json(PERTURBED_DATASET_PATH, orient="records", lines=True, force_ascii=False)

    label_counts = {str(key): int(value) for key, value in stress_df["label"].value_counts().to_dict().items()}
    write_json(
        {
            "rows": int(len(stress_df)),
            "labels": label_counts,
            "dataset_path": str(PERTURBED_DATASET_PATH),
        },
        outputs_dir / "stress_test_summary.json",
    )


if __name__ == "__main__":
    main()
