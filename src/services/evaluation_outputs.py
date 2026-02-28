from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json

from src.config.loader import PROJECT_ROOT, load_config


CONFIG = load_config()
OUTPUTS_DIR = PROJECT_ROOT / CONFIG.get("paths", {}).get("outputs_dir", "outputs")


def write_evaluation_artifacts(
    model_name: str,
    run_id: str | None,
    split_summary: dict[str, Any],
    metrics_summary: dict[str, Any],
    classification_report_dict: dict[str, Any],
    confusion_matrix: list[list[int]],
    predictions: list[dict[str, Any]] | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_suffix = f"-{run_id}" if run_id else ""
    output_dir = OUTPUTS_DIR / "evaluations" / model_name / f"{timestamp}{run_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "model": model_name,
        "timestamp": timestamp,
        "run_id": run_id,
        "split": split_summary,
        "metrics": metrics_summary,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_dir / "classification_report.json").write_text(
        json.dumps(classification_report_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (output_dir / "confusion_matrix.json").write_text(
        json.dumps(confusion_matrix, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if predictions:
        (output_dir / "predictions.json").write_text(
            json.dumps(predictions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return output_dir
