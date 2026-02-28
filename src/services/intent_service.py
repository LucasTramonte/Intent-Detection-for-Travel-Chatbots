from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.config.loader import PROJECT_ROOT
from src.config.runtime_settings import get_runtime_settings
from src.models.registry import create_model, get_model_logger
from src.services.evaluation_outputs import write_evaluation_artifacts
from src.utils.dataset_utils import read_jsonl
from src.utils.run_utils import format_run_message


RUNTIME_SETTINGS = get_runtime_settings()
DEFAULT_DATASET = RUNTIME_SETTINGS.data.default_dataset
RANDOM_STATE = RUNTIME_SETTINGS.evaluation.random_state
TEST_SIZE = RUNTIME_SETTINGS.evaluation.test_size

SPLITS_DIR = RUNTIME_SETTINGS.data.splits_dir
TRAIN_SPLIT_PATH = RUNTIME_SETTINGS.data.train_file
TEST_SPLIT_PATH = RUNTIME_SETTINGS.data.test_file

def resolve_dataset_path(dataset: str | None) -> Path:
    if dataset is None:
        if TRAIN_SPLIT_PATH.exists():
            return TRAIN_SPLIT_PATH
        return DEFAULT_DATASET

    path = Path(dataset)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def split_train_test(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "label" not in data.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    train_df, test_df = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_train_test_frames(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if dataset_path.resolve() == TRAIN_SPLIT_PATH.resolve() and TEST_SPLIT_PATH.exists():
        train_df = read_jsonl(TRAIN_SPLIT_PATH)
        test_df = read_jsonl(TEST_SPLIT_PATH)
        return train_df, test_df, "prepared_splits"

    if dataset_path.parent.resolve() == SPLITS_DIR.resolve() and TEST_SPLIT_PATH.exists() and dataset_path.name.startswith("train"):
        train_df = read_jsonl(dataset_path)
        test_df = read_jsonl(TEST_SPLIT_PATH)
        return train_df, test_df, "prepared_test_with_custom_train"

    data = read_jsonl(dataset_path)
    train_df, test_df = split_train_test(data)
    return train_df, test_df, "runtime_split"


def load_eval_test_frame(dataset_path: Path) -> tuple[pd.DataFrame, int | None, str]:
    if dataset_path.resolve() == TRAIN_SPLIT_PATH.resolve() and TEST_SPLIT_PATH.exists():
        train_rows = int(len(read_jsonl(TRAIN_SPLIT_PATH)))
        test_df = read_jsonl(TEST_SPLIT_PATH)
        return test_df, train_rows, "prepared_splits"

    if dataset_path.resolve() == TEST_SPLIT_PATH.resolve():
        train_rows = int(len(read_jsonl(TRAIN_SPLIT_PATH))) if TRAIN_SPLIT_PATH.exists() else None
        test_df = read_jsonl(TEST_SPLIT_PATH)
        return test_df, train_rows, "prepared_test_only"

    data = read_jsonl(dataset_path)
    train_df, test_df = split_train_test(data)
    return test_df, int(len(train_df)), "runtime_split"


def _metrics_payload(y_true, y_pred) -> tuple[dict[str, Any], dict[str, float], list[list[int]]]:
    report_raw = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
    report_dict: dict[str, Any] = report_raw if isinstance(report_raw, dict) else {}
    matrix = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=1)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=1)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=1)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=1)),
    }
    return report_dict, metrics, matrix.tolist()


def _build_prediction_rows(texts: list[str], y_true_labels: list[str], y_pred_labels: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text, y_true_label, y_pred_label in zip(texts, y_true_labels, y_pred_labels):
        rows.append(
            {
                "text": text,
                "true_label": y_true_label,
                "predicted_label": y_pred_label,
                "is_correct": bool(y_true_label == y_pred_label),
            }
        )
    return rows


def _save_outputs_evaluation(
    model_name: str,
    logger: logging.Logger,
    run_id: str | None,
    split_summary: dict[str, Any],
    test_texts: list[str],
    y_true_labels: list[str],
    y_pred_labels: list[str],
) -> None:
    report_dict, metrics_summary, matrix = _metrics_payload(y_true_labels, y_pred_labels)
    prediction_rows = _build_prediction_rows(test_texts, y_true_labels, y_pred_labels)

    output_dir = write_evaluation_artifacts(
        model_name=model_name,
        run_id=run_id,
        split_summary=split_summary,
        metrics_summary=metrics_summary,
        classification_report_dict=report_dict,
        confusion_matrix=matrix,
        predictions=prediction_rows,
    )

    logger.info(format_run_message(run_id, f"Structured evaluation artifacts saved to {output_dir}"))
    logger.info(
        format_run_message(run_id, "Holdout metrics - accuracy: %.4f | macro_f1: %.4f | weighted_f1: %.4f"),
        metrics_summary["accuracy"],
        metrics_summary["macro_f1"],
        metrics_summary["weighted_f1"],
    )


def run_train(model: str, dataset: str | None, run_id: str | None = None) -> None:
    dataset_path = resolve_dataset_path(dataset)
    train_df, test_df, split_source = load_train_test_frames(dataset_path)

    split_summary = {
        "strategy": "stratified_train_test",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "dataset": str(dataset_path),
        "train_count": int(len(train_df)),
        "test_count": int(len(test_df)),
        "split_source": split_source,
        "leakage_check": "train and holdout test are disjoint by row split",
        "run_id": run_id,
    }

    classifier = create_model(model)
    model_logger = get_model_logger(model)
    model_logger.info(format_run_message(run_id, f"Starting training flow for model '{model}'"))

    X_train, y_train = classifier.preprocess_train_data(train_df)
    classifier.train_model(X_train, y_train)
    model_logger.info(format_run_message(run_id, "Model trained successfully."))

    X_test, y_test = classifier.preprocess_eval_data(test_df)
    y_true_labels, y_pred_labels = classifier.predict_labels(X_test, y_test)

    _save_outputs_evaluation(
        model_name=model,
        logger=model_logger,
        run_id=run_id,
        split_summary=split_summary,
        test_texts=test_df["text"].tolist(),
        y_true_labels=y_true_labels,
        y_pred_labels=y_pred_labels,
    )
    model_logger.info(format_run_message(run_id, f"Training flow completed for model '{model}'"))


def run_predict(model: str, text: str, run_id: str | None = None) -> None:
    classifier = create_model(model)
    model_logger = get_model_logger(model)
    model_logger.info(format_run_message(run_id, f"Starting prediction flow for model '{model}'"))
    predicted_intent = classifier.predict_intent(text)
    model_logger.info(format_run_message(run_id, "Prediction completed for model '%s' | intent='%s'"), model, predicted_intent)


def run_evaluate(model: str, dataset: str | None, run_id: str | None = None) -> None:
    dataset_path = resolve_dataset_path(dataset)
    test_df, train_count, split_source = load_eval_test_frame(dataset_path)

    split_summary = {
        "strategy": "stratified_train_test",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "dataset": str(dataset_path),
        "train_count": train_count,
        "test_count": int(len(test_df)),
        "split_source": split_source,
        "leakage_check": "evaluation executed on holdout split; no fit on test labels",
        "run_id": run_id,
    }

    classifier = create_model(model)
    model_logger = get_model_logger(model)
    model_logger.info(format_run_message(run_id, f"Starting evaluation flow for model '{model}'"))

    classifier.load_model()
    X_test, y_test = classifier.preprocess_eval_data(test_df)
    y_true_labels, y_pred_labels = classifier.predict_labels(X_test, y_test)

    _save_outputs_evaluation(
        model_name=model,
        logger=model_logger,
        run_id=run_id,
        split_summary=split_summary,
        test_texts=test_df["text"].tolist(),
        y_true_labels=y_true_labels,
        y_pred_labels=y_pred_labels,
    )
    model_logger.info(format_run_message(run_id, "Evaluation completed for model '%s' on holdout split."), model)
