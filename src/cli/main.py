from __future__ import annotations

import argparse
from uuid import uuid4

from src.config.loader import PROJECT_ROOT, load_config
from src.data.data_augmentation import expand_training_dataset
from src.data.dataset_preparation import prepare_dataset_splits
from src.services.intent_service import run_evaluate, run_predict, run_train
from src.utils.logging_utils import get_project_logger


CONFIG = load_config()
LOGS_DIR = PROJECT_ROOT / CONFIG.get("paths", {}).get("logs_dir", "logs")
CLI_LOG_PATH = LOGS_DIR / "cli.log"
cli_logger = get_project_logger("intent_cli", CLI_LOG_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for travel intent detection models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", choices=["classic", "bert"], default="classic")
    train_parser.add_argument("--dataset", type=str, default=None, help="Path to training JSONL dataset")

    predict_parser = subparsers.add_parser("predict", help="Predict intent for input text")
    predict_parser.add_argument("--model", choices=["classic", "bert"], default="classic")
    predict_parser.add_argument("--text", type=str, required=True, help="Text input for prediction")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    evaluate_parser.add_argument("--model", choices=["classic", "bert"], default="classic")
    evaluate_parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation JSONL dataset")

    prepare_data_parser = subparsers.add_parser("prepare-data", help="Create deterministic train/validation/test splits")
    prepare_data_parser.add_argument("--dataset", type=str, default=None, help="Path to raw JSONL dataset")

    augment_data_parser = subparsers.add_parser("augment-data", help="Create expanded training dataset with rule-based augmentation")
    augment_data_parser.add_argument("--dataset", type=str, default=None, help="Path to source JSONL dataset")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_id = uuid4().hex[:8]

    cli_logger.info("[run_id=%s] Starting command '%s' with model '%s'", run_id, args.command, getattr(args, "model", "n/a"))

    try:
        if args.command == "train":
            run_train(args.model, args.dataset, run_id=run_id)
        elif args.command == "predict":
            run_predict(args.model, args.text, run_id=run_id)
        elif args.command == "evaluate":
            run_evaluate(args.model, args.dataset, run_id=run_id)
        elif args.command == "prepare-data":
            prepare_dataset_splits(args.dataset, run_id=run_id)
        elif args.command == "augment-data":
            expand_training_dataset(args.dataset, run_id=run_id)

        cli_logger.info("[run_id=%s] Command '%s' completed successfully", run_id, args.command)
    except Exception:
        cli_logger.exception("[run_id=%s] Command '%s' failed", run_id, args.command)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
