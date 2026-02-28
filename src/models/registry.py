from __future__ import annotations

import logging

from src.models.base import IntentModelBase
from src.models.bert import IntentClassifierBert, logger as bert_logger
from src.models.classic import IntentClassifier, logger as classic_logger

MODEL_REGISTRY: dict[str, type[IntentModelBase]] = {
    "classic": IntentClassifier,
    "bert": IntentClassifierBert,
}

MODEL_LOGGERS: dict[str, logging.Logger] = {
    "classic": classic_logger,
    "bert": bert_logger,
}


def create_model(model_key: str) -> IntentModelBase:
    try:
        model_class = MODEL_REGISTRY[model_key]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_key}'. Supported: {supported}") from exc

    return model_class()


def get_model_logger(model_key: str) -> logging.Logger:
    try:
        return MODEL_LOGGERS[model_key]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_LOGGERS))
        raise ValueError(f"Unsupported model '{model_key}'. Supported: {supported}") from exc
