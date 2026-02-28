from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class IntentModelBase(ABC):
    model_key: str

    @abstractmethod
    def preprocess_train_data(self, data: pd.DataFrame) -> tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def preprocess_eval_data(self, data: pd.DataFrame) -> tuple[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def train_model(self, features: Any, labels: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_intent(self, text: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def predict_labels(self, features: Any, labels: Any) -> tuple[list[str], list[str]]:
        raise NotImplementedError
