import warnings
import sys
from typing import Any, Optional, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.base import clone

from src.config.loader import PROJECT_ROOT, load_config
from src.models.base import IntentModelBase
from src.utils.logging_utils import get_project_logger

warnings.filterwarnings(
    "ignore",
    message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None",
    category=UserWarning,
)

CONFIG = load_config()
CLASSIC_CONFIG = CONFIG.get("classic", {})
PATHS_CONFIG = CONFIG.get("paths", {})

MODEL_DIR = PROJECT_ROOT / CLASSIC_CONFIG.get("model_dir", "models/traditional")
MODEL_PATH = MODEL_DIR / CLASSIC_CONFIG.get("model_file", "best_model.joblib")
LABEL_ENCODER_PATH = MODEL_DIR / CLASSIC_CONFIG.get("label_encoder_file", "label_encoder.joblib")
LOGS_DIR = PROJECT_ROOT / PATHS_CONFIG.get("logs_dir", "logs")
LOG_FILE_PATH = LOGS_DIR / CLASSIC_CONFIG.get("log_file", "intent_classifier.log")

logger = get_project_logger("intent_classifier", LOG_FILE_PATH)

# Backward-compatible module alias for previously serialized joblib models
# that reference the old module path "intent_classifier".
sys.modules.setdefault("intent_classifier", sys.modules[__name__])


class PredictorModel(Protocol):
    def predict(self, X: Any) -> Any:
        ...


class IntentClassifier(IntentModelBase):
    model_key = "classic"

    def __init__(self, max_iter=None, n_estimators=None, random_state=None):
        max_iter = CLASSIC_CONFIG.get("max_iter", 1000) if max_iter is None else max_iter
        n_estimators = CLASSIC_CONFIG.get("n_estimators", 100) if n_estimators is None else n_estimators
        random_state = CLASSIC_CONFIG.get("random_state", 42) if random_state is None else random_state
        max_features = CLASSIC_CONFIG.get("max_features", 5000)

        self.vectorizer = TfidfVectorizer(
            tokenizer=self.french_tokenizer,
            max_features=max_features,
        )
        self.label_encoder = LabelEncoder()
        self.models = {
            "Random Forest": RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight="balanced"),
            "Logistic Regression": LogisticRegression(max_iter=max_iter, class_weight="balanced"),
            "SVM": SVC(class_weight="balanced", kernel="linear"),
            "Naive Bayes": MultinomialNB(),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_estimators),
            "KNN": KNeighborsClassifier(),
        }
        self.is_trained = False
        self.best_model: Optional[PredictorModel] = None
        self.best_f1_macro = 0

    @staticmethod
    def french_tokenizer(text):
        return text.lower().replace("'", " ").split()

    def preprocess_train_data(self, data: pd.DataFrame):
        X = data["text"]
        y = self.label_encoder.fit_transform(data["label"])
        return X, y

    def preprocess_eval_data(self, data: pd.DataFrame):
        X = data["text"]
        y = self.label_encoder.transform(data["label"])
        return X, y

    def preprocess_data(self, data: pd.DataFrame):
        return self.preprocess_train_data(data)

    def train_model(self, X, y):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        class_counts = pd.Series(y).value_counts()
        min_class_count = int(class_counts.min())
        n_splits = max(2, min(5, min_class_count))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        logger.info("Using %d-fold Stratified CV (min class count: %d)", n_splits, min_class_count)

        best_estimator = None
        best_f1_macro = 0

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")

            f1_scores = []
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                pipeline = Pipeline(
                    [
                        ("vectorizer", clone(self.vectorizer)),
                        ("classifier", clone(model)),
                    ]
                )
                pipeline.fit(X_train_fold, y_train_fold)
                y_pred = pipeline.predict(X_val_fold)

                f1_macro = f1_score(y_val_fold, y_pred, average="macro")
                f1_scores.append(f1_macro)

            mean_f1_macro = np.mean(f1_scores)
            logger.info(f"{model_name} Mean Macro F1-score: {mean_f1_macro:.4f}")

            if mean_f1_macro > best_f1_macro:
                best_f1_macro = mean_f1_macro
                best_estimator = clone(model)

        if best_estimator is None:
            raise RuntimeError("No estimator selected during cross-validation training.")

        final_pipeline = Pipeline(
            [
                ("vectorizer", clone(self.vectorizer)),
                ("classifier", clone(best_estimator)),
            ]
        )
        final_pipeline.fit(X, y)

        self.best_model = final_pipeline
        self.best_f1_macro = best_f1_macro
        self.is_trained = True
        logger.info(f"Best model selected and refit on training split. Mean Macro F1-score: {best_f1_macro:.4f}")

        joblib.dump(self.best_model, MODEL_PATH)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        logger.info("Best model and label encoder saved successfully.")

    def load_model(self):
        if MODEL_PATH.exists() and LABEL_ENCODER_PATH.exists():
            logger.info("Loading saved model and label encoder...")
            self.best_model = joblib.load(MODEL_PATH)
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            self.is_trained = True
            logger.info("Model and label encoder loaded successfully.")
        else:
            logger.error("Model or label encoder file not found. Please train the model first.")
            raise FileNotFoundError("Trained model or label encoder file not found.")

    def evaluate_model(self, X, y):
        y_true_labels, y_pred_labels = self.predict_labels(X, y)

        report: str = str(
            classification_report(
                y_true_labels,
                y_pred_labels,
                target_names=self.label_encoder.classes_,
                zero_division=1,
            )
        )
        matrix = confusion_matrix(y_true_labels, y_pred_labels)

        logger.info("Classification Report:\n" + report)
        logger.info("Confusion Matrix:\n" + str(matrix))

    def predict_intent(self, text):
        if not self.is_trained:
            self.load_model()

        if self.best_model is None:
            raise RuntimeError("Model is not available. Train or load a model before prediction.")

        prediction = self.best_model.predict([text])
        return self.label_encoder.inverse_transform(prediction)[0]

    def predict_labels(self, features, labels) -> tuple[list[str], list[str]]:
        if not self.is_trained:
            self.load_model()

        if self.best_model is None:
            raise RuntimeError("Model is not available for label prediction.")

        y_pred_encoded = self.best_model.predict(features)
        y_true_labels = self.label_encoder.inverse_transform(labels).tolist()
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_encoded).tolist()
        return y_true_labels, y_pred_labels
