import os

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import (
    CamembertForSequenceClassification,
    CamembertTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config.loader import PROJECT_ROOT, load_config
from src.models.base import IntentModelBase
from src.utils.logging_utils import get_project_logger

CONFIG = load_config()
BERT_CONFIG = CONFIG.get("bert", {})
PATHS_CONFIG = CONFIG.get("paths", {})

MODEL_DIR = PROJECT_ROOT / BERT_CONFIG.get("model_dir", "models/camembert")
LOGS_DIR = PROJECT_ROOT / PATHS_CONFIG.get("logs_dir", "logs")
LABEL_ENCODER_PATH = MODEL_DIR / BERT_CONFIG.get("label_encoder_file", "label_encoder.pth")
LOG_FILE_PATH = LOGS_DIR / BERT_CONFIG.get("log_file", "intent_classifier_bert.log")

logger = get_project_logger("intent_classifier_bert", LOG_FILE_PATH)


class IntentClassifierBert(IntentModelBase):
    model_key = "bert"

    def __init__(self, model_name=None, max_len=None, batch_size=None, epochs=None):
        model_name = BERT_CONFIG.get("model_name", "camembert-base") if model_name is None else model_name
        max_len = BERT_CONFIG.get("max_len", 128) if max_len is None else max_len
        batch_size = BERT_CONFIG.get("batch_size", 16) if batch_size is None else batch_size
        epochs = BERT_CONFIG.get("epochs", 15) if epochs is None else epochs

        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)
        self.model = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=9)
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.class_weights = None

    class IntentDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
            item["labels"] = self.labels[idx].clone().detach()
            return item

        def __len__(self):
            return len(self.labels)

    def _encode_texts(self, texts: list[str]):
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

    def preprocess_train_data(self, data: pd.DataFrame):
        X = data["text"].tolist()
        labels: list[str] = data["label"].astype(str).tolist()
        labels_encoded = self.label_encoder.fit_transform(labels)

        encodings = self._encode_texts(X)

        class_counts = torch.bincount(torch.tensor(labels_encoded))
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        self.class_weights = class_weights.to(self.model.device)

        input_count = int(encodings["input_ids"].shape[0])
        if input_count != total_samples:
            raise ValueError("Tokenized inputs and labels have mismatched lengths.")

        return encodings, torch.tensor(labels_encoded)

    def preprocess_eval_data(self, data: pd.DataFrame):
        X = data["text"].tolist()
        labels: list[str] = data["label"].astype(str).tolist()
        labels_encoded = self.label_encoder.transform(labels)

        encodings = self._encode_texts(X)

        input_count = int(encodings["input_ids"].shape[0])
        if input_count != len(labels):
            raise ValueError("Tokenized inputs and labels have mismatched lengths.")

        return encodings, torch.tensor(labels_encoded)

    def preprocess_data(self, data: pd.DataFrame):
        return self.preprocess_train_data(data)

    def train_model(self, encodings, y):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        dataset = self.IntentDataset(encodings, y)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Training loss: {avg_train_loss:.4f}")

        self.model.save_pretrained(str(MODEL_DIR))
        self.tokenizer.save_pretrained(str(MODEL_DIR))
        torch.save(self.label_encoder, LABEL_ENCODER_PATH)
        self.is_trained = True
        logger.info("Model and LabelEncoder trained and saved successfully.")

    def load_model(self):
        if os.path.exists(MODEL_DIR) and os.path.exists(LABEL_ENCODER_PATH):
            logger.info("Loading saved model, tokenizer, and label encoder...")
            self.model = CamembertForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
            self.tokenizer = CamembertTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
            self.label_encoder = torch.load(str(LABEL_ENCODER_PATH), weights_only=False)
            self.is_trained = True
            logger.info("Model, tokenizer, and label encoder loaded successfully.")
        else:
            logger.error("Model directory or label encoder file not found. Please train the model first.")
            raise FileNotFoundError("Model directory or label encoder file not found.")

    def evaluate_model(self, encodings, y):
        y_true_labels, y_pred_labels = self.predict_labels(encodings, y)

        report: str = str(classification_report(y_true_labels, y_pred_labels, target_names=self.label_encoder.classes_))
        matrix = confusion_matrix(y_true_labels, y_pred_labels)

        logger.info("Classification Report:\n" + report)
        logger.info("Confusion Matrix:\n" + str(matrix))

    def predict_intent(self, text):
        if not self.is_trained:
            try:
                self.model = CamembertForSequenceClassification.from_pretrained(str(MODEL_DIR), num_labels=9)
                self.tokenizer = CamembertTokenizer.from_pretrained(str(MODEL_DIR))
                self.label_encoder = torch.load(str(LABEL_ENCODER_PATH), weights_only=False)
                self.is_trained = True
                logger.info(f"Model and LabelEncoder loaded successfully from '{MODEL_DIR}'.")
            except Exception as e:
                logger.error(f"Error loading the trained model or LabelEncoder: {e}")
                return None

        encodings = self.tokenizer([text], truncation=True, padding=True, max_length=self.max_len, return_tensors="pt")
        inputs = {key: val for key, val in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        return self.label_encoder.inverse_transform([prediction])[0]

    def predict_labels(self, features, labels) -> tuple[list[str], list[str]]:
        if not self.is_trained:
            self.load_model()

        dataset = self.IntentDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.eval()
        predictions_encoded: list[int] = []
        true_labels_encoded: list[int] = []

        for batch in dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions_encoded.extend(preds.cpu().numpy().tolist())
                true_labels_encoded.extend(batch["labels"].cpu().numpy().tolist())

        y_true_labels = self.label_encoder.inverse_transform(true_labels_encoded).tolist()
        y_pred_labels = self.label_encoder.inverse_transform(predictions_encoded).tolist()
        return y_true_labels, y_pred_labels
