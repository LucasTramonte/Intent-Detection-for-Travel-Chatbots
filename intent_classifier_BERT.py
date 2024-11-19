import os
import logging
import pandas as pd
import argparse
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification, get_linear_schedule_with_warmup

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class IntentClassifier_BERT:
    def __init__(self, model_name='camembert-base', max_len=128, batch_size=16, epochs=15):
        # Camembert Tokenizer and Model
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

        def __getitem__(self, idx): # allows the dataset to behave like a list
            item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx].clone().detach()
            return item # When the dataset is indexed (e.g., dataset[0]), it returns the tokenized input and the label for that index.

        def __len__(self): # Important for creating batches during training.
            return len(self.labels) # returns the number of items (samples) in the dataset

    def preprocess_data(self, data: pd.DataFrame):
        """
        Tokenizes input text data and encodes labels for training.
        """
        X = data['text'].tolist()
        y = data['label'].tolist()
        y = self.label_encoder.fit_transform(y)  # Encoding labels as integers

        # Tokenize the text data
        encodings = self.tokenizer(X, truncation=True, padding=True, max_length=self.max_len, return_tensors='pt')
        
        # Calculate class weights for handling class imbalance
        class_counts = torch.bincount(torch.tensor(y))
        total_samples = len(y)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        self.class_weights = class_weights.to(self.model.device)  # Store the class weights on the same device as the model
        
        # Ensure the lengths of encodings and labels match
        if len(encodings['input_ids']) != len(y):
            raise ValueError("Tokenized inputs and labels have mismatched lengths.")
        
        return encodings, torch.tensor(y)

    def train_model(self, encodings, y):
        """
        Trains the BERT model using the provided data.
        """
        dataset = self.IntentDataset(encodings, y)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                total_train_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Training loss: {avg_train_loss:.4f}")

        # Save the trained model, tokenizer, and LabelEncoder
        self.model.save_pretrained('camembert_model')
        self.tokenizer.save_pretrained('camembert_model')
        torch.save(self.label_encoder, 'label_encoder.pth')
        self.is_trained = True
        logger.info("Model and LabelEncoder trained and saved successfully.")

    def load_model(self):
        model_dir = 'camembert_model'
        label_encoder_path = 'label_encoder.pth'

        if os.path.exists(model_dir) and os.path.exists(label_encoder_path):
            logger.info("Loading saved model, tokenizer, and label encoder...")

            # Load model and tokenizer
            self.model = CamembertForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            self.tokenizer = CamembertTokenizer.from_pretrained(model_dir, local_files_only=True)

            # Load label encoder
            self.label_encoder = torch.load(label_encoder_path)

            self.is_trained = True
            logger.info("Model, tokenizer, and label encoder loaded successfully.")
        else:
            logger.error("Model directory or label encoder file not found. Please train the model first.")
            raise FileNotFoundError("Model directory or label encoder file not found.")
        
    def evaluate_model(self, encodings, y):
        """
        Evaluates the model using a test dataset.
        """
        if not self.is_trained:
            logger.error("Model not trained. Please train the model first.")
            self.load_model()  # Load the trained model before evaluation

        # Create dataset and DataLoader for evaluation
        dataset = self.IntentDataset(encodings, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.eval()
        predictions = []
        true_labels = []

        for batch in dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())

        # Calculate classification report and confusion matrix
        report = classification_report(true_labels, predictions, target_names=self.label_encoder.classes_)
        matrix = confusion_matrix(true_labels, predictions)

        logger.info("Classification Report:\n" + report)
        logger.info("Confusion Matrix:\n" + str(matrix))
        
        # Save the report to a text file
        try:
            with open('evaluation_report_Camembert.txt', 'w') as f:
                f.write("Classification Report:\n")
                f.write(report)
                f.write("\nConfusion Matrix:\n")
                f.write(str(matrix))
            logger.info("Evaluation report saved to evaluation_report_Camembert.txt")
        except Exception as e:
            logger.error(f"Error while saving the evaluation report: {e}")
    
    def predict_intent(self, text):
        """
        Predicts the intent of a single input text.
        """
        if not self.is_trained:
            try:
                self.model = CamembertForSequenceClassification.from_pretrained('camembert_model', num_labels=9)
                self.tokenizer = CamembertTokenizer.from_pretrained('camembert_model')
                self.label_encoder = torch.load('label_encoder.pth')  # Load the trained LabelEncoder
                self.is_trained = True
                logger.info("Model and LabelEncoder loaded successfully from 'camembert_model'.")
            except Exception as e:
                logger.error(f"Error loading the trained model or LabelEncoder: {e}")
                return None

        # Tokenizing the input text
        encodings = self.tokenizer([text], truncation=True, padding=True, max_length=self.max_len, return_tensors='pt')
        
        # No need to clone or detach here, just directly pass the tensors
        inputs = {key: val for key, val in encodings.items()}
        
        # Perform the prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
        
        # Decode the label
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_label


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Intent Classification with Camembert')
    parser.add_argument('--dataset', type=str, help='Path to the CSV dataset for training or evaluation')
    parser.add_argument('--predict', type=str, help='Text input to predict the intent')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')
    parser.add_argument('--evaluate', action='store_true', help='Flag to evaluate the model')
    parser.add_argument('--test', type=str, help='Path to a test CSV dataset for evaluation')
    args = parser.parse_args()

    classifier = IntentClassifier_BERT()

    if args.train:
        if not args.dataset:
            raise ValueError("Please provide a dataset path with --dataset when training the model.")
        
        # Load the dataset
        data = pd.read_csv(args.dataset)
        
        # Preprocess the dataset and get tokenized inputs and labels
        encodings, y = classifier.preprocess_data(data)
        
        # Train the model on the provided dataset
        classifier.train_model(encodings, y)
        print("Model trained successfully.")

    if args.predict:
        # Predict the intent of the given input text
        predicted_intent = classifier.predict_intent(args.predict)
        print(f"Predicted Intent: {predicted_intent}")

    if args.evaluate:
        if not args.dataset:
            raise ValueError("Please provide a dataset path with --dataset when evaluating the model.")
        
        # Load the dataset
        data = pd.read_csv(args.dataset)
        
        # Preprocess the dataset and get tokenized inputs and labels
        encodings, y = classifier.preprocess_data(data)
        
        # Evaluate the model on the provided dataset
        classifier.evaluate_model(encodings, y)
        print("Evaluation completed.")

if __name__ == '__main__':
    main()
