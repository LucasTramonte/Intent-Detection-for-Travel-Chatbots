import os
import logging
import pandas as pd
import numpy as np

import joblib
import argparse #CLI
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self, max_iter=1000, n_estimators=100, random_state=42):
        self.vectorizer = TfidfVectorizer(tokenizer=self.french_tokenizer,token_pattern=None, max_features=5000)
        self.label_encoder = LabelEncoder()
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, class_weight='balanced'),
            'Logistic Regression': LogisticRegression(max_iter=max_iter, class_weight='balanced'),
            'SVM': SVC(class_weight='balanced', kernel='linear'),
            'Naive Bayes': MultinomialNB(),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=n_estimators),
            'KNN': KNeighborsClassifier()
        }
        self.is_trained = False
        self.best_model: Optional[BaseEstimator] = None
        self.best_f1_macro = 0  # Track macro F1-score as evaluation metric
        
    @staticmethod
    def french_tokenizer(text):
        # Tokenize and handle French-specific characters
        return text.lower().replace("'", " ").split()
    
    
    """
    def set_class_weights(self, y):
        # Encode the weights after fitting label encoder to map them to integer labels
        original_weights = {
            'translate': 1, 
            'travel_alert': 1, 
            'flight_status': 1, 
            'lost_luggage': 1.2,    # Higher weight for lost_luggage
            'travel_suggestion': 1, 
            'carry_on': 1, 
            'book_hotel': 1, 
            'book_flight': 1, 
            'out_of_scope': 1.3   # Adjusted weight for out_of_scope
        }
        
        # Encode the original class weights dictionary
        encoded_weights = {self.label_encoder.transform([k])[0]: v for k, v in original_weights.items()}
        
        # Add the encoded class weights to each model that supports it
        self.models['Random Forest'].class_weight = encoded_weights
        self.models['Logistic Regression'].class_weight = encoded_weights
        self.models['SVM'].class_weight = encoded_weights
    """

    def preprocess_data(self, data: pd.DataFrame):
        X = data['text']
        y = self.label_encoder.fit_transform(data['label'])
        #self.set_class_weights(y)
        return X, y

    def train_model(self, X, y):
        # StratifiedKFold Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        best_model = None
        best_f1_macro = 0

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            f1_scores = []
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Train the model on the current fold
                pipeline = make_pipeline(self.vectorizer, model)
                pipeline.fit(X_train_fold, y_train_fold)
                
                # Predict the values for the validation fold
                y_pred = pipeline.predict(X_val_fold)

                f1_macro = f1_score(y_val_fold, y_pred, average='macro')
                f1_scores.append(f1_macro)

            # Calculate mean F1-score across all folds
            mean_f1_macro = np.mean(f1_scores)
            logger.info(f"{model_name} Mean Macro F1-score: {mean_f1_macro:.4f}")
            
            # Track the best model based on macro F1-score
            if mean_f1_macro > best_f1_macro:
                best_f1_macro = mean_f1_macro
                best_model = pipeline

        self.best_model = best_model
        self.best_f1_macro = best_f1_macro
        self.is_trained = True
        logger.info(f"Best Model: {best_model} with Mean Macro F1-score: {best_f1_macro:.4f}")

        # Save the best model and label encoder
        joblib.dump(self.best_model, 'best_model.joblib')
        joblib.dump(self.label_encoder, 'label_encoder.joblib')
        logger.info("Best model and label encoder saved successfully.")

    def load_model(self):
        if os.path.exists('best_model.joblib') and os.path.exists('label_encoder.joblib'):
            logger.info("Loading saved model and label encoder...")
            self.best_model = joblib.load('best_model.joblib')
            self.label_encoder = joblib.load('label_encoder.joblib')
            self.is_trained = True
            logger.info("Model and label encoder loaded successfully.")
        else:
            logger.error("Model or label encoder file not found. Please train the model first.")
            raise FileNotFoundError("Trained model or label encoder file not found.")

    def evaluate_model(self, X, y):
        if not self.is_trained:
            logger.info("Model is not trained, loading the model...")
            self.load_model()  # Load the trained model before evaluation
        
        y_pred = self.best_model.predict(X)
        
        # Generate the classification report and confusion matrix
        report = classification_report(y, y_pred, target_names=self.label_encoder.classes_, zero_division=1)
        matrix = confusion_matrix(y, y_pred)

        logger.info("Classification Report:\n" + report)
        logger.info("Confusion Matrix:\n" + str(matrix))
        
        try:
            with open('evaluation_report.txt', 'w') as f:
                f.write("Classification Report:\n")
                f.write(report)
                f.write("\nConfusion Matrix:\n")
                f.write(str(matrix))
            logger.info("Evaluation report saved to evaluation_report.txt")
        except Exception as e:
            logger.error(f"Error while saving the evaluation report: {e}")

    def predict_intent(self, text):
        if not self.is_trained:
            self.load_model()
        
        prediction = self.best_model.predict([text])
        predicted_label = self.label_encoder.inverse_transform(prediction)[0]
        return predicted_label

def main():
    parser = argparse.ArgumentParser(description='Intent Classification with Multiple Models')
    parser.add_argument('--dataset', type=str, help='Path to the CSV dataset for training or evaluation')
    parser.add_argument('--predict', type=str, help='Text input to predict the intent')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')
    parser.add_argument('--test', type=str, help='Path to a test CSV dataset for evaluation')
    args = parser.parse_args()
    
    classifier = IntentClassifier()

    if args.train:
        if not args.dataset:
            raise ValueError("Please provide a dataset path with --dataset when training the model.")
        
        data = pd.read_csv(args.dataset)
        X, y = classifier.preprocess_data(data)
        classifier.train_model(X, y)
        logger.info("Model trained successfully.")

    if args.predict:
        predicted_intent = classifier.predict_intent(args.predict)
        print(f"Predicted Intent: {predicted_intent}")
    
    # Evaluation phase
    if args.test:
        test_data = pd.read_csv(args.test)
        X_test, y_test = classifier.preprocess_data(test_data)
        classifier.evaluate_model(X_test, y_test)

if __name__ == '__main__':
    main()
