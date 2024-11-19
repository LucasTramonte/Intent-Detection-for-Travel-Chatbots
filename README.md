# Intent-Detection-for-Travel-Chatbots
This project implements a machine learning-based solution for intent detection in a travel assistant chatbot. The goal is to classify user messages into predefined intent categories such as "translate", "travel alert", "flight status", "lost luggage", and others, enabling the chatbot to provide accurate and contextually relevant responses.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Streamlit App](#streamlit-app)
- [Conclusion](#conclusion)


## Project Overview

The chatbot needs to classify user inputs into several categories, including:
- **translate**: User wants to translate a sentence to another language.
- **travel_alert**: User is asking about travel alerts for a specific destination.
- **flight_status**: User is asking about the status of a flight.
- **lost_luggage**: User is reporting lost luggage.
- **travel_suggestion**: User wants travel recommendations.
- **carry_on**: User is asking about carry-on baggage policies.
- **book_hotel**: User wants to book a hotel.
- **book_flight**: User wants to book a flight.

Additionally, the chatbot should identify requests that fall **out of scope** and respond accordingly.

## Installation

### Prerequisites
To run this project, ensure you have Python 3.7+ installed.

```bash
pip install -r requirements.txt
```

### Clone the Repository
```bash
git clone https://github.com/LucasTramonte/Intent-Detection-for-Travel-Chatbots.git
cd Intent-Detection-for-Travel-Chatbots
```

## Data Preprocessing

Data preprocessing is done by the intent_classifier.py script (and optionally intent_classifier_BERT.py for BERT-based models). The text data is processed using a TF-IDF vectorizer for traditional machine learning models, and labels are encoded using LabelEncoder. It also splits the data into training and testing sets.

For the BERT-based model in intent_classifier_BERT.py, data preprocessing includes tokenizing the input text using the CamembertTokenizer and encoding the labels.

- Text data is vectorized using TfidfVectorizer or tokenized using CamembertTokenizer for the BERT-based model.
- Labels are encoded into numerical values using LabelEncoder.
- The dataset is split into training and validation sets.
## Modeling

The intent classification is performed using both traditional machine learning algorithms and a BERT-based deep learning model. These models are evaluated on their accuracy, and the best-performing model is selected:

- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- BERT-based Model: The BERT model is implemented in the intent_classifier_BERT.py script. This script uses the CamembertForSequenceClassification model from Hugging Face for French-language intent classification. 

## Usage
To train the model using a dataset, run the following command:

```bash
python intent_classifier.py --dataset Assets/Data/intent-detection-train.csv --train
```

To train the BERT-based model, you can use the following command:

```bash
python intent_classifier_BERT.py --dataset Assets/Data/intent-detection-train.csv --train
```

Once the model is trained, it can be used for predictions. To make a prediction for a single input text, you can use the following command:

```bash
python intent_classifier.py --predict "Dois-je faire quoi si je vais au Bresil"
```

or :

```bash
python intent_classifier_BERT.py --predict "Je recherche un vol"
```

## Evaluation

After training, the model is evaluated using the evaluate_model method. The model performance is assessed using:

- Classification Report: Includes precision, recall, and F1-score for each class.
- Confusion Matrix: To evaluate true positives, true negatives, false positives, and false negatives.

To evaluate the model with a dataset, use the following command:

```bash
python intent_classifier.py --dataset Assets/Data/intent-detection-train.csv
```

To evaluate the BERT model with a dataset, use:

```bash
python intent_classifier_BERT.py --dataset Assets/Data/intent-detection-test.csv --evaluate
```

An evaluation report is saved as evaluation_report.txt for the traditional model, and evaluation_report_Camembert.txt for the BERT-based model, which includes all metrics for the trained model.

## Streamlit App

This project includes a **Streamlit app** for real-time intent classification. It allows users to interact with both traditional machine learning models and the BERT-based model via a simple web interface.

A user-friendly web application built with Streamlit is available in the app.py script and can be accessed [here](https://lucastramonte-illuin-project-app-yo5nco.streamlit.app/).

## Conclusion

This project demonstrates the use of both traditional machine learning models and deep learning models (specifically BERT-based models) to detect intents in user queries for a travel assistant chatbot. The models were trained, evaluated, and the best-performing model was selected to make accurate predictions.

Further improvements can be made by:

- Tuning hyperparameters and using cross-validation to reduce overfitting.
- Using more advanced models such as XGBoost for better performance.
- Exploring additional data preprocessing methods such as stemming, lemmatization, or using pre-trained embeddings for more robust features.


