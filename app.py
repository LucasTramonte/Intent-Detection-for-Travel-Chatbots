import streamlit as st
from intent_classifier import IntentClassifier   # For ML models
#from intent_classifier_BERT import IntentClassifier_BERT  # For Camembert (BERT)

# Title of the app
st.title("Travel Chatbot Intent Classification")

# Introduction in markdown format
st.markdown("""
            
This project implements a machine learning-based solution for intent detection in a travel assistant chatbot.

## Purpose
The goal is to classify user messages into predefined intent categories such as "translate", "travel alert", "flight status", "lost luggage", and others, enabling the chatbot to provide accurate and contextually relevant responses.
   
### How It Works:
1. **Type your message** in French.
2. **Click 'Classify Intent'** to see the predicted intent.

""")

# Input text area for the user to type their message in French
user_input = st.text_input("Type your message in French:")

# Button to trigger intent classification
if st.button("Classify Intent"):
    ml_classifier = IntentClassifier()
    predicted_intent = ml_classifier.predict_intent(user_input)
    st.write(f"Predicted Intent (Traditional ML): {predicted_intent}")
    
    #BERT_classifier = IntentClassifier_BERT()
    #predicted_intent = BERT_classifier.predict_intent(user_input)
    #st.write(f"Predicted Intent (BERT): {predicted_intent}")
    