import streamlit as st
import pickle
from textblob import TextBlob

with open('H:\Programming\PDS\project\knn_model.pkl', 'rb') as model_file:
    sentiment_model = pickle.load(model_file)

st.title("Sentiment Analysis App")

input_sentence = st.text_input("Enter a sentence:")

if input_sentence:
    processed_sentence = input_sentence.lower()

    blob = TextBlob(processed_sentence)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Display results
    st.subheader("Sentiment Analysis Result:")
    st.write(f"Input Sentence: {input_sentence}")
    st.write(f"Sentiment: {sentiment} (Polarity: {polarity:.2f})")

st.sidebar.title("About")
st.sidebar.write("This is a simple sentiment analysis web app.")
st.sidebar.write("It uses a pre-trained model to analyze the sentiment of your input sentence.")