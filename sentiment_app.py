import streamlit as st
from transformers import pipeline


st.title("Sentiment Analysis")


# Load a pre-trained sentiment analysis model

@st.cache_resource
def load_model():
    print("Loading...")
    return pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_model()

text = st.text_area("**Write your feedback:**")

if st.button("Submit"):
    if text.strip():
        result=classifier(text)
        st.write("Text:",text)
        st.write("Result:",result)
    else:
        st.warning("Please, write your feedback first.")
