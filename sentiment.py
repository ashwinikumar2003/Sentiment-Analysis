import streamlit as st
from transformers import pipeline

# Load the sentiment-analysis pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_pipeline()

# Streamlit app
st.title("Sentiment Analysis Tool")
st.write("This app uses a pretrained model to analyze the sentiment of your text.")

# User input
user_input = st.text_area("Enter text to analyze:", "")

if user_input:
    with st.spinner("Analyzing..."):
        results = sentiment_pipeline(user_input)
        st.success("Analysis complete!")

        for result in results:
            st.write(f"**Label**: {result['label']}, **Score**: {result['score']:.4f}")
