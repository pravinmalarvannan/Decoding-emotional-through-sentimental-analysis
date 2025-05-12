import joblib
import streamlit as st
import re
import string

# Load saved models
model = joblib.load("logistic_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl,")
label_encoder = joblib.load("label_encoder.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"@\w+|http\S+|#\w+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Streamlit app
st.title("Social Media Emotion Analyzer")
st.write("Enter a tweet to detect its emotional sentiment")

user_input = st.text_area("Tweet Text", "")

if st.button("Analyze Sentiment"):
    cleaned = clean_text(user_input)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    emotion = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Emotion: **{emotion}**")
