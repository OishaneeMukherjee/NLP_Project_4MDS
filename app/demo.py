import streamlit as st
from src.preprocessing import clean_text, lemmatize_text
from src.feature_engineering import extract_features
from src.predict import load_model, predict
import joblib

# Load vectorizer and model
vectorizer = joblib.load("models/vectorizer.pkl")
model = load_model("models/svm.pkl")

st.title("Resume Screening System")
resume_input = st.text_area("Paste your resume text here:")

if st.button("Classify"):
    cleaned = lemmatize_text(clean_text(resume_input))
    features = vectorizer.transform([cleaned])
    prediction = predict(model, features)
    st.success(f"Predicted Category: {prediction[0]}")
