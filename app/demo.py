import os
import sys
import pandas as pd
import streamlit as st
import joblib

# Adjust path to import modules from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import load_data
from preprocessing import preprocess_resume
from section_parser import extract_sections
from feature_engineering import vectorize_resume
from train_classifier import train_model
from predict import predict_category

# Load pre-trained model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'svm.pkl'))
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Resume Screening", layout="wide")
st.title("Resume Screening Application")

st.markdown("""
This application allows you to:
- Upload a CSV file of resumes
- Automatically extract sections like Education, Experience, and Skills
- Preprocess the text and classify resumes into job categories
""")

uploaded_file = st.file_uploader("Upload your resume CSV", type=["csv"])

CHUNK_SIZE = 50  # Tune this based on available memory
CHUNK_OVERLAP = 0  # Optional: overlap in rows between chunks if needed

def process_in_chunks(file, model):
    predictions = []
    for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE, usecols=['Resume_str']):
        # Optional: Extract sections
        chunk['Sections'] = chunk['Resume_str'].apply(extract_sections)

        # Preprocessing
        chunk['cleaned_resume'] = chunk['Resume_str'].apply(preprocess_resume)

        # Feature extraction
        X_features = vectorize_resume(chunk['cleaned_resume'])

        # Predict
        chunk['Predicted_Category'] = model.predict(X_features)

        predictions.append(chunk)

    return pd.concat(predictions, ignore_index=True)

if uploaded_file is not None:
    with st.spinner("Processing resumes..."):
        df_result = process_in_chunks(uploaded_file, model)

    st.success("Resume classification complete!")
    st.subheader("Sample Predictions")
    st.dataframe(df_result[['Resume_str', 'Predicted_Category']].head())

    csv_download = df_result.to_csv(index=False)
    st.download_button("Download Full Predictions", data=csv_download, file_name="resume_predictions.csv", mime="text/csv")
else:
    st.info("Please upload a CSV file containing a 'Resume_str' column.")

def load_resume_text():
    print("Choose Input Method:")
    print("1. Paste resume text")
    print("2. Upload .txt file path")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        print("\nPaste your resume text (end with a blank line):")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        return "\n".join(lines)

    elif choice == "2":
        path = input("Enter the path to the .txt resume file: ").strip()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            print("\u274c File does not exist.")
            sys.exit(1)

    else:
        print("\u274c Invalid input method.")
        sys.exit(1)

def main():
    print("\U0001f50d Resume Screening Started")

    resume_text = load_resume_text()

    # Clean + Lemmatize
    cleaned_text = clean_text(resume_text)
    lemmatized_text = lemmatize_text(cleaned_text)

    # Load model and vectorizer
    model = joblib.load("models/svm_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    # Predict
    predicted_category = predict(lemmatized_text, model, vectorizer)
    print(f"\n\u2705 Predicted Category: {predicted_category}")

    # Extract Info
    print("\n\ud83d\udccc Extracted Contact Information:")
    emails = extract_emails(resume_text)
    phones = extract_phone_numbers(resume_text)
    links = extract_links(resume_text)
    print("Emails:", emails if emails else "None")
    print("Phones:", phones if phones else "None")
    print("Links:", links if links else "None")

    # Extract Sections
    print("\n\U0001f9e9 Extracted Resume Sections:")
    sections = extract_sections(resume_text)
    for section, content in sections.items():
        print(f"\n--- {section} ---\n{content[:500]}...")  # preview first 500 chars

    print("\n\u2714 Done.")

if __name__ == "__main__":
    main()
