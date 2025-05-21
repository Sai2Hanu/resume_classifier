import streamlit as st
import joblib
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import os
import gdown
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")
    logger.error(f"Error downloading NLTK resources: {e}")

# Download .pkl files from Google Drive (replace with your file IDs)
@st.cache_resource
def download_pkl_files():
    try:
        if not os.path.exists('best_xgb_model.pkl'):
            gdown.download('https://drive.google.com/uc?id=<YOUR_MODEL_FILE_ID>', 'best_xgb_model.pkl', quiet=False)
        if not os.path.exists('tfidf_vectorizer.pkl'):
            gdown.download('https://drive.google.com/uc?id=<YOUR_VECTORIZER_FILE_ID>', 'tfidf_vectorizer.pkl', quiet=False)
        if not os.path.exists('label_encoder.pkl'):
            gdown.download('https://drive.google.com/uc?id=<YOUR_LABEL_ENCODER_FILE_ID>', 'label_encoder.pkl', quiet=False)
        logger.info("All .pkl files downloaded successfully")
    except Exception as e:
        st.error(f"Error downloading .pkl files: {e}")
        logger.error(f"Error downloading .pkl files: {e}")
        st.stop()

# Load the saved model and preprocessing objects
try:
    download_pkl_files()
    model = joblib.load('best_xgb_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    logger.info("Model and preprocessing objects loaded successfully")
except Exception as e:
    st.error(f"Error loading model or preprocessing objects: {e}")
    logger.error(f"Error loading model or preprocessing objects: {e}")
    st.stop()

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english') + list(string.punctuation))
lemmatizer = WordNetLemmatizer()

# Text cleaning function (same as in the notebook)
def clean_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Error cleaning text: {e}")
        logger.error(f"Error cleaning text: {e}")
        return ''

# Streamlit app layout
st.title("Resume Role Classifier")
st.subheader("Predict the job role from resume text")
st.markdown("Paste a resume below to classify the job role (e.g., Peoplesoft Admin, React Dev).")

# Text input area for resume
resume_text = st.text_area("Resume Text", height=300, placeholder="Paste the resume text here...")

# Predict button
if st.button("Predict Role"):
    if resume_text.strip():
        try:
            # Preprocess the resume text
            cleaned_text = clean_text(resume_text)
            if not cleaned_text:
                st.error("Text cleaning failed. Please check the input.")
                st.stop()
            tfidf_features = vectorizer.transform([cleaned_text])

            # Make prediction
            prediction = model.predict(tfidf_features)
            predicted_role = le.inverse_transform(prediction)[0]

            # Log and display the result
            logger.info(f"Predicted role: {predicted_role}")
            st.success(f"Predicted Job Role: **{predicted_role}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            logger.error(f"Error during prediction: {e}")
    else:
        st.error("Please enter resume text to make a prediction.")
        logger.warning("Empty resume text provided")

# Instructions and footer
st.markdown("""
### Instructions
- Paste the full text of a resume in the text area above.
- Click the "Predict Role" button to see the predicted job role.
- Ensure the resume includes relevant skills and experience for accurate predictions.
- Example roles: Peoplesoft Admin, React Dev, Sql Dev, Workday HEXAWARE.
""")
