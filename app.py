import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Download necessary NLTK datasets required for preprocessing
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)

setup_nltk()

# Initialize text processing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Standardize casing
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize input
    tokens = word_tokenize(text)
    # Filter stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load trained vectorizer and classification model
@st.cache_resource
def load_assets():
    vec = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('fake_news_model.pkl')
    return vec, model

try:
    vectorizer, model = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Failed to load model artifacts. Ensure .pkl files are present. Error: {e}")

# User Interface
st.title("📰 Fake News Detection System")
st.markdown("Analyze news headlines or full article text to evaluate potential authenticity.")

user_input = st.text_area(
    label="News Content Input",
    height=200,
    placeholder="Paste news title or article text here..."
)

if st.button("Predict Authenticity", type="primary"):
    if not assets_loaded:
        st.error("Model files not loaded properly.")
    elif not user_input.strip():
        st.warning("Please enter news text prior to running prediction.")
    else:
        # Pipeline: Preprocess -> Vectorize -> Predict
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        
        # Display Results
        st.subheader("Analysis Result:")
        if prediction == 1:
            st.success("✅ REAL / AUTHENTIC NEWS")
        else:
            st.error("🚨 FAKE / UNVERIFIED NEWS")