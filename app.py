import re
import joblib
import nltk
import streamlit as st
from google import genai
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Fake News Detector & Fact Checker",
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
    text = text.lower()
    # Remove leading location or publisher tags (e.g., "WASHINGTON (Reuters) -", "GENEVA —")
    text = re.sub(r'^[a-z\s,]+(?:\(reuters\))?\s*[\-—]\s*', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
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
st.title("📰 Fake News Detector & Fact Checker")
st.markdown("Analyze news headlines using Machine Learning style classification and Gemini AI real-time fact-checking.")

user_input = st.text_area(
    label="News Content Input",
    height=180,
    placeholder="Paste news title or article text here..."
)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("Predict Style (ML Model)", type="primary", use_container_width=True)

with col2:
    gemini_btn = st.button("AI Fact Check (Gemini)", use_container_width=True)

# 1. Machine Learning Prediction Logic
if predict_btn:
    if not assets_loaded:
        st.error("Model files not loaded properly.")
    elif not user_input.strip():
        st.warning("Please enter news text prior to running prediction.")
    else:
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        
        probabilities = model.predict_proba(vectorized_text)[0]
        fake_prob = probabilities[0] * 100
        real_prob = probabilities[1] * 100
        prediction = model.predict(vectorized_text)[0]
        
        st.subheader("ML Analysis Result:")
        if prediction == 1:
            st.success(f"✅ REAL NEWS STYLE ({real_prob:.2f}% Confidence)")
            st.progress(int(real_prob))
        else:
            st.error(f"🚨 FAKE NEWS STYLE ({fake_prob:.2f}% Confidence)")
            st.progress(int(fake_prob))
            
        st.write("---")
        st.markdown("**Confidence Breakdown:**")
        st.write(f"- **Real News Probability:** {real_prob:.2f}%")
        st.write(f"- **Fake News Probability:** {fake_prob:.2f}%")

# 2. Gemini AI Fact-Check Logic
if gemini_btn:
    if not user_input.strip():
        st.warning("Please enter a claim or headline to fact check.")
    else:
        # Retrieve key from Streamlit Secrets or fall back to provided token
        api_key = st.secrets.get("GEMINI_API_KEY", "AQ.Ab8RN6LxD8a3RFB5VBtlNEnEfeDARCkVbO2V9iqOsPhaReqM_w")
        
        if not api_key:
            st.error("Missing Gemini API Key. Please add `GEMINI_API_KEY` to your Streamlit Secrets.")
        else:
            try:
                with st.spinner("Analyzing claims using Gemini AI..."):
                    # Initialize Gemini Client
                    client = genai.Client(api_key=api_key)
                    
                    prompt = (
                        "You are an expert real-time fact-checker. Analyze the following news claim or headline.\n"
                        "1. Clearly state whether the statement is TRUE, FALSE, UNVERIFIED, or MISLEADING.\n"
                        "2. Provide an estimated confidence rating (e.g. 95%).\n"
                        "3. Give a clear, concise 2-3 sentence explanation explaining why.\n\n"
                        f"Claim to verify: \"{user_input}\""
                    )
                    
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    
                    st.subheader("🤖 Gemini AI Fact-Check Analysis:")
                    st.info(response.text)
            except Exception as ex:
                st.error(f"Gemini API Exception: {ex}")
