import re
import joblib
import nltk
import requests
import streamlit as st
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
    text = re.sub(r'^[a-z\s,]+(?:\(reuters\))?\s*[\-—]\s*', '', text)
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

# 2. Gemini Fact Check Logic via Direct REST API
if gemini_btn:
    if not user_input.strip():
        st.warning("Please enter a claim or headline to fact check.")
    else:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        
        if not api_key:
            st.error("Missing GEMINI_API_KEY. Please set an 'AIzaSy...' API key in Streamlit Secrets.")
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            
            prompt = (
                "You are an expert real-time fact-checker. Analyze the following news statement:\n"
                "1. State clearly whether it is TRUE, FALSE, MISLEADING, or UNVERIFIED.\n"
                "2. Provide an estimated confidence rating percentage.\n"
                "3. Provide a brief 2-3 sentence explanation with verifiable facts.\n\n"
                f"Statement: \"{user_input}\""
            )
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            
            try:
                with st.spinner("Analyzing claim with Gemini AI..."):
                    res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
                    if res.status_code == 200:
                        data = res.json()
                        response_text = data['candidates'][0]['content']['parts'][0]['text']
                        st.subheader("🤖 Gemini AI Fact-Check Result:")
                        st.info(response_text)
                    else:
                        st.error(f"API Error ({res.status_code}): Ensure your API key starts with 'AIzaSy' and is valid.")
            except Exception as ex:
                st.error(f"Request failed: {ex}")
