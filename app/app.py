import streamlit as st
import os
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -------------------------
# Download Required NLTK Data
# -------------------------
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")

# -------------------------
# Text Preprocessing Setup
# -------------------------
stop_words = set(stopwords.words("english"))
negation_words = {"not", "nor", "never", "no"}
stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()

def custom_preprocessor(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and word.isalpha()
    ]
    return " ".join(words)


@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "model", "news_credibility_model.pkl")
    return joblib.load(model_path)

model = load_model()


st.set_page_config(
    page_title="News Credibility Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 News Credibility Detection System")
st.markdown("Analyze whether a news article is **Fake or Real** using an optimized ML model.")


option = st.radio(
    "Choose Input Method:",
    ("Enter Text Manually", "Upload .txt File")
)

news_text = ""

if option == "Enter Text Manually":
    news_text = st.text_area("Paste News Article Here", height=200)

elif option == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file is not None:
        news_text = uploaded_file.read().decode("utf-8")
        st.text_area("File Content", news_text, height=200)


threshold = st.slider(
    "Set Decision Threshold (Real Probability)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

st.markdown("---")


if st.button("🔍 Analyze News"):

    if news_text.strip() == "":
        st.warning("⚠ Please enter or upload some news text.")
    else:
        try:
            probabilities = model.predict_proba([news_text])[0]

            fake_prob = probabilities[0]
            real_prob = probabilities[1]

  
            if real_prob >= threshold:
                st.success(f"✅ Prediction: REAL ({real_prob*100:.2f}% confidence)")
            else:
                st.error(f"⚠ Prediction: FAKE ({fake_prob*100:.2f}% confidence)")


            st.subheader("Prediction Confidence")

            st.write(f"Real Probability: {real_prob:.4f}")
            st.write(f"Fake Probability: {fake_prob:.4f}")

            # Progress Bar
            st.progress(float(real_prob))

            # Bar Chart
            st.write("Real vs Fake Probability Distribution")
            st.bar_chart({
                "Real": real_prob,
                "Fake": fake_prob
            })

        except Exception as e:
            st.error(f"Error during prediction: {e}")