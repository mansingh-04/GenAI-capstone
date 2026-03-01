import streamlit as st
import os
import joblib
import re
import string
from collections import Counter


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

st.title("📰 Intelligent News Credibility Analysis")
st.markdown("Detect whether a news article is **Fake or Real** using an optimized Linear SVM model.")

st.markdown("---")


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

st.markdown("---")


if st.button("🔍 Analyze News"):

    if news_text.strip() == "":
        st.warning("⚠ Please enter or upload some news text.")
    else:
        try:
            prediction = model.predict([news_text])[0]
            decision_score = model.decision_function([news_text])[0]

            st.subheader("Prediction Result")

            if prediction == 1:
                st.success("✅ REAL News")
            else:
                st.error("⚠ FAKE News")

  
            st.subheader("Model Confidence")

            st.write(f"Decision Margin Score: {decision_score:.4f}")

            confidence_strength = abs(decision_score)

            if confidence_strength > 3:
                st.info("High Confidence Prediction")
            elif confidence_strength > 1:
                st.info("Moderate Confidence Prediction")
            else:
                st.warning("Low Confidence Prediction (Text may be ambiguous)")


            st.markdown("---")
            st.subheader("Text Analysis")

            words = news_text.split()
            word_count = len(words)
            unique_words = len(set(words))
            char_count = len(news_text)

            col1, col2, col3 = st.columns(3)

            col1.metric("Word Count", word_count)
            col2.metric("Unique Words", unique_words)
            col3.metric("Character Count", char_count)

  
            word_freq = Counter(words)
            common_words = dict(word_freq.most_common(10))

            st.write("Top 10 Most Frequent Words")
            st.bar_chart(common_words)


            st.markdown("---")
            st.subheader("Model Information")

            st.write("• Algorithm: Linear Support Vector Machine (LinearSVC)")
            st.write("• Feature Extraction: TF-IDF (Unigrams + Bigrams)")
            st.write("• Max Features: 15,000")
            st.write("• Hyperparameter Optimization: 3-Fold Cross Validation")
            st.write("• Evaluation Metric: F1-Score")

        except Exception as e:
            st.error(f"Error during prediction: {e}")