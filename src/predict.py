import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load model, vectorizer, label encoder
# -----------------------------
with open("outputs/logreg_model.pkl", "rb") as f:
    vectorizer, model, label_encoder = pickle.load(f)

# Load positive and negative words for lexicon features
with open("data/lexicon/positive-words.txt", "r") as f:
    positive_words = set(f.read().splitlines())

with open("data/lexicon/negative-words.txt", "r") as f:
    negative_words = set(f.read().splitlines())

# -----------------------------
# Lexicon feature extraction
# -----------------------------
def lexicon_features(text):
    words = str(text).lower().split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    return [pos_count, neg_count]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="✨", layout="centered")
st.title("✨ Twitter Sentiment Analysis")
st.write("Type a sentence below and I’ll predict its **sentiment** (Positive, Negative, Neutral, Irrelevant).")

# User input
user_text = st.text_area("📝 Enter your text here:", height=100, placeholder="E.g. I love this product!")

if st.button("🔍 Analyze Sentiment"):
    if user_text.strip():
        # TF-IDF features
        tfidf_feat = vectorizer.transform([user_text])
        # Lexicon features
        lex_feat = np.array(lexicon_features(user_text)).reshape(1, -1)
        # Combine
        X_input = np.hstack([tfidf_feat.toarray(), lex_feat])
        
        # Predict
        pred = model.predict(X_input)[0]
        sentiment = label_encoder.inverse_transform([pred])[0]
        
        # Sentiment colors
        sentiment_colors = {
            "Negative": "🔴 Negative",
            "Positive": "🟢 Positive",
            "Neutral": "🟡 Neutral",
            "Irrelevant": "⚪ Irrelevant"
        }
        
        st.subheader("Prediction:")
        st.markdown(
            f"<h2 style='color:darkblue;'>{sentiment_colors[sentiment]}</h2>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please type something before analyzing.")

# -----------------------------
# Example test cases
# -----------------------------
st.markdown("---")
st.subheader("💡 Try with these examples:")

examples = [
    "This is the worst service ever.",
    "I absolutely love this product!",
    "It works as expected, nothing special.",
    "The capital of France is Paris."
]

for ex in examples:
    if st.button(ex):
        tfidf_feat = vectorizer.transform([ex])
        lex_feat = np.array(lexicon_features(ex)).reshape(1, -1)
        X_input = np.hstack([tfidf_feat.toarray(), lex_feat])
        pred = model.predict(X_input)[0]
        sentiment = label_encoder.inverse_transform([pred])[0]
        st.markdown(
            f"<b>Input:</b> {ex} <br/> <b>Prediction:</b> {sentiment_colors[sentiment]}",
            unsafe_allow_html=True
        )
