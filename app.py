import streamlit as st
import pickle
import numpy as np

# ---------------------------
# Load model + vectorizer + label encoder
# ---------------------------
with open("outputs/logreg_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

label_encoder = pickle.load(open("outputs/label_encoder.pkl", "rb"))

# Sentiment color + emoji map
sentiment_colors = {
    "Negative": ("🔴 Negative", "#FF4C4C"),
    "Positive": ("🟢 Positive", "#4CAF50"),
    "Neutral": ("🟡 Neutral", "#FFD700"),
    "Irrelevant": ("⚪ Irrelevant", "#B0B0B0")
}

# ---------------------------
# Positive phrase enhancement
# ---------------------------
positive_phrases = [
    "love", "amazing", "great", "awesome", "fantastic", "wonderful", "best"
]

def handle_positive_phrases(text):
    # Check for positive phrases
    if any(phrase in text.lower() for phrase in positive_phrases):
        return "Positive"
    return None  # If no positive phrase found, proceed with model prediction

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="✨", layout="centered")

st.title("✨ Twitter Sentiment Analysis")
st.write("Type a sentence below and I’ll predict its **sentiment** with confidence scores.")

# User input
user_text = st.text_area("📝 Enter your text here:", height=100, placeholder="E.g. I love this product!")

if st.button("🔍 Analyze Sentiment"):
    if user_text.strip():
        # Check for positive phrases before model prediction
        sentiment = handle_positive_phrases(user_text)
        
        if sentiment is None:
            # Transform input for model if no positive phrases found
            X_input = vectorizer.transform([user_text])

            # Prediction
            pred = model.predict(X_input)[0]
            sentiment = label_encoder.inverse_transform([pred])[0]

        # Probabilities
        proba = model.predict_proba(X_input)[0] if sentiment == "Positive" else model.predict_proba(X_input)[0]
        proba_dict = {label_encoder.inverse_transform([i])[0]: p for i, p in enumerate(proba)}

        # Styled card for prediction
        label, color = sentiment_colors[sentiment]
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:white;">{label}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence scores
        st.subheader("📊 Confidence Scores:")
        for s, p in proba_dict.items():
            lbl, clr = sentiment_colors[s]
            st.markdown(f"**{lbl}**: {p*100:.2f}%")

    else:
        st.warning("⚠️ Please type something before analyzing.")

# Example test cases
st.markdown("---")
st.subheader("💡 Try with these examples:")

examples = [
    "This is the worst service ever.",
    "I absolutely love this product!",
    "It works as expected, nothing special.",
    "The capital of France is Paris."
]

cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex):
        # Check for positive phrases before model prediction
        sentiment = handle_positive_phrases(ex)
        
        if sentiment is None:
            X_input = vectorizer.transform([ex])
            pred = model.predict(X_input)[0]
            sentiment = label_encoder.inverse_transform([pred])[0]
        
        label, color = sentiment_colors[sentiment]

        st.markdown(
            f"""
            <div style="border:1px solid {color}; padding:10px; border-radius:8px; margin-top:10px;">
                <b>Input:</b> {ex} <br/>
                <b>Prediction:</b> <span style="color:{color};">{label}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>✨ Built with Streamlit | Project by Ramarao</p>",
    unsafe_allow_html=True
)
