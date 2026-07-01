import streamlit as st
import pickle
import numpy as np
import pandas as pd

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

TEXT_COLUMN_CANDIDATES = [
    "tweet",
    "text",
    "content",
    "message",
    "body",
    "post",
    "comment",
    "review"
]

def handle_positive_phrases(text):
    # Check for positive phrases
    if any(phrase in text.lower() for phrase in positive_phrases):
        return "Positive"
    return None  # If no positive phrase found, proceed with model prediction

def detect_text_column(columns):
    normalized_columns = {str(column).strip().lower(): column for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return columns[0] if len(columns) > 0 else None

def predict_text(text):
    sentiment = handle_positive_phrases(text)
    if sentiment is not None:
        return sentiment, {}

    X_input = vectorizer.transform([text])
    pred = model.predict(X_input)[0]
    sentiment = label_encoder.inverse_transform([pred])[0]

    proba = model.predict_proba(X_input)[0]
    proba_dict = {
        label_encoder.inverse_transform([i])[0]: probability
        for i, probability in enumerate(proba)
    }

    return sentiment, proba_dict

def render_prediction(text, sentiment, proba_dict):
    label, color = sentiment_colors[sentiment]
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
            <h2 style="color:white;">{label}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    if proba_dict:
        st.subheader("📊 Confidence Scores:")
        for item, probability in proba_dict.items():
            lbl, _ = sentiment_colors[item]
            st.markdown(f"**{lbl}**: {probability*100:.2f}%")
    else:
        st.info(f"Rule-based positive phrase match used for: {text}")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="✨", layout="centered")

st.title("✨ Twitter Sentiment Analysis")
st.write("Type a sentence below and I’ll predict its **sentiment** with confidence scores.")

st.subheader("Batch CSV Prediction")
uploaded_file = st.file_uploader(
    "Upload a CSV with tweet, text, content, message, body, post, comment, or review text",
    type=["csv"]
)

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    detected_column = detect_text_column(batch_df.columns)

    if detected_column is None:
        st.warning("Upload a CSV with at least one text column.")
    else:
        text_column = st.selectbox(
            "Text column",
            batch_df.columns,
            index=list(batch_df.columns).index(detected_column)
        )
        usable_rows = batch_df[batch_df[text_column].astype(str).str.strip() != ""].copy()

        if usable_rows.empty:
            st.warning("No non-empty text rows were found.")
        elif st.button("Analyze CSV"):
            predictions = []
            for text in usable_rows[text_column].astype(str):
                sentiment, proba_dict = predict_text(text)
                predictions.append({
                    "predicted_sentiment": sentiment,
                    "confidence": max(proba_dict.values()) if proba_dict else None
                })

            results_df = usable_rows.copy()
            predictions_df = pd.DataFrame(predictions, index=usable_rows.index)
            for column in predictions_df.columns:
                results_df[column] = predictions_df[column]
            st.dataframe(results_df.head(25), use_container_width=True)
            st.download_button(
                "Download predictions",
                results_df.to_csv(index=False).encode("utf-8"),
                "tweet_sentiment_predictions.csv",
                "text/csv"
            )

# User input
user_text = st.text_area("📝 Enter your text here:", height=100, placeholder="E.g. I love this product!")

if st.button("🔍 Analyze Sentiment"):
    if user_text.strip():
        sentiment, proba_dict = predict_text(user_text)
        render_prediction(user_text, sentiment, proba_dict)
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
        sentiment, _ = predict_text(ex)

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
    "<p style='text-align:center; color:grey;'>✨ Built with Streamlit | Project by gdvtramarao</p>",
    unsafe_allow_html=True
)
