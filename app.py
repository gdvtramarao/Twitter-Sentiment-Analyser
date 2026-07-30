import pickle

import streamlit as st

from model_artifacts import LABEL_ENCODER_ARTIFACT_PATH, MODEL_ARTIFACT_PATH
from xquik_export import MAX_EXPORT_BYTES, XquikExportError, load_xquik_texts


# ---------------------------
# Load model + vectorizer + label encoder
# ---------------------------
@st.cache_resource
def load_model_artifacts():
    with MODEL_ARTIFACT_PATH.open("rb") as artifact:
        loaded_vectorizer, loaded_model = pickle.load(artifact)

    with LABEL_ENCODER_ARTIFACT_PATH.open("rb") as artifact:
        loaded_label_encoder = pickle.load(artifact)

    return loaded_vectorizer, loaded_model, loaded_label_encoder


vectorizer, model, label_encoder = load_model_artifacts()

# Sentiment color + emoji map
sentiment_colors = {
    "Negative": ("🔴 Negative", "#FF4C4C"),
    "Positive": ("🟢 Positive", "#4CAF50"),
    "Neutral": ("🟡 Neutral", "#FFD700"),
    "Irrelevant": ("⚪ Irrelevant", "#B0B0B0"),
}

# ---------------------------
# Positive phrase enhancement
# ---------------------------
positive_phrases = [
    "love",
    "amazing",
    "great",
    "awesome",
    "fantastic",
    "wonderful",
    "best",
]


def handle_positive_phrases(text: str) -> str | None:
    # Check for positive phrases
    if any(phrase in text.lower() for phrase in positive_phrases):
        return "Positive"
    return None  # If no positive phrase found, proceed with model prediction


def predict_sentiment(text: str) -> tuple[str, dict[str, float]]:
    sentiment = handle_positive_phrases(text)
    if sentiment is not None:
        return sentiment, {}

    model_input = vectorizer.transform([text])
    prediction = model.predict(model_input)[0]
    sentiment = str(label_encoder.inverse_transform([prediction])[0])
    probabilities = model.predict_proba(model_input)[0]
    scores = {
        str(label_encoder.inverse_transform([index])[0]): float(probability)
        for index, probability in enumerate(probabilities)
    }
    return sentiment, scores


def predict_sentiments(texts: list[str]) -> list[str]:
    sentiments = [""] * len(texts)
    model_positions: list[int] = []
    model_texts: list[str] = []

    for position, text in enumerate(texts):
        sentiment = handle_positive_phrases(text)
        if sentiment is None:
            model_positions.append(position)
            model_texts.append(text)
        else:
            sentiments[position] = sentiment

    if model_texts:
        predictions = model.predict(vectorizer.transform(model_texts))
        labels = label_encoder.inverse_transform(predictions)
        for position, label in zip(model_positions, labels):
            sentiments[position] = str(label)

    return sentiments


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="✨",
    layout="centered",
)

st.title("✨ Twitter Sentiment Analysis")
st.write(
    "Type a sentence below and I’ll predict its **sentiment** with confidence scores."
)

# User input
user_text = st.text_area(
    "📝 Enter your text here:",
    height=100,
    placeholder="E.g. I love this product!",
)

if st.button("🔍 Analyze Sentiment"):
    if user_text.strip():
        sentiment, probability_scores = predict_sentiment(user_text)
        label, color = sentiment_colors[sentiment]
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:white;">{label}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if probability_scores:
            st.subheader("📊 Confidence Scores:")
            for sentiment_name, probability in probability_scores.items():
                score_label, _ = sentiment_colors[sentiment_name]
                st.markdown(f"**{score_label}**: {probability * 100:.2f}%")

    else:
        st.warning("⚠️ Please type something before analyzing.")

st.markdown("---")
st.subheader("Analyze Xquik Export")
st.caption(
    "Upload an Xquik extraction JSON or CSV file, or a JSONL file from the Xquik CLI."
)
xquik_export = st.file_uploader(
    "Upload Xquik export",
    type=["json", "jsonl", "csv"],
)

if xquik_export is not None:
    if xquik_export.size > MAX_EXPORT_BYTES:
        st.error("Xquik export exceeds the 10 MB limit.")
    else:
        try:
            texts = load_xquik_texts(xquik_export.getvalue())
        except XquikExportError as error:
            st.error(f"Invalid Xquik export. {error}")
        else:
            if not texts:
                st.warning("No tweet text found in the uploaded export.")
            else:
                sentiments = predict_sentiments(texts)
                st.dataframe(
                    [
                        {
                            "text": text,
                            "sentiment": sentiment,
                        }
                        for text, sentiment in zip(texts, sentiments)
                    ]
                )

# Example test cases
st.markdown("---")
st.subheader("💡 Try with these examples:")

examples = [
    "This is the worst service ever.",
    "I absolutely love this product!",
    "It works as expected, nothing special.",
    "The capital of France is Paris.",
]

cols = st.columns(2)
for index, example in enumerate(examples):
    if cols[index % 2].button(example):
        sentiment, _ = predict_sentiment(example)
        label, color = sentiment_colors[sentiment]

        st.markdown(
            f"""
            <div style="border:1px solid {color}; padding:10px; border-radius:8px; margin-top:10px;">
                <b>Input:</b> {example} <br/>
                <b>Prediction:</b> <span style="color:{color};">{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>✨ Built with Streamlit | Project by gdvtramarao</p>",
    unsafe_allow_html=True,
)
