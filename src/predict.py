import pickle
import sys
import numpy as np

# Check if text is passed as command-line argument
if len(sys.argv) < 2:
    print("Usage: python src/predict.py \"Your text here\"")
    sys.exit(1)

text = sys.argv[1]

# Load the saved (vectorizer, model) tuple
with open("outputs/logreg_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Load label encoder
with open("outputs/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Transform input text
X_input = vectorizer.transform([text])

# Predict
y_pred = model.predict(X_input)[0]
y_label = label_encoder.inverse_transform([y_pred])[0]

# Predict probabilities
probs = model.predict_proba(X_input)[0]

print(f"\nInput Text: {text}")
print(f"Predicted Sentiment: {y_label}\n")

print("Confidence Scores:")
for label, prob in zip(label_encoder.classes_, probs):
    print(f"  {label}: {prob:.4f}")
