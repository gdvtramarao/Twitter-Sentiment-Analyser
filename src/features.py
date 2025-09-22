import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os
from scipy.sparse import hstack  # for combining sparse TF-IDF with lexicon counts

# -----------------------------
# Load positive and negative words
# -----------------------------
with open("data/lexicon/positive-words.txt", "r") as f:
    positive_words = set(f.read().splitlines())

with open("data/lexicon/negative-words.txt", "r") as f:
    negative_words = set(f.read().splitlines())

# -----------------------------
# Function to compute lexicon features
# -----------------------------
def lexicon_features(text):
    words = str(text).lower().split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    return [pos_count, neg_count]

# -----------------------------
# Load datasets
# -----------------------------
train_file = "data/twitter_training.csv"
val_file = "data/twitter_validation.csv"

train_df = pd.read_csv(train_file, header=None)
val_df = pd.read_csv(val_file, header=None)

# The dataset has 4 columns: [id, entity, sentiment, text]
train_df.columns = ["id", "entity", "sentiment", "text"]
val_df.columns = ["id", "entity", "sentiment", "text"]

# Drop rows with missing text
train_df = train_df.dropna(subset=["text"])
val_df = val_df.dropna(subset=["text"])

# -----------------------------
# Encode labels (sentiment)
# -----------------------------
label_encoder = LabelEncoder()
train_df["label_encoded"] = label_encoder.fit_transform(train_df["sentiment"])
val_df["label_encoded"] = label_encoder.transform(val_df["sentiment"])

print("\nLabel mapping:")
for i, c in enumerate(label_encoder.classes_):
    print(f"{c} -> {i}")

# -----------------------------
# TF-IDF Vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

X_train_tfidf = vectorizer.fit_transform(train_df["text"])
X_val_tfidf = vectorizer.transform(val_df["text"])

# -----------------------------
# Lexicon features
# -----------------------------
train_lexicon_feats = np.array(train_df["text"].apply(lexicon_features).tolist())
val_lexicon_feats = np.array(val_df["text"].apply(lexicon_features).tolist())

# Combine TF-IDF with lexicon features
X_train_combined = hstack([X_train_tfidf, train_lexicon_feats])
X_val_combined = hstack([X_val_tfidf, val_lexicon_feats])

y_train = train_df["label_encoded"].values
y_val = val_df["label_encoded"].values

print("\nShapes:")
print("X_train_combined:", X_train_combined.shape)
print("X_val_combined:", X_val_combined.shape)

# -----------------------------
# Ensure outputs folder exists
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Save combined features
# -----------------------------
with open("outputs/X_train_combined.pkl", "wb") as f:
    pickle.dump(X_train_combined, f)
with open("outputs/X_val_combined.pkl", "wb") as f:
    pickle.dump(X_val_combined, f)
with open("outputs/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open("outputs/y_val.pkl", "wb") as f:
    pickle.dump(y_val, f)

# Save TF-IDF vectorizer and label encoder
with open("outputs/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("outputs/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nSaved TF-IDF vectorizer, label encoder, and combined features to outputs/")
