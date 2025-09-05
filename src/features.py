import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os

# Load datasets
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

# Encode labels (sentiment)
label_encoder = LabelEncoder()
train_df["label_encoded"] = label_encoder.fit_transform(train_df["sentiment"])
val_df["label_encoded"] = label_encoder.transform(val_df["sentiment"])

print("\nLabel mapping:")
for i, c in enumerate(label_encoder.classes_):
    print(f"{c} -> {i}")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

X_train_tfidf = vectorizer.fit_transform(train_df["text"])
X_val_tfidf = vectorizer.transform(val_df["text"])

y_train = train_df["label_encoded"].values
y_val = val_df["label_encoded"].values

print("\nShapes:")
print("X_train_tfidf:", X_train_tfidf.shape)
print("X_val_tfidf:", X_val_tfidf.shape)

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Save outputs
with open("outputs/X_train_tfidf.pkl", "wb") as f:
    pickle.dump(X_train_tfidf, f)
with open("outputs/X_val_tfidf.pkl", "wb") as f:
    pickle.dump(X_val_tfidf, f)
with open("outputs/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open("outputs/y_val.pkl", "wb") as f:
    pickle.dump(y_val, f)

# Save the fitted TF-IDF vectorizer
with open("outputs/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save label encoder
with open("outputs/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nSaved TF-IDF vectorizer and label encoder to outputs/")
