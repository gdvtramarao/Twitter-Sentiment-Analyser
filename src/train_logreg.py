import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load combined features
# -----------------------------
X_train = pickle.load(open("outputs/X_train_combined.pkl", "rb"))
X_val = pickle.load(open("outputs/X_val_combined.pkl", "rb"))
y_train = pickle.load(open("outputs/y_train.pkl", "rb"))
y_val = pickle.load(open("outputs/y_val.pkl", "rb"))

# Load TF-IDF vectorizer (for transforming new inputs later)
vectorizer = pickle.load(open("outputs/vectorizer.pkl", "rb"))

# Load label encoder
label_encoder = pickle.load(open("outputs/label_encoder.pkl", "rb"))

print("Training Logistic Regression...")

# Logistic Regression setup
logreg = LogisticRegression(
    max_iter=500,
    solver="saga",           
    multi_class="multinomial",
    n_jobs=-1,
    C=2.0,
    class_weight="balanced"
)

# Train model
logreg.fit(X_train, y_train)

# Evaluate
y_pred = logreg.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print("\nAccuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# -----------------------------
# Save model + vectorizer
# -----------------------------
os.makedirs("outputs", exist_ok=True)
with open("outputs/logreg_model.pkl", "wb") as f:
    pickle.dump((vectorizer, logreg, label_encoder), f)

print("\nSaved Logistic Regression model, vectorizer, and label encoder to outputs/logreg_model.pkl")
