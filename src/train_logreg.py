import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed features
X_train = pickle.load(open("outputs/X_train_tfidf.pkl", "rb"))
X_val = pickle.load(open("outputs/X_val_tfidf.pkl", "rb"))
y_train = pickle.load(open("outputs/y_train.pkl", "rb"))
y_val = pickle.load(open("outputs/y_val.pkl", "rb"))

print("Training Logistic Regression...")

# Logistic Regression (multinomial, better for multi-class problems)
logreg = LogisticRegression(
    max_iter=500,
    solver="saga",           # handles multinomial & large sparse data
    multi_class="multinomial",
    n_jobs=-1,
    C=2.0,                   # slightly less regularization (improves accuracy)
    class_weight="balanced"  # helps with class imbalance
)

# Train
logreg.fit(X_train, y_train)

# Evaluate
y_pred = logreg.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print("\nAccuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# Load the fitted vectorizer (from features.py)
vectorizer = pickle.load(open("outputs/vectorizer.pkl", "rb"))

# Save both model and vectorizer together
os.makedirs("outputs", exist_ok=True)
with open("outputs/logreg_model.pkl", "wb") as f:
    pickle.dump((vectorizer, logreg), f)

print("\n✅ Saved Logistic Regression model and vectorizer to outputs/logreg_model.pkl")
