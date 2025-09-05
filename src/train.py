import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# ========================
# 1. Load cleaned dataset
# ========================
df = pd.read_csv("outputs/train_cleaned.csv")

# Ensure no NaN in clean_text
df["clean_text"] = df["clean_text"].fillna("").astype(str)

# ========================
# 2. Load encoders
# ========================
tfidf = joblib.load("outputs/tfidf_vectorizer.pkl")
le = joblib.load("outputs/label_encoder.pkl")

# Encode labels
df["label_encoded"] = le.transform(df["sentiment"])

# ========================
# 3. Train/Validation Split
# ========================
X_train, X_val, y_train, y_val = train_test_split(
    df["clean_text"], df["label_encoded"], test_size=0.2, random_state=42, stratify=df["label_encoded"]
)

# Vectorize text
X_train_tfidf = tfidf.transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

# ========================
# 4. Train Logistic Regression
# ========================
print("\nTraining Logistic Regression...")
clf = LogisticRegression(max_iter=1000, solver="liblinear")
clf.fit(X_train_tfidf, y_train)

# ========================
# 5. Evaluation
# ========================
y_pred = clf.predict(X_val_tfidf)

print("\nAccuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# ========================
# 6. Save Model
# ========================
joblib.dump(clf, "outputs/logreg_model.pkl")
print("\nSaved Logistic Regression model to outputs/logreg_model.pkl")
