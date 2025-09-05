import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download nltk resources (only first time)
nltk.download("stopwords")
nltk.download("wordnet")

# ---------- load training data ----------
colnames = ["id", "entity", "sentiment", "text"]
train = pd.read_csv("data/twitter_training.csv", names=colnames, header=None, encoding="utf-8")

# drop id & entity (not useful)
train = train.drop(columns=["id", "entity"])

# drop rows with missing text
train = train.dropna(subset=["text"])

# drop duplicates
train = train.drop_duplicates(subset=["text"])

print("After dropping missing/duplicates:", train.shape)

# ---------- preprocessing ----------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text)
    text = re.sub(r'@\w+', '', text)        # remove mentions
    text = re.sub(r'http\S+', '', text)     # remove URLs
    text = re.sub(r'#', '', text)           # remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # keep only letters & spaces
    text = text.lower().strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

train["clean_text"] = train["text"].apply(clean_text)

print("\nSample cleaned tweets:")
print(train[["text", "clean_text"]].head(10))

# save cleaned dataset
train.to_csv("outputs/train_cleaned.csv", index=False)
print("\nSaved cleaned dataset: outputs/train_cleaned.csv")
