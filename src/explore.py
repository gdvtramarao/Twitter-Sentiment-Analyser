# src/step3_explore.py
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1) find CSV files in data/ ----------
csv_files = glob.glob("data/*.csv")
print("CSV files found:", csv_files)
if not csv_files:
    raise SystemExit("No CSV files found in data/. Put your CSVs inside the data/ folder.")

def choose_file(files, keywords):
    for kw in keywords:
        for f in files:
            if kw in os.path.basename(f).lower():
                return f
    return None

train_file = choose_file(csv_files, ["train", "training"]) or csv_files[0]
val_file   = choose_file(csv_files, ["val", "validation", "test"])   or (csv_files[1] if len(csv_files) > 1 else None)

print("Using train file:", train_file)
print("Using validation file:", val_file)

# ---------- 2) load files with custom headers ----------
colnames = ["id", "entity", "sentiment", "text"]

train = pd.read_csv(train_file, names=colnames, header=None, encoding="utf-8")
val = pd.read_csv(val_file, names=colnames, header=None, encoding="utf-8") if val_file else None

print("\n--- TRAIN (with headers) ---")
print(train.head())
print(train.dtypes)


# ---------- 3) detect likely text & label columns ----------
def guess_columns(df):
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ['text','tweet','content','review','comment'])]
    label_cols = [c for c in df.columns if any(k in c.lower() for k in ['sentiment','label','target','polarity','class','rating'])]
    return text_cols, label_cols

t_cols, l_cols = guess_columns(train)
print("\nLikely text columns (train):", t_cols)
print("Likely label columns (train):", l_cols)

# pick the first sensible candidate (or raise a note)
if not t_cols:
    raise SystemExit("No obvious text column found in training file. Inspect `train.columns` above and set `text_col` manually.")
if not l_cols:
    print("Warning: No obvious label column found. This dataset may be unlabeled or the label column has a non-standard name.")

text_col = t_cols[0]
label_col = l_cols[0] if l_cols else None

print("\nSelected text column:", text_col)
print("Selected label column:", label_col)

# ---------- 4) quick look & missing values ----------
print("\n--- Head (first 5 rows) ---")
print(train[[text_col] + ([label_col] if label_col else [])].head())

print("\n--- column dtypes ---")
print(train.dtypes)

print("\n--- missing values (train) ---")
print(train[[text_col] + ([label_col] if label_col else [])].isna().sum())

# show a few rows with missing text or label (if any)
missing_text = train[train[text_col].isna()]
if not missing_text.empty:
    print("\nRows with missing text (show up to 5):")
    print(missing_text.head())

if label_col:
    missing_label = train[train[label_col].isna()]
    if not missing_label.empty:
        print("\nRows with missing label (show up to 5):")
        print(missing_label.head())

# ---------- 5) label value inspection ----------
if label_col:
    print("\nLabel value counts (train):")
    print(train[label_col].value_counts(dropna=False))

    unique_labels = sorted(train[label_col].dropna().unique().tolist())
    print("Unique labels:", unique_labels)
else:
    unique_labels = []

# ---------- 6) basic text stats ----------
# convert to str to avoid errors
train['__text_str'] = train[text_col].astype(str)
train['text_word_count'] = train['__text_str'].str.split().apply(len)
train['text_char_count'] = train['__text_str'].str.len()

print("\nText length stats (words):")
print(train['text_word_count'].describe())

# sample longest & shortest tweets
print("\nTop-5 longest (words):")
print(train.sort_values('text_word_count', ascending=False)[[text_col]].head())

print("\nTop-5 shortest (words):")
print(train.sort_values('text_word_count', ascending=True)[[text_col]].head())

# ---------- 7) duplicates ----------
dupe_count = train.duplicated(subset=[text_col]).sum()
print(f"\nDuplicate tweets (by text) in train: {dupe_count}")
if dupe_count > 0:
    print("Example duplicates:")
    print(train[train.duplicated(subset=[text_col], keep=False)].head(6))

# ---------- 8) quick plots (saved to outputs/) ----------
os.makedirs("outputs", exist_ok=True)

# text length histogram
plt.figure()
plt.hist(train['text_word_count'].clip(upper=1000), bins=30)  # clip very long to keep plot readable
plt.xlabel("Words per tweet")
plt.ylabel("Count")
plt.title("Text length distribution (train)")
plt.tight_layout()
plt.savefig("outputs/text_length_hist.png")
plt.close()
print("Saved: outputs/text_length_hist.png")

# label distribution barplot
if label_col:
    plt.figure()
    train[label_col].value_counts().plot(kind='bar')
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Label distribution (train)")
    plt.tight_layout()
    plt.savefig("outputs/label_distribution.png")
    plt.close()
    print("Saved: outputs/label_distribution.png")

# ---------- 9) save a human-inspectable sample ----------
train.sample(200, random_state=42)[[text_col] + ([label_col] if label_col else [])].to_csv("outputs/train_sample_200.csv", index=False)
print("Saved a sample for manual inspection: outputs/train_sample_200.csv")

# ---------- 10) summary advice printed ----------
print("\n>>> SUMMARY / things to check next:")
print(" - Are text rows empty or full of HTML/URLs/emojis/garbage? (open outputs/train_sample_200.csv)")
if label_col:
    print(" - Are labels numeric or strings? Inspect `Unique labels:` printed above.")
else:
    print(" - You do not seem to have a label column in the training file. If the dataset is unlabeled, we need a labeled dataset for supervised training.")
print(" - Is the class distribution heavily imbalanced? (check outputs/label_distribution.png)")
print(" - Any large fraction of duplicates or missing text? (see counts above)")
print("\nWhen you're done inspecting, tell me `done step 3` and I will give Step 4 (cleaning & preprocessing).")
