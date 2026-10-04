import os
import glob
import pandas as pd
import re
from cleantext import clean
from langdetect import detect
from textblob import TextBlob

# -----------------------------
# 1. CONFIG
# -----------------------------
INPUT_FOLDER = "C:/Users/HP/Desktop/Interview Prep/Projects/InsightIO/data-pipeline/data/"       # folder where your CSVs are stored
OUTPUT_FILE = "cleaned_reviews.csv"

# -----------------------------
# 2. LOAD ALL CSV FILES
# -----------------------------
def load_all_files():
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    print(files)
    print(f"Found {len(files)} files")

    df_list = []
    for f in files:
        print("Loading:", f)
        try:
            # df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            df_list.append(df)
        except:
            print("Error loading:", f)
    
    if not df_list:
        raise Exception("No CSV files loaded!")

    return pd.concat(df_list, ignore_index=True)

# -----------------------------
# 3. NORMALIZE COLUMNS
# -----------------------------
def normalize_columns(df):

    # Standard schema for all reviews
    schema = {
        "review_text": ["review", "text", "content", "body", "comment"],
        "rating": ["rating", "score", "stars"],
        "author": ["user", "username", "author", "name"],
        "date": ["date", "timestamp", "created_at"],
        "app_version": ["version", "app_version"]
    }

    normalized = pd.DataFrame()

    for std_col, possible_cols in schema.items():
        for col in df.columns:
            if col.lower() in possible_cols:
                normalized[std_col] = df[col]
                break
        else:
            normalized[std_col] = None  # fill missing

    # Keep original source file
    normalized["source"] = df["source_file"]
    return normalized

# -----------------------------
# 4. CLEAN REVIEW TEXT
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    text = clean(
        text,
        to_ascii=False,
        lower=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_emoji=True
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text

# -----------------------------
# 5. ADD LANGUAGE + SENTIMENT
# -----------------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def sentiment_score(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0

# -----------------------------
# 6. FULL PROCESS PIPELINE
# -----------------------------
def process_pipeline():

    print("==== Loading all files ====")
    df_raw = load_all_files()

    print("Total raw rows:", len(df_raw))

    print("==== Normalizing columns ====")
    df = normalize_columns(df_raw)
    print(len(df))

    print("==== Cleaning text ====")
    df["review_text"] = df["review_text"].apply(clean_text)
    print(len(df))

    print("==== Removing empty rows ====")
    df = df[df["review_text"].str.len() > 3]
    print(len(df))

    print("==== Adding language detection ====")
    df["language"] = df["review_text"].apply(detect_language)
    print(len(df))

    print("==== Adding sentiment score ====")
    df["sentiment"] = df["review_text"].apply(sentiment_score)
    print(len(df))

    print("==== Removing duplicate reviews ====")
    df = df.drop_duplicates(subset=["review_text", "rating", "author", "date"])
    print(len(df))

    # print("Total rows:", len(df))
    # print("Unique review_text:", df["review_text"].nunique())
    # print("Unique rating:", df["rating"].nunique())
    # print("Unique author:", df["author"].nunique())
    # print("Unique date:", df["date"].nunique())


    print("==== Saving final cleaned dataset ====")
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Done! Saved cleaned dataset to {OUTPUT_FILE}")
    print("Final rows:", len(df))



# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    process_pipeline()
