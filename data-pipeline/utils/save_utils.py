import pandas as pd
import os

def save_data(df, filename="combined_reviews.csv"):
    os.makedirs("data", exist_ok=True)
    filepath = f"data/{filename}"

    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)

    print(f"Saved {len(df)} records → {filepath}")
