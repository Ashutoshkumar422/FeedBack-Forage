from google_play_scraper import reviews
import pandas as pd

def fetch_google_play(app_id="com.phonepe.app", count=1000):
    result, _ = reviews(app_id, lang="en", country="in", count=count)

    df = pd.DataFrame(result)[["content", "score", "at", "reviewCreatedVersion"]]
    df.columns = ["text", "rating", "timestamp", "app_version"]

    df["source"] = "google_play"
    return df
