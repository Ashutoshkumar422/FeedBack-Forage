import tweepy
import pandas as pd

def fetch_twitter(api_key, api_secret, bearer_token, query="PhonePe", limit=400):

    client = tweepy.Client(bearer_token=bearer_token)

    tweets = client.search_recent_tweets(
        query=query,
        max_results=100,
        tweet_fields=["created_at", "lang"]
    )

    data = []
    for t in tweets.data:
        if t.lang == "en":
            data.append({
                "text": t.text,
                "rating": None,
                "timestamp": t.created_at,
                "app_version": None,
                "source": "twitter"
            })

    return pd.DataFrame(data)

