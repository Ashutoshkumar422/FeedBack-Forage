from collectors.google_play_collector import fetch_google_play
# from collectors.app_store_collector import fetch_app_store
from collectors.twitter_collector import fetch_twitter
# from collectors.reddit_collector import fetch_reddit
from utils.save_utils import save_data
import pandas as pd
import config

def run_all():
    # print("Collecting Google Play data...")
    # df_gp = fetch_google_play()

    # print("Collecting App Store data...")
    # df_ios = fetch_app_store()

    print("Collecting Twitter data...")
    df_tw = fetch_twitter(
        api_key=config.TWITTER_API_KEY,
        api_secret=config.TWITTER_API_SECRET,
        bearer_token=config.TWITTER_BEARER
    )

    # print("Collecting Reddit data...")
    # df_rd = fetch_reddit(
    #     client_id=config.REDDIT_CLIENT_ID,
    #     client_secret=config.REDDIT_SECRET,
    #     user_agent=config.REDDIT_AGENT
    # )

    df = pd.concat([ df_tw], ignore_index=True)

    print(f"TOTAL collected: {len(df)} records")
    save_data(df, filename="combined_reviews.csv")

if __name__ == "__main__":
    run_all()
