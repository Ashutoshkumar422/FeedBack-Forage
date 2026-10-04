import requests
import pandas as pd
import time

def fetch_all_app_reviews(app_id, country='us', max_pages=50):
    """
    Fetch reviews with automatic pagination
    """
    all_reviews = []
    
    for page in range(1, max_pages + 1):
        print(f"Fetching page {page}...")
        url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"Failed to fetch page {page}")
                break
                
            data = response.json()
            
            # Check if there are entries
            if 'entry' not in data.get('feed', {}):
                print(f"No more reviews found at page {page}")
                break
            
            entries = data['feed']['entry']
            
            for entry in entries:
                review = {
                    'author': entry.get('author', {}).get('name', {}).get('label', 'N/A'),
                    'rating': entry.get('im:rating', {}).get('label', 'N/A'),
                    'title': entry.get('title', {}).get('label', 'N/A'),
                    'content': entry.get('content', {}).get('label', 'N/A'),
                    'date': entry.get('updated', {}).get('label', 'N/A'),
                    'version': entry.get('im:version', {}).get('label', 'N/A'),
                    'vote_count': entry.get('im:voteCount', {}).get('label', '0')
                }
                all_reviews.append(review)
            
            # Be respectful with rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
    
    return all_reviews

# Example usage
app_id = "1170055821"  # PhonePe
reviews = fetch_all_app_reviews(app_id, country='in', max_pages=100)

# Save to CSV
df = pd.DataFrame(reviews)
df.to_csv('app_store_reviews_complete.csv', index=False)
print(f"\nTotal reviews scraped: {len(reviews)}")
print(f"\nRating distribution:")
print(df['rating'].value_counts().sort_index())