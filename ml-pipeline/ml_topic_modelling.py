import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# Topic Modeling
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Sentiment Analysis
from transformers import pipeline
import torch

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "C:/Users/HP/Desktop/Interview Prep/Projects/InsightIO/augmented_reviews_with_llm.csv"
OUTPUT_FILE = "reviews_with_topics_and_sentiment1.csv"
MODEL_SAVE_PATH = "models/"

# -----------------------------
# 1. LOAD AND PREPARE DATA
# -----------------------------
def load_augmented_data(filepath):
    """Load augmented dataset"""
    print("="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"\n✅ Loaded {len(df)} records")
    print(f"   - Original: {len(df[df['is_augmented']==False])}")
    print(f"   - Augmented: {len(df[df['is_augmented']==True])}")
    
    # Show data info
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample review:")
    print(f"'{df['review_text'].iloc[0]}'")
    
    return df

# -----------------------------
# 2. TEXT PREPROCESSING FOR TOPIC MODELING
# -----------------------------
def preprocess_for_topics(text):
    """Clean text for better topic modeling"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# -----------------------------
# 3. IMPROVED BERTOPIC - TOPIC MODELING
# -----------------------------
def perform_topic_modeling(df, num_topics=6):
    """
    Use BERTopic to discover topics in feedback
    IMPROVED VERSION with better clustering parameters
    """
    print("\n" + "="*60)
    print("STEP 2: TOPIC MODELING (BERTopic - IMPROVED)")
    print("="*60)
    
    # Prepare texts
    print("\n📝 Preprocessing texts...")
    df['text_processed'] = df['review_text'].apply(preprocess_for_topics)
    texts = df['text_processed'].tolist()
    
    print(f"✅ Processed {len(texts)} texts")
    
    # Initialize embedding model
    print("\n🤖 Loading sentence transformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # IMPROVED: Custom vectorizer with better parameters
    print("\n📊 Configuring vectorizer...")
    vectorizer_model = CountVectorizer(
        stop_words='english',
        min_df=2,  # Word must appear in at least 2 documents
        max_df=0.85,  # NEW: Ignore words appearing in >85% of docs
        ngram_range=(1, 3),  # NEW: Include phrases (1-3 words)
        max_features=2000  # NEW: Larger vocabulary
    )
    
    # IMPROVED: Custom UMAP for better dimensionality reduction
    print("\n🗺️  Configuring UMAP for dimensionality reduction...")
    umap_model = UMAP(
        n_neighbors=15,  # Balance between local and global structure
        n_components=5,  # Standard for topic modeling
        min_dist=0.0,  # Tight clusters
        metric='cosine',  # Best for text embeddings
        random_state=42
    )
    
    # IMPROVED: Custom HDBSCAN for better clustering
    print("\n🔍 Configuring HDBSCAN for clustering...")
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,  # DECREASED from default 15
        min_samples=5,  # Minimum samples in neighborhood
        metric='euclidean',
        cluster_selection_method='eom',  # Excess of Mass
        prediction_data=True
    )
    
    # Initialize BERTopic with improved settings
    print("\n🎯 Training BERTopic model...")
    print(f"   Target topics: {num_topics}")
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,  # NEW: Custom UMAP
        hdbscan_model=hdbscan_model,  # NEW: Custom HDBSCAN
        nr_topics=num_topics,  # Force specific number of topics
        min_topic_size=10,  # CHANGED from 15 to 10
        top_n_words=10,  # Show more keywords per topic
        verbose=True,
        calculate_probabilities=True
    )
    
    # Fit model
    topics, probabilities = topic_model.fit_transform(texts)
    
    # DEBUG: Check what was discovered
    unique_topics = len(set(topics))
    print(f"\n🔍 Initial discovery: {unique_topics} unique topics (including outliers)")
    
    # Add results to dataframe
    df['topic'] = topics
    df['topic_probability'] = [prob.max() for prob in probabilities]
    
    print("\n✅ Topic modeling complete!")
    
    # Show topic distribution
    print("\n📊 Topic Distribution:")
    topic_counts = df['topic'].value_counts().sort_index()
    for topic_id, count in topic_counts.items():
        percentage = (count / len(df)) * 100
        if topic_id == -1:
            print(f"   Topic {topic_id} (Outliers): {count} documents ({percentage:.1f}%)")
        else:
            print(f"   Topic {topic_id}: {count} documents ({percentage:.1f}%)")
    
    # Get topic info with more detail
    topic_info = topic_model.get_topic_info()
    
    print("\n🏷️  Topic Keywords (Top 8 per topic):")
    for topic_id in sorted(df['topic'].unique()):
        if topic_id != -1:  # -1 is outlier topic
            keywords = topic_model.get_topic(topic_id)
            top_words = [word for word, _ in keywords[:8]]
            print(f"   Topic {topic_id}: {', '.join(top_words)}")
            
            # Show 2 sample reviews
            samples = df[df['topic'] == topic_id].head(2)
            for idx, sample in samples.iterrows():
                print(f"      Sample: '{sample['review_text'][:70]}...'")
    
    return df, topic_model, topic_info

# -----------------------------
# 4. IMPROVED TOPIC MAPPING
# -----------------------------
def map_topics_to_categories(df, topic_model):
    """
    Map discovered topics to business categories
    IMPROVED with scoring system and expanded keywords
    """
    print("\n" + "="*60)
    print("STEP 3: MAPPING TOPICS TO CATEGORIES")
    print("="*60)
    
    # Expanded keyword dictionary for better matching
    category_keywords = {
        'payment_issues': [
            'payment', 'pay', 'paid', 'transaction', 'refund', 'money', 
            'amount', 'deducted', 'deduct', 'failed', 'fail', 'bank', 
            'wallet', 'checkout', 'purchase', 'bought', 'charge', 'credit'
        ],
        'login_problems': [
            'login', 'signin', 'sign', 'password', 'otp', 'verify',
            'verification', 'access', 'account', 'unlock', 'forgot', 
            'reset', 'locked', 'authenticate', 'registration', 'register', 'not working'
        ],
        'app_crashes_bugs': [
            'crash', 'crashes', 'crashing','crashed' 'freeze', 'freezes', 'frozen',
            'bug', 'bugs', 'error', 'errors', 'issue', 'problem', 
            'hang', 'stuck', 'loading', 'load', 'slow', 'lag', 'laggy',
            'response', 'glitch', 'malfunction','not opening','white screen','black screen','slow','poor'
        ],
        'ui_ux_issues': [
            'interface', 'ui', 'ux', 'design', 'layout', 'button', 'screen',
            'confusing', 'complicated', 'difficult', 'hard', 'navigation','outdated',
            'navigate', 'menu', 'find', 'locate', 'understand', 'clarity', 'not accessible','notifications','glitch','annoying'
        ],
        'fraud_security': [
            'fraud', 'fraudulent', 'scam', 'scammer', 'security', 'secure',
            'hack', 'hacked', 'hacker', 'unsafe', 'safe', 'trust', 
            'suspicious', 'fake', 'phishing', 'steal', 'stolen', 'breach','frustrating','loss','fake'
        ],
        'cashback_rewards': [
            'cash back', 'cash', 'back', 'reward', 'rewards', 'points',
            'offer', 'offers', 'discount', 'discounts', 'promo', 'promotion',
            'deal', 'deals', 'bonus', 'credit', 'voucher', 'coupon','no cashbacks'
        ],
        'positive_feedback': [
            'good','super','great','stars','five stars','excellent','support','user friendly','easy','smooth','smart','fast','awesome','secure'
        ]
    }
    
    topic_mapping = {}
    mapping_confidence = {}
    
    print("\n🔍 Analyzing topics and calculating match scores...")
    
    for topic_id in sorted(df['topic'].unique()):
        if topic_id == -1:
            topic_mapping[-1] = "uncategorized"
            mapping_confidence[-1] = 0.0
            continue
        
        # Get top words for this topic
        keywords = topic_model.get_topic(topic_id)
        topic_words = [word.lower() for word, _ in keywords[:15]]
        
        print(f"\n📌 Topic {topic_id}:")
        print(f"   Keywords: {', '.join(topic_words[:5])}")
        
        # Calculate match scores for each category
        category_scores = {}
        for category, cat_keywords in category_keywords.items():
            # Exact matches
            exact_matches = sum(1 for word in topic_words if word in cat_keywords)
            
            # Partial matches (e.g., "payment" matches "payments")
            partial_matches = sum(1 for topic_word in topic_words 
                                for cat_word in cat_keywords 
                                if cat_word in topic_word or topic_word in cat_word)
            
            # Score: exact matches worth 1.0, partial matches worth 0.5
            score = exact_matches + (partial_matches * 0.5)
            category_scores[category] = score
        
        # Assign to category with highest score
        if max(category_scores.values()) > 0:
            best_category = max(category_scores, key=category_scores.get)
            confidence = category_scores[best_category] / len(topic_words)
            
            topic_mapping[topic_id] = best_category
            mapping_confidence[topic_id] = confidence
            
            print(f"   ✅ Mapped to: {best_category} (confidence: {confidence:.2f})")
            
            # Show top 3 category scores for transparency
            top_3 = dict(sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3])
            print(f"   └─ Top scores: {top_3}")
        else:
            topic_mapping[topic_id] = "other_issues"
            mapping_confidence[topic_id] = 0.0
            print(f"   ⚠️  Mapped to: other_issues (no clear match)")
    
    # Apply mapping
    df['topic_category'] = df['topic'].map(topic_mapping)
    df['mapping_confidence'] = df['topic'].map(mapping_confidence)
    
    print("\n✅ Topic mapping complete!")
    print("\n📊 Initial Category Distribution:")
    category_dist = df['topic_category'].value_counts()
    for category, count in category_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    return df, topic_mapping

# -----------------------------
# 5. KEYWORD-BASED RECLASSIFICATION
# -----------------------------
def keyword_reclassify(df):
    """
    Reclassify 'other_issues' and 'uncategorized' using direct keyword matching
    This catches reviews that BERTopic couldn't properly cluster
    """
    print("\n" + "="*60)
    print("STEP 3.5: KEYWORD-BASED RECLASSIFICATION")
    print("="*60)
    
    def classify_by_keywords(text):
        """Direct keyword matching as fallback"""
        text_lower = str(text).lower()
        
        # Check in order of priority (most specific first)
        if any(word in text_lower for word in [
            'payment', 'pay', 'paid', 'transaction', 'refund', 'money', 
            'amount', 'deducted', 'deduct', 'failed', 'fail', 'bank', 
            'wallet', 'checkout', 'purchase', 'bought', 'charge', 'credit'
        ]):
            return 'payment_issues'
        elif any(word in text_lower for word in [
            'login', 'signin', 'sign', 'password', 'otp', 'verify',
            'verification', 'access', 'account', 'unlock', 'forgot', 
            'reset', 'locked', 'authenticate', 'registration', 'register', 'not working'
        ]):
            return 'login_problems'
        elif any(word in text_lower for word in [
            'crash', 'crashes', 'crashing', 'freeze', 'freezes', 'frozen',
            'bug', 'bugs', 'error', 'errors', 'issue', 'problem', 
            'hang', 'stuck', 'loading', 'load', 'slow', 'lag', 'laggy',
            'response', 'glitch', 'malfunction','not opening','white screen','black screen','slow','poor'
        ]):
            return 'app_crashes_bugs'
        elif any(word in text_lower for word in [
            'interface', 'ui', 'ux', 'design', 'layout', 'button', 'screen',
            'confusing', 'complicated', 'difficult', 'hard', 'navigation','outdated',
            'navigate', 'menu', 'find', 'locate', 'understand', 'clarity', 'not accessible'
        ]):
            return 'ui_ux_issues'
        elif any(word in text_lower for word in [
            'fraud', 'fraudulent', 'scam', 'scammer', 'security', 'secure',
            'hack', 'hacked', 'hacker', 'unsafe', 'safe', 'trust', 
            'suspicious', 'fake', 'phishing', 'steal', 'stolen', 'breach','frustrating','loss','fake'
        ]):
            return 'fraud_security'
        elif any(word in text_lower for word in [
            'cash back', 'cash', 'back', 'reward', 'rewards', 'points',
            'offer', 'offers', 'discount', 'discounts', 'promo', 'promotion',
            'deal', 'deals', 'bonus', 'credit', 'voucher', 'coupon'
        ]):
            return 'cashback_rewards'
        else:
            return 'other_issues'
    
    # Only reclassify problematic categories
    mask = df['topic_category'].isin(['other_issues', 'uncategorized'])
    before_count = mask.sum()
    
    if before_count > 0:
        print(f"\n🔄 Reclassifying {before_count} reviews using keyword matching...")
        
        df.loc[mask, 'topic_category'] = df.loc[mask, 'review_text'].apply(classify_by_keywords)
        
        after_count = df['topic_category'].isin(['other_issues', 'uncategorized']).sum()
        reclassified = before_count - after_count
        
        print(f"✅ Successfully reclassified: {reclassified} reviews")
        print(f"   Remaining in 'other_issues': {after_count}")
        
        if reclassified > 0:
            print("\n📊 Updated Category Distribution:")
            category_dist = df['topic_category'].value_counts()
            for category, count in category_dist.items():
                percentage = (count / len(df)) * 100
                print(f"   {category}: {count} ({percentage:.1f}%)")
    else:
        print("✅ No reclassification needed - all reviews properly categorized!")
    
    return df

# -----------------------------
# 6. SENTIMENT ANALYSIS WITH TRANSFORMERS
# -----------------------------
def perform_sentiment_analysis(df, batch_size=32):
    """
    Use pre-trained transformer model for sentiment analysis
    More accurate than TextBlob
    """
    print("\n" + "="*60)
    print("STEP 4: SENTIMENT ANALYSIS")
    print("="*60)
    
    # Check if GPU available
    device = 0 if torch.cuda.is_available() else -1
    print(f"\n🖥️  Device: {'GPU (CUDA)' if device == 0 else 'CPU'}")
    
    # Load sentiment model
    print("\n🤖 Loading sentiment analysis model...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
    
    print("✅ Model loaded!")
    
    # Analyze in batches
    print(f"\n📊 Analyzing sentiment for {len(df)} reviews (batch size: {batch_size})...")
    
    texts = df['review_text'].tolist()
    sentiments = []
    confidences = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:i+batch_size]
        # Truncate to 512 characters (model limit)
        batch = [text[:512] for text in batch]
        
        results = sentiment_analyzer(batch)
        
        for result in results:
            sentiments.append(result['label'])
            confidences.append(result['score'])
    
    # Add to dataframe
    df['sentiment_label'] = sentiments
    df['sentiment_confidence'] = confidences
    
    # Create sentiment score (-1 to +1)
    df['sentiment_score'] = df.apply(
        lambda x: x['sentiment_confidence'] if x['sentiment_label'] == 'POSITIVE' 
        else -x['sentiment_confidence'],
        axis=1
    )
    
    print("\n✅ Sentiment analysis complete!")
    
    # Show distribution
    print("\n📊 Sentiment Distribution:")
    sent_counts = df['sentiment_label'].value_counts()
    for label, count in sent_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count} ({percentage:.1f}%)")
    
    print(f"\nAverage sentiment score: {df['sentiment_score'].mean():.3f}")
    print(f"   (Range: -1 = very negative, +1 = very positive)")
    
    return df

# -----------------------------
# 7. TOPIC-SENTIMENT ANALYSIS
# -----------------------------
def create_topic_sentiment_matrix(df):
    """
    Create matrix showing sentiment by topic
    This shows which topics have most negative feedback
    """
    print("\n" + "="*60)
    print("STEP 5: TOPIC-SENTIMENT ANALYSIS")
    print("="*60)
    
    # Group by topic and calculate average sentiment
    topic_sentiment = df.groupby('topic_category').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_label': lambda x: (x == 'NEGATIVE').sum()
    }).round(3)
    
    topic_sentiment.columns = ['avg_sentiment', 'std_sentiment', 'count', 'negative_count']
    topic_sentiment['negative_pct'] = (topic_sentiment['negative_count'] / topic_sentiment['count'] * 100).round(1)
    
    # Sort by most negative
    topic_sentiment = topic_sentiment.sort_values('avg_sentiment')
    
    print("\n🎯 CRITICAL INSIGHTS - Topics ranked by negativity:")
    print("\n" + topic_sentiment.to_string())
    
    # Flag critical issues
    print("\n🚨 CRITICAL ISSUES (Highly negative sentiment):")
    critical = topic_sentiment[topic_sentiment['avg_sentiment'] < -0.3]
    
    if len(critical) > 0:
        for category in critical.index:
            count = int(critical.loc[category, 'count'])
            neg_pct = critical.loc[category, 'negative_pct']
            avg_sent = critical.loc[category, 'avg_sentiment']
            print(f"   ⚠️  {category.upper()}")
            print(f"      └─ {count} reviews | {neg_pct}% negative | {avg_sent} avg sentiment")
    else:
        print("   ✅ No critical issues detected!")
    
    # Highlight positive areas
    print("\n✨ POSITIVE AREAS (Good sentiment):")
    positive = topic_sentiment[topic_sentiment['avg_sentiment'] > 0.2]
    
    if len(positive) > 0:
        for category in positive.index:
            count = int(positive.loc[category, 'count'])
            avg_sent = positive.loc[category, 'avg_sentiment']
            print(f"   👍 {category}: {avg_sent} avg sentiment ({count} reviews)")
    else:
        print("   ⚠️  No strongly positive areas found")
    
    return topic_sentiment

# -----------------------------
# 8. VISUALIZATIONS
# -----------------------------
def create_visualizations(df, topic_model, save_path="visualizations/"):
    """Create insightful visualizations"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*60)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Topic Distribution
    plt.figure(figsize=(12, 6))
    topic_counts = df['topic_category'].value_counts()
    sns.barplot(x=topic_counts.values, y=topic_counts.index, palette='viridis')
    plt.title('Distribution of Feedback Topics', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Topic Category')
    plt.tight_layout()
    plt.savefig(f'{save_path}topic_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: topic_distribution.png")
    plt.close()
    
    # 2. Sentiment by Topic
    plt.figure(figsize=(12, 8))
    topic_sentiment = df.groupby('topic_category')['sentiment_score'].mean().sort_values()
    colors = ['red' if x < 0 else 'green' for x in topic_sentiment.values]
    sns.barplot(x=topic_sentiment.values, y=topic_sentiment.index, palette=colors)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.title('Average Sentiment by Topic', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment Score (-1 to +1)')
    plt.ylabel('Topic Category')
    plt.tight_layout()
    plt.savefig(f'{save_path}sentiment_by_topic.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: sentiment_by_topic.png")
    plt.close()
    
    # 3. Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='sentiment_score', bins=50, kde=True)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral')
    plt.title('Overall Sentiment Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: sentiment_distribution.png")
    plt.close()
    
    # 4. Topic-Sentiment Heatmap
    plt.figure(figsize=(10, 8))
    pivot = df.groupby(['topic_category', 'sentiment_label']).size().unstack(fill_value=0)
    sns.heatmap(pivot, annot=True, fmt='d', cmap='RdYlGn', cbar_kws={'label': 'Count'})
    plt.title('Topic vs Sentiment Heatmap', fontsize=16, fontweight='bold')
    plt.ylabel('Topic Category')
    plt.xlabel('Sentiment')
    plt.tight_layout()
    plt.savefig(f'{save_path}topic_sentiment_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: topic_sentiment_heatmap.png")
    plt.close()
    
    # 5. Word Clouds for each topic
    print("\n📊 Generating word clouds...")
    for category in df['topic_category'].unique():
        if category not in ["uncategorized", "other_issues"]:
            texts = ' '.join(df[df['topic_category'] == category]['review_text'].tolist())
            
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=50
            ).generate(texts)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Word Cloud: {category}', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_path}wordcloud_{category}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print("✅ Saved: All word clouds")
    
    print(f"\n✅ All visualizations saved to: {save_path}")

# -----------------------------
# 9. SAVE MODELS AND RESULTS
# -----------------------------
def save_results(df, topic_model, topic_mapping, model_path="models/"):
    """Save processed data and models"""
    import os
    os.makedirs(model_path, exist_ok=True)
    
    print("\n" + "="*60)
    print("STEP 7: SAVING RESULTS")
    print("="*60)
    
    # Save processed dataframe
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved processed data: {OUTPUT_FILE}")
    
    # Save topic model
    topic_model.save(f"{model_path}bertopic_model")
    print(f"✅ Saved BERTopic model: {model_path}bertopic_model/")
    
    # Save topic mapping
    with open(f"{model_path}topic_mapping.pkl", 'wb') as f:
        pickle.dump(topic_mapping, f)
    print(f"✅ Saved topic mapping: {model_path}topic_mapping.pkl")
    
    print("\n" + "="*60)
    print("ALL RESULTS SAVED!")
    print("="*60)

# -----------------------------
# 10. GENERATE SUMMARY REPORT
# -----------------------------
def generate_summary_report(df, topic_sentiment):
    """Generate executive summary"""
    print("\n" + "="*60)
    print("📊 EXECUTIVE SUMMARY REPORT")
    print("="*60)
    
    total_reviews = len(df)
    negative_reviews = len(df[df['sentiment_label'] == 'NEGATIVE'])
    positive_reviews = len(df[df['sentiment_label'] == 'POSITIVE'])
    
    print(f"\n📈 OVERALL METRICS:")
    print(f"   Total Reviews Analyzed: {total_reviews}")
    print(f"   Positive Reviews: {positive_reviews} ({positive_reviews/total_reviews*100:.1f}%)")
    print(f"   Negative Reviews: {negative_reviews} ({negative_reviews/total_reviews*100:.1f}%)")
    print(f"   Average Sentiment: {df['sentiment_score'].mean():.3f}")
    
    print(f"\n🎯 TOP ISSUES (Most Negative):")
    top_issues = topic_sentiment.head(3)
    for i, (category, row) in enumerate(top_issues.iterrows(), 1):
        print(f"\n   {i}. {category.upper()}")
        print(f"      - {int(row['count'])} reviews")
        print(f"      - {row['negative_pct']:.1f}% negative")
        print(f"      - Avg sentiment: {row['avg_sentiment']:.3f}")
        
        # Get sample negative review
        samples = df[(df['topic_category'] == category) & 
                    (df['sentiment_label'] == 'NEGATIVE')]
        if len(samples) > 0:
            sample = samples.iloc[0]
            print(f"      - Sample: '{sample['review_text'][:100]}...'")
    
    print(f"\n✨ TOP POSITIVE AREAS:")
    positive_topics = topic_sentiment.tail(2)
    for category, row in positive_topics.iterrows():
        if row['avg_sentiment'] > 0:
            print(f"   ✅ {category}: {row['avg_sentiment']:.3f} avg sentiment ({int(row['count'])} reviews)")
    
    print("\n" + "="*60)

# -----------------------------
# 11. MAIN PIPELINE
# -----------------------------
def ml_pipeline():
    """Complete ML pipeline"""
    print("\n" + "🤖"*30)
    print("MACHINE LEARNING PIPELINE - TOPIC MODELING & SENTIMENT ANALYSIS")
    print("🤖"*30 + "\n")
    
    # Step 1: Load data
    df = load_augmented_data(INPUT_FILE)
    
    # Step 2: Topic modeling (IMPROVED)
    df, topic_model, topic_info = perform_topic_modeling(df, num_topics=6)
    
    # Step 3: Map topics to categories (IMPROVED)
    df, topic_mapping = map_topics_to_categories(df, topic_model)
    
    # Step 3.5: Keyword-based reclassification (NEW)
    df = keyword_reclassify(df)
    
    # Step 4: Sentiment analysis
    df = perform_sentiment_analysis(df, batch_size=32)
    
    # Step 5: Topic-sentiment analysis
    topic_sentiment = create_topic_sentiment_matrix(df)
    
    # Step 6: Create visualizations
    create_visualizations(df, topic_model)
    
    # Step 7: Save everything
    save_results(df, topic_model, topic_mapping)
    
    # Step 8: Generate report
    generate_summary_report(df, topic_sentiment)
    
    print("\n" + "="*60)
    print("✅ ML PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nOutput files created:")
    print(f"   1. {OUTPUT_FILE} - Processed data with topics & sentiment")
    print(f"   2. visualizations/ - All charts and word clouds")
    print(f"   3. models/ - Saved BERTopic model")
    print(f"\nNext steps:")
    print(f"   1. Review visualizations in visualizations/ folder")
    print(f"   2. Check {OUTPUT_FILE} for processed data")
    print(f"   3. Proceed to LLM Summarization (next phase)")
    
    return df, topic_model, topic_sentiment

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    df, topic_model, topic_sentiment = ml_pipeline()