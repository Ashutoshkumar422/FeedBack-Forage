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
OUTPUT_FILE = "reviews_with_topics_and_sentiment.csv"
MODEL_SAVE_PATH = "models/"

# Topic categories we expect
EXPECTED_TOPICS = {
    0: "payment_issues",
    1: "login_problems", 
    2: "app_crashes_bugs",
    3: "ui_ux_issues",
    4: "fraud_security",
    5: "cashback_rewards"
}

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
# 3. BERTOPIC - ADVANCED TOPIC MODELING
# -----------------------------
def perform_topic_modeling(df, num_topics=6):
    """
    Use BERTopic to discover topics in feedback
    BERTopic is better than LDA for small datasets
    """
    print("\n" + "="*60)
    print("STEP 2: TOPIC MODELING (BERTopic)")
    print("="*60)
    
    # Prepare texts
    print("\n📝 Preprocessing texts...")
    df['text_processed'] = df['review_text'].apply(preprocess_for_topics)
    texts = df['text_processed'].tolist()
    
    print(f"✅ Processed {len(texts)} texts")
    
    # Initialize embedding model
    print("\n🤖 Loading sentence transformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Custom vectorizer to remove common stop words
    vectorizer_model = CountVectorizer(
        stop_words='english',
        min_df=2,  # Word must appear in at least 2 documents
        ngram_range=(1, 2)  # Consider both single words and bigrams
    )
    
    # Initialize BERTopic
    print("\n🎯 Training BERTopic model...")
    print(f"   Target topics: {num_topics}")

    
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        nr_topics=num_topics,  # Force specific number of topics
        min_topic_size=15,  # Minimum documents per topic
        verbose=True,
        calculate_probabilities=True
    )
    
    # Fit model
    topics, probabilities = topic_model.fit_transform(texts)
    
    # Add results to dataframe
    df['topic'] = topics
    df['topic_probability'] = [prob.max() for prob in probabilities]
    
    print("\n✅ Topic modeling complete!")
    
    # Show topic distribution
    print("\n📊 Topic Distribution:")
    topic_counts = df['topic'].value_counts().sort_index()
    for topic_id, count in topic_counts.items():
        print(f"   Topic {topic_id}: {count} documents ({count/len(df)*100:.1f}%)")
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    print("\n🏷️  Topic Keywords:")
    for topic_id in sorted(df['topic'].unique()):
        if topic_id != -1:  # -1 is outlier topic
            keywords = topic_model.get_topic(topic_id)
            top_words = [word for word, _ in keywords[:5]]
            print(f"   Topic {topic_id}: {', '.join(top_words)}")
    
    return df, topic_model, topic_info

# -----------------------------
# 4. MAP TOPICS TO BUSINESS CATEGORIES
# -----------------------------
def map_topics_to_categories(df, topic_model):
    """
    Manually map discovered topics to business categories
    Based on top keywords
    """
    print("\n" + "="*60)
    print("STEP 3: MAPPING TOPICS TO CATEGORIES")
    print("="*60)
    
    # Analyze keywords and create mapping
    topic_mapping = {}
    
    print("\n🔍 Analyzing topics...")
    for topic_id in sorted(df['topic'].unique()):
        if topic_id == -1:
            topic_mapping[-1] = "uncategorized"
            continue
        
        # Get top words for this topic
        keywords = topic_model.get_topic(topic_id)
        top_words = [word for word, _ in keywords[:10]]
        
        print(f"\nTopic {topic_id} keywords: {', '.join(top_words[:5])}")
        
        # Rule-based mapping based on keywords
        keywords_str = ' '.join(top_words).lower()
        
        if any(word in keywords_str for word in ['payment', 'money', 'transaction', 'refund', 'pay']):
            category = "payment_issues"
        elif any(word in keywords_str for word in ['login', 'password', 'otp', 'signin', 'access']):
            category = "login_problems"
        elif any(word in keywords_str for word in ['crash', 'freeze', 'bug', 'error', 'loading']):
            category = "app_crashes_bugs"
        elif any(word in keywords_str for word in ['interface', 'ui', 'design', 'confusing', 'button']):
            category = "ui_ux_issues"
        elif any(word in keywords_str for word in ['fraud', 'scam', 'security', 'hack', 'unsafe']):
            category = "fraud_security"
        elif any(word in keywords_str for word in ['cash back', 'reward', 'points', 'offer', 'discount']):
            category = "cashback_rewards"
        else:
            category = "other_issues"
        
        topic_mapping[topic_id] = category
        print(f"   → Mapped to: {category}")
    
    # Apply mapping
    df['topic_category'] = df['topic'].map(topic_mapping)
    
    print("\n✅ Topic mapping complete!")
    print("\n📊 Category Distribution:")
    print(df['topic_category'].value_counts())
    
    return df, topic_mapping

# -----------------------------
# 5. SENTIMENT ANALYSIS WITH TRANSFORMERS
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
    print(f"\n🖥️  Device: {'GPU' if device == 0 else 'CPU'}")
    
    # Load sentiment model
    print("\n🤖 Loading sentiment analysis model...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
    
    print("✅ Model loaded!")
    
    # Analyze in batches
    print(f"\n📊 Analyzing sentiment for {len(df)} reviews...")
    
    texts = df['review_text'].tolist()
    sentiments = []
    confidences = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
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
    print(df['sentiment_label'].value_counts())
    print(f"\nAverage sentiment score: {df['sentiment_score'].mean():.3f}")
    print(f"   (Range: -1 = very negative, +1 = very positive)")
    
    return df

# -----------------------------
# 6. COMBINE TOPIC + SENTIMENT FOR INSIGHTS
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
    print("\n🚨 CRITICAL ISSUES (High negative sentiment):")
    critical = topic_sentiment[topic_sentiment['avg_sentiment'] < -0.3]
    for category in critical.index:
        count = critical.loc[category, 'count']
        neg_pct = critical.loc[category, 'negative_pct']
        avg_sent = critical.loc[category, 'avg_sentiment']
        print(f"   ⚠️  {category}: {count} reviews, {neg_pct}% negative, avg sentiment: {avg_sent}")
    
    return topic_sentiment

# -----------------------------
# 7. VISUALIZATIONS
# -----------------------------
def create_visualizations(df, topic_model, save_path="visualizations/"):
    """Create insightful visualizations"""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("\n" + "="*60)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("="*60)
    
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
    
    # 5. Word Cloud for each topic
    for category in df['topic_category'].unique():
        if category != "uncategorized":
            texts = ' '.join(df[df['topic_category'] == category]['review_text'].tolist())
            
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis'
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
# 8. SAVE MODELS AND RESULTS
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
# 9. GENERATE SUMMARY REPORT
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
        sample = df[(df['topic_category'] == category) & 
                   (df['sentiment_label'] == 'NEGATIVE')].iloc[0]
        print(f"      - Sample: '{sample['review_text'][:100]}...'")
    
    print(f"\n✨ TOP POSITIVE AREAS:")
    positive_topics = topic_sentiment.tail(2)
    for category, row in positive_topics.iterrows():
        if row['avg_sentiment'] > 0:
            print(f"   ✅ {category}: {row['avg_sentiment']:.3f} avg sentiment")
    
    print("\n" + "="*60)

# -----------------------------
# 10. MAIN PIPELINE
# -----------------------------
def ml_pipeline():
    """Complete ML pipeline"""
    print("\n" + "🤖"*30)
    print("MACHINE LEARNING PIPELINE - TOPIC MODELING & SENTIMENT ANALYSIS")
    print("🤖"*30 + "\n")
    
    # Step 1: Load data
    df = load_augmented_data(INPUT_FILE)
    
    # Step 2: Topic modeling
    df, topic_model, topic_info = perform_topic_modeling(df, num_topics=6)
    
    # Step 3: Map topics to categories
    df, topic_mapping = map_topics_to_categories(df, topic_model)
    
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
    print(f"\nNext steps:")
    print("   1. Review visualizations in visualizations/ folder")
    print("   2. Check {OUTPUT_FILE} for processed data")
    print("   3. Proceed to LLM Summarization (next phase)")
    
    return df, topic_model, topic_sentiment

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    df, topic_model, topic_sentiment = ml_pipeline()