import pandas as pd
import numpy as np
import random
from textblob import TextBlob
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
# from googletrans import Translator  # Commented out due to dependency issues
import time
from tqdm import tqdm
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILE = "C:/Users/HP/Desktop/Interview Prep/Projects/InsightIO/cleaned_reviews.csv"
OUTPUT_FILE = "augmented_reviews_with_llm.csv"
TARGET_SIZE = 2000  # Target number of records
AUGMENTATIONS_PER_RECORD = 3  # How many variations per original

# API KEY - Add your OpenAI API key here
OPENAI_API_KEY = "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"

# -----------------------------
# 1. SYNONYM REPLACEMENT
# -----------------------------
def augment_synonym(text, num_variations=2):
    """Replace words with synonyms using WordNet"""
    variations = []
    
    try:
        aug = naw.SynonymAug(aug_src='wordnet')
        for _ in range(num_variations):
            augmented = aug.augment(text)
            if isinstance(augmented, list):
                augmented = augmented[0]
            variations.append(augmented)
    except Exception as e:
        print(f"Synonym augmentation error: {e}")
        variations = [text] * num_variations
    
    return variations

# -----------------------------
# 2. BACK-TRANSLATION (DISABLED)
# -----------------------------
def augment_back_translation(text, intermediate_lang='hi'):
    """Translate to another language and back to English"""
    # Disabled due to googletrans dependency issues
    # Not used in mixed mode anyway
    return text

# -----------------------------
# 3. PARAPHRASING WITH CONTEXT
# -----------------------------
def augment_contextual(text):
    """Use contextual word embeddings to replace words"""
    try:
        aug = naw.ContextualWordEmbsAug(
            model_path='distilbert-base-uncased',
            action="substitute"
        )
        augmented = aug.augment(text)
        if isinstance(augmented, list):
            augmented = augmented[0]
        return augmented
    except Exception as e:
        print(f"Contextual augmentation error: {e}")
        return text

# -----------------------------
# 4. ADD REALISTIC VARIATIONS
# -----------------------------
def add_typos_and_slang(text, typo_rate=0.1):
    """Add common typos and casual language"""
    words = text.split()
    
    # Common typos mapping
    typo_map = {
        'the': 'teh',
        'and': 'nd',
        'app': 'ap',
        'payment': 'pymnt',
        'transaction': 'transactn',
        'problem': 'problm',
        'issue': 'isue',
        'not': 'nt',
        'working': 'wrking'
    }
    
    # Slang/casual replacements
    casual_map = {
        'is not': 'isnt',
        'cannot': 'cant',
        'does not': 'doesnt',
        'do not': 'dont',
        'very bad': 'terrible',
        'very good': 'awesome',
        'not working': 'broken'
    }
    
    # Apply random typos
    result = []
    for word in words:
        if random.random() < typo_rate and word.lower() in typo_map:
            result.append(typo_map[word.lower()])
        else:
            result.append(word)
    
    text = ' '.join(result)
    
    # Apply casual language
    for formal, casual in casual_map.items():
        if formal in text.lower():
            text = text.replace(formal, casual)
    
    return text

# -----------------------------
# 5. TEMPLATE-BASED AUGMENTATION
# -----------------------------
def augment_with_templates(text, rating):
    """Add context-appropriate prefixes/suffixes based on sentiment"""
    
    if rating >= 4:  # Positive
        prefixes = [
            "Love this app! ",
            "Great experience. ",
            "Highly recommend. ",
            "Amazing! "
        ]
        suffixes = [
            " Keep it up!",
            " Worth trying.",
            " Five stars!",
            " Best app ever."
        ]
    elif rating <= 2:  # Negative
        prefixes = [
            "Disappointed. ",
            "Terrible experience. ",
            "Waste of time. ",
            "Not recommended. "
        ]
        suffixes = [
            " Fix this ASAP!",
            " Needs improvement.",
            " Very frustrated.",
            " Won't use again."
        ]
    else:  # Neutral
        prefixes = ["", "Okay app. ", "Average. "]
        suffixes = ["", " Could be better.", " Decent."]
    
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    
    return f"{prefix}{text}{suffix}".strip()

# -----------------------------
# 6. LLM-BASED AUGMENTATION (OPENAI)
# -----------------------------
def augment_with_llm(text, api_key=None, num_variations=1):
    """
    Use OpenAI GPT to generate high-quality paraphrases
    """
    
    if api_key is None or api_key == "":
        print("⚠️  No API key provided for LLM augmentation - skipping")
        return [text] * num_variations
    
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f'''Generate {num_variations} natural variations of this customer feedback.
Keep the same core issue and sentiment, but vary the wording naturally.
Sound like real customer feedback (casual, may have minor typos).

Original: "{text}"

Return only the variations, one per line. No numbering or extra text.'''
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" for better quality
            messages=[
                {"role": "system", "content": "You are a helpful assistant that paraphrases customer feedback naturally. Return only the paraphrased text, nothing else."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.8,  # Higher temperature = more variation
            n=1
        )
        
        # Extract variations from response
        variations_text = response.choices[0].message.content.strip()
        
        # Split by newlines and clean up
        variations = []
        for line in variations_text.split('\n'):
            cleaned = line.strip()
            # Remove numbering if present (1., 2., etc.)
            cleaned = cleaned.lstrip('0123456789.- ')
            if cleaned and len(cleaned) > 10:  # Valid variation
                variations.append(cleaned)
        
        # Return requested number of variations
        if variations:
            return variations[:num_variations]
        else:
            return [text] * num_variations
        
    except Exception as e:
        print(f"LLM augmentation error: {e}")
        return [text] * num_variations

# -----------------------------
# 7. SMART AUGMENTATION STRATEGY (WITH LLM)
# -----------------------------
def augment_record(row, method='mixed', num_variations=3, api_key=None):
    """
    Apply multiple augmentation techniques to a single record
    
    Methods:
    - 'synonym': Synonym replacement only
    - 'contextual': Contextual embeddings
    - 'llm': LLM-based paraphrasing
    - 'mixed': Combination of all techniques (recommended)
    """
    
    text = row['review_text']
    rating = row['rating'] if pd.notna(row['rating']) else 3
    
    augmented_records = []
    
    if method == 'mixed':
        # Strategy: Use different techniques for diversity INCLUDING LLM
        techniques = [
            ('synonym', augment_synonym),
            ('contextual', lambda t: [augment_contextual(t)]),
            ('template', lambda t: [augment_with_templates(t, rating)]),
            ('llm', lambda t: augment_with_llm(t, api_key=api_key, num_variations=1))  # ✨ LLM ADDED
        ]
        
        for i in range(num_variations):
            technique_name, technique_func = techniques[i % len(techniques)]
            
            try:
                if technique_name == 'synonym':
                    variants = technique_func(text, 1)
                    augmented_text = variants[0] if variants else text
                elif technique_name == 'llm':
                    variants = technique_func(text)
                    augmented_text = variants[0] if variants else text
                else:
                    variants = technique_func(text)
                    augmented_text = variants[0] if variants else text
                
                # Add some randomness (but not for LLM - it's already creative)
                if technique_name != 'llm' and random.random() < 0.3:
                    augmented_text = add_typos_and_slang(augmented_text)
                
                # Create augmented record
                new_row = row.copy()
                new_row['review_text'] = augmented_text
                new_row['is_augmented'] = True
                new_row['augmentation_method'] = technique_name
                new_row['original_id'] = row.name
                
                augmented_records.append(new_row)
                
            except Exception as e:
                print(f"Error in {technique_name}: {e}")
                continue
    
    elif method == 'synonym':
        variants = augment_synonym(text, num_variations)
        for variant in variants:
            new_row = row.copy()
            new_row['review_text'] = variant
            new_row['is_augmented'] = True
            new_row['augmentation_method'] = 'synonym'
            new_row['original_id'] = row.name
            augmented_records.append(new_row)
    
    elif method == 'llm':
        variants = augment_with_llm(text, api_key=api_key, num_variations=num_variations)
        for variant in variants:
            new_row = row.copy()
            new_row['review_text'] = variant
            new_row['is_augmented'] = True
            new_row['augmentation_method'] = 'llm'
            new_row['original_id'] = row.name
            augmented_records.append(new_row)
    
    return augmented_records

# -----------------------------
# 8. SELECTIVE AUGMENTATION (WITH LLM)
# -----------------------------
def selective_augmentation(df, target_size=2000, api_key=None):
    """
    Intelligently augment to reach target size
    Focus more on:
    - Minority classes (rare issues)
    - Extreme sentiments (very positive/negative)
    - Short reviews (need more examples)
    """
    
    # Add is_augmented flag to originals
    df['is_augmented'] = False
    df['augmentation_method'] = 'original'
    df['original_id'] = df.index
    
    current_size = len(df)
    needed = target_size - current_size
    
    if needed <= 0:
        print("Already at target size!")
        return df
    
    # Calculate how many augmentations per record
    augmentations_per_record = int(np.ceil(needed / current_size))
    print(f"Generating ~{augmentations_per_record} variations per record")
    
    # Priority scoring for augmentation
    df['augmentation_priority'] = 0
    
    # Priority 1: Negative sentiment (need more negative examples)
    df.loc[df['sentiment'] < -0.5, 'augmentation_priority'] += 3
    
    # Priority 2: Extreme ratings
    df.loc[df['rating'].isin([1, 5]), 'augmentation_priority'] += 2
    
    # Priority 3: Short reviews (might be underrepresented)
    df.loc[df['review_text'].str.len() < 50, 'augmentation_priority'] += 1
    
    # Sort by priority
    df_sorted = df.sort_values('augmentation_priority', ascending=False)
    
    augmented_records = []
    augmented_records.append(df)  # Keep originals
    
    print("Starting augmentation...")
    print(f"🤖 LLM augmentation: {'ENABLED ✅' if api_key else 'DISABLED ⚠️'}")
    
    for idx, row in tqdm(df_sorted.iterrows(), total=len(df_sorted)):
        if len(pd.concat(augmented_records)) >= target_size:
            break
        
        # Higher priority records get more augmentations
        num_augs = min(
            augmentations_per_record + int(row['augmentation_priority']),
            5  # Max 5 variations per record
        )
        
        try:
            # Pass API key to enable LLM augmentation
            new_records = augment_record(row, method='mixed', num_variations=num_augs, api_key=api_key)
            
            for record in new_records:
                augmented_records.append(pd.DataFrame([record]))
                
                if len(pd.concat(augmented_records)) >= target_size:
                    break
        
        except Exception as e:
            print(f"Error augmenting row {idx}: {e}")
            continue
    
    # Combine all records
    final_df = pd.concat(augmented_records, ignore_index=True)
    
    # Remove temporary priority column
    final_df = final_df.drop('augmentation_priority', axis=1, errors='ignore')
    
    return final_df

# -----------------------------
# 9. MAIN AUGMENTATION PIPELINE (WITH LLM)
# -----------------------------
def augmentation_pipeline(openai_api_key=None):
    print("="*50)
    print("STARTING AUGMENTATION PIPELINE")
    print("="*50)
    
    if openai_api_key:
        print("🤖 LLM Augmentation: ENABLED ✅")
        print("   Using: OpenAI GPT-3.5-Turbo")
    else:
        print("⚠️  LLM Augmentation: DISABLED")
        print("   (Will use only synonym, contextual, template methods)")
    
    # Load cleaned data
    print(f"\n1. Loading cleaned data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Original records: {len(df)}")
    
    # Show distribution
    print("\n2. Original data distribution:")
    print(f"   Sentiment distribution:")
    print(df['sentiment'].describe())
    if 'rating' in df.columns:
        print(f"\n   Rating distribution:")
        print(df['rating'].value_counts().sort_index())
    
    # Perform augmentation
    print(f"\n3. Augmenting to reach ~{TARGET_SIZE} records...")
    df_augmented = selective_augmentation(df, target_size=TARGET_SIZE, api_key=openai_api_key)
    
    print(f"\n4. Augmentation complete!")
    print(f"   Original records: {len(df)}")
    print(f"   Augmented records: {len(df_augmented)}")
    print(f"   New records created: {len(df_augmented) - len(df)}")
    
    # Statistics
    print("\n5. Augmentation statistics:")
    print(f"   Original records: {df_augmented['is_augmented'].value_counts().get(False, 0)}")
    print(f"   Augmented records: {df_augmented['is_augmented'].value_counts().get(True, 0)}")
    
    if 'augmentation_method' in df_augmented.columns:
        print("\n   Methods used:")
        method_counts = df_augmented['augmentation_method'].value_counts()
        print(method_counts)
        
        # Highlight LLM usage
        if 'llm' in method_counts:
            print(f"\n   🤖 LLM-generated variations: {method_counts['llm']} ({method_counts['llm']/len(df_augmented)*100:.1f}%)")
    
    # Save
    print(f"\n6. Saving to {OUTPUT_FILE}...")
    df_augmented.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*50)
    print("AUGMENTATION COMPLETE!")
    print("="*50)
    print(f"\nFinal dataset: {OUTPUT_FILE}")
    print(f"Total records: {len(df_augmented)}")
    
    return df_augmented

# -----------------------------
# 10. VALIDATION & QUALITY CHECK
# -----------------------------
def validate_augmentation(df):
    """Check quality of augmented data"""
    
    print("\n" + "="*50)
    print("QUALITY VALIDATION")
    print("="*50)
    
    augmented = df[df['is_augmented'] == True]
    
    if len(augmented) == 0:
        print("No augmented records found!")
        return
    
    # Check 1: Length distribution
    print("\n1. Text length comparison:")
    print(f"   Original avg length: {df[df['is_augmented']==False]['review_text'].str.len().mean():.1f}")
    print(f"   Augmented avg length: {augmented['review_text'].str.len().mean():.1f}")
    
    # Check 2: Sentiment preservation
    if 'sentiment' in df.columns:
        print("\n2. Sentiment preservation:")
        # Group by original_id and check if sentiment is similar
        sentiment_check = df.groupby('original_id')['sentiment'].std().mean()
        print(f"   Avg sentiment variation within groups: {sentiment_check:.3f}")
        print(f"   (Lower is better - should be < 0.3)")
    
    # Check 3: Diversity
    print("\n3. Diversity check:")
    print(f"   Unique texts: {df['review_text'].nunique()}")
    print(f"   Duplicate rate: {(1 - df['review_text'].nunique()/len(df))*100:.1f}%")
    
    # Check 4: Sample augmented records BY METHOD
    print("\n4. Sample augmented records by method:")
    
    for method in ['synonym', 'contextual', 'template', 'llm']:
        method_samples = df[df['augmentation_method'] == method]
        if len(method_samples) > 0:
            sample = method_samples.iloc[0]
            original = df[df['original_id'] == sample['original_id']][df['is_augmented'] == False].iloc[0]
            
            print(f"\n   📌 Method: {method.upper()}")
            print(f"   Original: '{original['review_text']}'")
            print(f"   Augmented: '{sample['review_text']}'")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    # Run augmentation WITH LLM
    df_final = augmentation_pipeline(openai_api_key=OPENAI_API_KEY)
    
    # Validate quality
    validate_augmentation(df_final)
    
    print("\n✅ All done! Ready for ML pipeline.")
    
    # Show cost estimate
    if OPENAI_API_KEY:
        llm_count = len(df_final[df_final['augmentation_method'] == 'llm'])
        estimated_cost = llm_count * 0.002  # Rough estimate
        print(f"\n💰 Estimated OpenAI cost: ${estimated_cost:.2f}")
        print(f"   (Based on {llm_count} LLM-generated variations)")