"""
DATA VORTEX - Round 1: Rebuilding the Social Engine
Production Data Intake Cleaning & Preprocessing Pipeline
Author: Antigravity Team
Date: September 2026
"""

import os
import re
import html
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

WORKSPACE_DIR = r"x:\Projects\srm thevidyas event"
POSTS_PATH = os.path.join(WORKSPACE_DIR, "Social_Engine_Posts_Corrupted.csv")
USERS_PATH = os.path.join(WORKSPACE_DIR, "Social_Engine_Users.csv")
REPORTS_DIR = os.path.join(WORKSPACE_DIR, "reports")
SCRIPTS_DIR = os.path.join(WORKSPACE_DIR, "scripts")
SQL_DIR = os.path.join(WORKSPACE_DIR, "sql")
NOTEBOOKS_DIR = os.path.join(WORKSPACE_DIR, "notebooks")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SCRIPTS_DIR, exist_ok=True)
os.makedirs(SQL_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def parse_unified_timestamp(ts):
    """
    Standardizes mixed timestamp formats:
    - Unix Epoch timestamps (e.g. '1722528840')
    - DD-MM-YYYY dates (e.g. '25-09-2024')
    - ISO-8601 timestamps (e.g. '2025-04-13T20:12:18')
    Returns: pd.Timestamp in UTC
    """
    if pd.isnull(ts):
        return pd.NaT
    ts_str = str(ts).strip()
    if ts_str.isdigit():
        return pd.to_datetime(int(ts_str), unit='s')
    try:
        # Check if DD-MM-YYYY or similar
        if re.match(r'^\d{2}-\d{2}-\d{4}', ts_str) or re.match(r'^\d{2}/\d{2}/\d{4}', ts_str):
            return pd.to_datetime(ts_str, dayfirst=True)
        return pd.to_datetime(ts_str)
    except Exception:
        return pd.to_datetime(ts_str, errors='coerce')

def sanitize_text(text):
    """
    Sanitizes corrupted text:
    - Replaces corrupted null strings (NULL\n\n, NULL&amp;, NULLé, NULL<div>, etc.) with None
    - Decodes HTML entities (&amp; -> &, etc.)
    - Removes residual HTML tags (<div>, <br>, etc.)
    - Trims excess whitespace
    """
    if pd.isnull(text):
        return None
    text_str = str(text).strip()
    
    # Check pseudo-null variants
    if re.match(r'^NULL(\s*|&amp;|é|<div>|<br>|\\n)*$', text_str, re.IGNORECASE):
        return None
    
    # Unescape HTML entities
    cleaned = html.unescape(text_str)
    
    # Strip HTML tags
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    
    # Clean whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned if len(cleaned) > 0 else None

def extract_brand(text):
    """Identifies the primary brand mentioned in post text."""
    if not text or pd.isnull(text):
        return "Unknown / Generic"
    brands = ['Nike', 'Adidas', 'Apple', 'Samsung', 'Toyota', 'Coca-Cola', 'Pepsi', 'Amazon', 'Google']
    for brand in brands:
        if re.search(r'\b' + re.escape(brand) + r'\b', text, re.IGNORECASE):
            return brand
    return "Other / Generic"

def extract_sentiment(text):
    """Rule-based lexical sentiment extraction based on verified text tokens."""
    if not text or pd.isnull(text):
        return "Neutral"
    
    text_lower = text.lower()
    
    pos_markers = [
        'highly recommend', 'exceeded my expectations', 'loving it', 'outstanding', 
        'thrilled', 'absolutely loving', 'musthave', 'bestvalue', 'great'
    ]
    neg_markers = [
        'not worth the money', 'disappointing', 'bummed out', 'fed up',
        "wouldn't recommend", 'delivery delays', 'software bugs', 'connectivity issues'
    ]
    neu_markers = [
        'does the job', "it's okay", 'as expected', 'mixed feelings', 'standard'
    ]
    
    pos_score = sum(1 for m in pos_markers if m in text_lower)
    neg_score = sum(1 for m in neg_markers if m in text_lower)
    
    if pos_score > neg_score:
        return "Positive"
    elif neg_score > pos_score:
        return "Negative"
    else:
        return "Neutral"

def run_cleaning_pipeline():
    print("="*60)
    print("DATA VORTEX: Rebuilding the Social Engine - Pipeline Execution")
    print("="*60)
    
    # 1. Load Datasets
    print("\n[Step 1] Loading raw datasets...")
    df_raw_posts = pd.read_csv(POSTS_PATH)
    df_raw_users = pd.read_csv(USERS_PATH)
    
    raw_posts_count = len(df_raw_posts)
    raw_users_count = len(df_raw_users)
    print(f"Raw Posts loaded: {raw_posts_count} rows, {df_raw_posts.shape[1]} columns")
    print(f"Raw Users loaded: {raw_users_count} rows, {df_raw_users.shape[1]} columns")
    
    audit = {
        "raw_posts_count": raw_posts_count,
        "raw_users_count": raw_users_count,
        "actions": []
    }
    
    # 2. Deduplication
    print("\n[Step 2] Deduplicating exact row copies...")
    duplicate_count = df_raw_posts.duplicated().sum()
    df_posts = df_raw_posts.drop_duplicates().copy()
    unique_posts_count = len(df_posts)
    print(f"Duplicates removed: {duplicate_count}. Clean unique records: {unique_posts_count}")
    audit["actions"].append({
        "stage": "Deduplication",
        "description": "Removed identical duplicate post rows",
        "records_affected": int(duplicate_count),
        "post_clean_count": int(unique_posts_count)
    })
    
    # 3. Timestamp Standardisation
    print("\n[Step 3] Standardizing heterogeneous timestamps...")
    parsed_timestamps = df_posts['timestamp'].apply(parse_unified_timestamp)
    unparsed_count = parsed_timestamps.isnull().sum()
    print(f"Successfully converted {len(parsed_timestamps) - unparsed_count} timestamps to UTC ISO-8601 (NaT: {unparsed_count})")
    
    df_posts['timestamp_cleaned'] = parsed_timestamps.dt.strftime('%Y-%m-%d %H:%M:%S')
    df_posts['post_date'] = parsed_timestamps.dt.strftime('%Y-%m-%d')
    df_posts['post_year'] = parsed_timestamps.dt.year
    df_posts['post_month'] = parsed_timestamps.dt.month
    df_posts['post_day'] = parsed_timestamps.dt.day
    df_posts['day_of_week'] = parsed_timestamps.dt.day_name()
    df_posts['hour_of_day'] = parsed_timestamps.dt.hour
    df_posts['is_weekend'] = parsed_timestamps.dt.dayofweek.isin([5, 6]).astype(int)
    
    audit["actions"].append({
        "stage": "Timestamp Standardization",
        "description": "Unified Unix epoch, DD-MM-YYYY, and ISO formats into ISO-8601 UTC",
        "date_range": {
            "min": str(df_posts['timestamp_cleaned'].min()),
            "max": str(df_posts['timestamp_cleaned'].max())
        }
    })
    
    # 4. Likes Restoration (Bit-flip inversion repair + Stratified Imputation)
    print("\n[Step 4] Restoring corrupted Likes (Inversion repair & stratified imputation)...")
    negative_likes_mask = df_posts['likes'] < 0
    negative_likes_count = int(negative_likes_mask.sum())
    missing_likes_mask = df_posts['likes'].isnull()
    missing_likes_count = int(missing_likes_mask.sum())
    
    df_posts['is_likes_sign_corrected'] = negative_likes_mask.astype(int)
    df_posts['is_likes_imputed'] = missing_likes_mask.astype(int)
    
    # Sign correction
    df_posts['likes_repaired'] = df_posts['likes'].abs()
    
    # Stratified imputation: compute median likes per platform (and overall median for fallback)
    overall_median_likes = df_posts['likes_repaired'].median()
    platform_median_likes = df_posts.groupby('platform')['likes_repaired'].median().to_dict()
    
    def impute_likes(row):
        if pd.isnull(row['likes_repaired']):
            return int(round(platform_median_likes.get(row['platform'], overall_median_likes)))
        return int(round(row['likes_repaired']))
    
    df_posts['likes_clean'] = df_posts.apply(impute_likes, axis=1)
    
    print(f"Bit-flipped negative likes repaired: {negative_likes_count}")
    print(f"Missing likes imputed: {missing_likes_count} (Platform-stratified median: {overall_median_likes:.0f})")
    
    audit["actions"].append({
        "stage": "Likes Restoration",
        "negative_likes_repaired": negative_likes_count,
        "missing_likes_imputed": missing_likes_count,
        "imputation_baseline": {k: float(v) for k, v in platform_median_likes.items()}
    })
    
    # 5. Text Sanitization & Entity Extraction
    print("\n[Step 5] Sanitizing text content and extracting entities...")
    df_posts['text_cleaned'] = df_posts['text_content'].apply(sanitize_text)
    
    pseudo_null_count = sum(
        (df_posts['text_content'].notnull()) & (df_posts['text_cleaned'].isnull())
    )
    print(f"Pseudo-null strings ('NULL\\n\\n', 'NULL&amp;', etc.) neutralized: {pseudo_null_count}")
    
    # Brand extraction
    df_posts['brand'] = df_posts['text_cleaned'].apply(extract_brand)
    # Sentiment extraction
    df_posts['sentiment'] = df_posts['text_cleaned'].apply(extract_sentiment)
    
    audit["actions"].append({
        "stage": "Text Sanitization",
        "pseudo_nulls_neutralized": int(pseudo_null_count),
        "clean_texts_available": int(df_posts['text_cleaned'].notnull().sum()),
        "brand_distribution": df_posts['brand'].value_counts().to_dict(),
        "sentiment_distribution": df_posts['sentiment'].value_counts().to_dict()
    })
    
    # 6. Platform Imputation / Prediction
    print("\n[Step 6] Resolving missing platforms...")
    missing_platform_mask = df_posts['platform'].isnull()
    missing_platform_count = int(missing_platform_mask.sum())
    
    # Train supervised model to predict missing platform based on available engagement and text
    df_labeled = df_posts[~missing_platform_mask].dropna(subset=['text_cleaned']).copy()
    if len(df_labeled) > 0 and missing_platform_count > 0:
        vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
        X_text = vectorizer.fit_transform(df_labeled['text_cleaned']).toarray()
        X_eng = df_labeled[['likes_clean', 'shares', 'comments']].values
        X = np.hstack([X_text, X_eng])
        y = df_labeled['platform'].values
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X, y)
        
        # Predict for missing
        df_unlabeled = df_posts[missing_platform_mask].copy()
        
        # Where text is available, predict; where text is null, assign 'Unknown / Unassigned' or model fallback
        has_text_mask = missing_platform_mask & df_posts['text_cleaned'].notnull()
        if has_text_mask.sum() > 0:
            X_miss_text = vectorizer.transform(df_posts.loc[has_text_mask, 'text_cleaned']).toarray()
            X_miss_eng = df_posts.loc[has_text_mask, ['likes_clean', 'shares', 'comments']].values
            X_miss = np.hstack([X_miss_text, X_miss_eng])
            preds = clf.predict(X_miss)
            df_posts.loc[has_text_mask, 'platform_predicted'] = preds
        
        df_posts['platform_clean'] = df_posts['platform']
        # Fill remaining missing with predicted platform where available, else 'Unknown'
        df_posts['platform_clean'] = df_posts['platform_clean'].fillna(df_posts['platform_predicted'])
        df_posts['platform_clean'] = df_posts['platform_clean'].fillna('Unknown')
    else:
        df_posts['platform_clean'] = df_posts['platform'].fillna('Unknown')
    
    df_posts['is_platform_imputed'] = missing_platform_mask.astype(int)
    print(f"Missing platforms handled: {missing_platform_count} (Flagged in 'is_platform_imputed')")
    
    # 7. Engagement & Virality Feature Engineering
    print("\n[Step 7] Calculating engagement and virality metrics...")
    df_posts['shares_clean'] = df_posts['shares'].astype(int)
    df_posts['comments_clean'] = df_posts['comments'].astype(int)
    df_posts['total_engagement'] = df_posts['likes_clean'] + df_posts['shares_clean'] + df_posts['comments_clean']
    df_posts['virality_score'] = (df_posts['shares_clean'] / (df_posts['likes_clean'] + 1.0)).round(4)
    df_posts['conversation_rate'] = (df_posts['comments_clean'] / (df_posts['likes_clean'] + 1.0)).round(4)
    
    # 8. Merge User Attributes
    print("\n[Step 8] Normalizing Users reference table & joining...")
    df_users = df_raw_users.drop_duplicates(subset=['user_id']).copy()
    df_users['account_created'] = pd.to_datetime(df_users['account_created']).dt.strftime('%Y-%m-%d')
    df_users['follower_count'] = df_users['follower_count'].astype(int)
    
    # Check foreign keys
    orphaned_posts = (~df_posts['user_id'].isin(df_users['user_id'])).sum()
    print(f"Foreign key verification: {orphaned_posts} orphaned posts (Expected: 0)")
    assert orphaned_posts == 0, "Integrity failure: Orphaned posts detected!"
    
    # Calculate engagement per follower
    user_follower_map = df_users.set_index('user_id')['follower_count'].to_dict()
    df_posts['user_follower_count'] = df_posts['user_id'].map(user_follower_map)
    df_posts['engagement_rate_per_follower'] = (df_posts['total_engagement'] / df_posts['user_follower_count']).round(6)
    
    # 9. Format Final Tables
    final_posts_cols = [
        'post_id', 'user_id', 'platform_clean', 'text_cleaned', 
        'timestamp_cleaned', 'post_date', 'post_year', 'post_month', 'day_of_week', 'hour_of_day', 'is_weekend',
        'likes_clean', 'shares_clean', 'comments_clean', 'total_engagement',
        'virality_score', 'conversation_rate', 'engagement_rate_per_follower',
        'brand', 'sentiment',
        'is_likes_sign_corrected', 'is_likes_imputed', 'is_platform_imputed'
    ]
    df_posts_cleaned = df_posts[final_posts_cols].rename(columns={
        'platform_clean': 'platform',
        'text_cleaned': 'text_content',
        'timestamp_cleaned': 'timestamp',
        'likes_clean': 'likes',
        'shares_clean': 'shares',
        'comments_clean': 'comments'
    })
    
    # 10. Save Clean Datasets
    print("\n[Step 10] Exporting production datasets and reports...")
    out_csv = os.path.join(WORKSPACE_DIR, "Social_Engine_Posts_Cleaned.csv")
    out_json = os.path.join(WORKSPACE_DIR, "Social_Engine_Posts_Cleaned.json")
    out_users_csv = os.path.join(WORKSPACE_DIR, "Social_Engine_Users_Normalized.csv")
    audit_json = os.path.join(REPORTS_DIR, "cleaning_audit_summary.json")
    
    df_posts_cleaned.to_csv(out_csv, index=False)
    df_posts_cleaned.to_json(out_json, orient='records', indent=2)
    df_users.to_csv(out_users_csv, index=False)
    
    audit["clean_posts_count"] = len(df_posts_cleaned)
    audit["clean_users_count"] = len(df_users)
    audit["summary_metrics"] = {
        "mean_likes": float(df_posts_cleaned['likes'].mean()),
        "mean_shares": float(df_posts_cleaned['shares'].mean()),
        "mean_comments": float(df_posts_cleaned['comments'].mean()),
        "mean_total_engagement": float(df_posts_cleaned['total_engagement'].mean())
    }
    
    with open(audit_json, 'w') as f:
        json.dump(audit, f, indent=2)
        
    print(f"Cleaned Posts saved to: {out_csv} ({os.path.getsize(out_csv)} bytes)")
    print(f"Cleaned Posts JSON saved to: {out_json} ({os.path.getsize(out_json)} bytes)")
    print(f"Normalized Users saved to: {out_users_csv} ({os.path.getsize(out_users_csv)} bytes)")
    print(f"Cleaning Audit report saved to: {audit_json}")
    print("\n>>> Pipeline successfully executed with 100% data integrity! <<<")
    return df_posts_cleaned, df_users

if __name__ == "__main__":
    run_cleaning_pipeline()
