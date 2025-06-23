import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import re


def clean_creative_text(text):
    """Clean and standardize creative text"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text


def extract_creative_features(df):
    """Extract features from creative text"""
    df['creative_length'] = df['creative_text'].apply(lambda x: len(str(x)))
    df['creative_word_count'] = df['creative_text'].apply(lambda x: len(str(x).split()))
    
    # Extract common keywords
    keywords = ['cheap', 'flight', 'travel', 'book', 'deal', 'offer', 'dream', 'world', 'family', 'resort']
    for keyword in keywords:
        df[f'has_{keyword}'] = df['creative_text'].apply(lambda x: keyword in str(x).lower()).astype(int)
    
    return df


def calculate_metrics(df):
    """Calculate additional marketing metrics"""
    # Basic ratios
    df['cpm'] = np.where(df['impressions'] > 0, (df['spend'] / df['impressions']) * 1000, 0)
    df['cpc'] = np.where(df['impressions'] > 0, df['spend'] / df['impressions'], 0)
    df['ctr'] = np.where(df['impressions'] > 0, df['sales'] / df['impressions'], 0)
    df['roas'] = np.where(df['spend'] > 0, df['sales'] / df['spend'], 0)
    
    # Efficiency metrics
    df['spend_per_sale'] = np.where(df['sales'] > 0, df['spend'] / df['sales'], 0)
    df['impressions_per_sale'] = np.where(df['sales'] > 0, df['impressions'] / df['sales'], 0)
    
    return df


def add_time_features(df):
    """Add time-based features"""
    # Convert week to datetime (assuming week 1 starts from a reference date)
    reference_date = datetime(2024, 1, 1)  # You can adjust this reference date
    
    df['date'] = df['week'].apply(lambda x: reference_date + timedelta(weeks=x-1))
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Seasonality features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday_season'] = df['month'].isin([6, 7, 8, 12]).astype(int)  # Summer and December
    
    return df


def add_aggregated_features(df):
    """Add aggregated features by geo and channel"""
    # Geo-level aggregations
    geo_agg = df.groupby(['week', 'geo']).agg({
        'spend': 'sum',
        'impressions': 'sum',
        'sales': 'sum'
    }).reset_index()
    
    geo_agg.columns = ['week', 'geo', 'geo_total_spend', 'geo_total_impressions', 'geo_total_sales']
    
    # Channel-level aggregations
    channel_agg = df.groupby(['week', 'channel']).agg({
        'spend': 'sum',
        'impressions': 'sum',
        'sales': 'sum'
    }).reset_index()
    
    channel_agg.columns = ['week', 'channel', 'channel_total_spend', 'channel_total_impressions', 'channel_total_sales']
    
    # Merge back to original dataframe
    df = df.merge(geo_agg, on=['week', 'geo'], how='left')
    df = df.merge(channel_agg, on=['week', 'channel'], how='left')
    
    # Calculate share metrics
    df['spend_share_of_geo'] = np.where(df['geo_total_spend'] > 0, df['spend'] / df['geo_total_spend'], 0)
    df['spend_share_of_channel'] = np.where(df['channel_total_spend'] > 0, df['spend'] / df['channel_total_spend'], 0)
    
    return df


def process_weekly_data(input_path="./data/raw/dummy_weekly_data.csv", 
                       output_path="./data/processed/processed_weekly_data.csv"):
    """Main function to process weekly data"""
    
    # Load raw data
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Clean creative text
    print("Cleaning creative text...")
    df['creative_text'] = df['creative_text'].apply(clean_creative_text)
    
    # Extract creative features
    print("Extracting creative features...")
    df = extract_creative_features(df)
    
    # Calculate marketing metrics
    print("Calculating marketing metrics...")
    df = calculate_metrics(df)
    
    # Add time features
    print("Adding time features...")
    df = add_time_features(df)
    
    # Add aggregated features
    print("Adding aggregated features...")
    df = add_aggregated_features(df)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed data
    print(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)
    
    print(f"Processing complete! Processed {len(df)} rows with {len(df.columns)} columns")
    
    return df


if __name__ == "__main__":
    # Process the data
    processed_df = process_weekly_data()
    
    # Display summary statistics
    print("\nProcessed Data Summary:")
    print(f"Shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")
    print(f"Date range: {processed_df['date'].min()} to {processed_df['date'].max()}")
    print(f"Geographies: {processed_df['geo'].unique()}")
    print(f"Channels: {processed_df['channel'].unique()}") 