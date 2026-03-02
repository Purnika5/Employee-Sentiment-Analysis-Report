#!/usr/bin/env python3
"""
Employee Sentiment Analysis - Main Analysis Script
Consolidated implementation combining core analysis, advanced ML, and visualizations
"""

# =============================
# IMPORTS & SETUP
# =============================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import os
import json
import time
from datetime import datetime
from collections import Counter

# NLTK & Text Processing
import nltk
from nltk.corpus import words
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

# ML & Stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Advanced ML (optional)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from bertopic import BERTopic
    from sklearn.ensemble import IsolationForest
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    print("Warning: Advanced ML components not available. Install transformers, bertopic, and scikit-learn for full functionality.")
    ADVANCED_COMPONENTS_AVAILABLE = False

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('words', quiet=True)

# Initialize components
sia = SentimentIntensityAnalyzer()
english_vocab = set(words.words())

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create visualizations folder
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

print("Libraries imported successfully!")
print(f"Advanced components available: {ADVANCED_COMPONENTS_AVAILABLE}")

# =============================
# UTILITY FUNCTIONS
# =============================

def remove_special_char(text):
    """Remove special characters from text"""
    if pd.isnull(text): return text
    return re.sub(r"[^A-Za-z0-9\s.,!?\'\"-]", '', text)

def is_english(text, threshold=0.5):
    """Check if text is primarily English"""
    if pd.isnull(text): return False
    tokens = re.findall(r'\b\w+\b', text.lower())
    if not tokens: return False
    eng = sum(1 for t in tokens if t in english_vocab)
    return (eng / len(tokens)) >= threshold

def sentiment_label(text):
    """Basic VADER sentiment analysis"""
    if pd.isnull(text): return 'Neutral'
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def create_employee_id(df):
    """Create synthetic employee IDs"""
    df['employee_id'] = df['job-title'].fillna('Unknown') + '_' + df['curr/ex-flg']
    df['employee_id'] = df['employee_id'].str.replace(' ', '_')
    return df

# =============================
# ADVANCED SENTIMENT ANALYSIS
# =============================

def advanced_sentiment_analysis(text, threshold_pos=0.6, threshold_neg=0.4):
    """Advanced sentiment analysis using RoBERTa"""
    if pd.isnull(text) or not isinstance(text, str) or text.strip() == '':
        return 'Neutral'

    try:
        # Load model if not already loaded (global variable to avoid reloading)
        global roberta_pipeline
        if 'roberta_pipeline' not in globals():
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            roberta_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=0 if hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available() else -1,
                return_all_scores=True
            )

        # Get sentiment scores
        results = roberta_pipeline(text[:512])
        scores = {result['label']: result['score'] for result in results[0]}

        # Map to our labels
        pos_score = scores.get('LABEL_2', 0)  # Positive
        neg_score = scores.get('LABEL_0', 0)  # Negative

        # Classification with thresholds
        if pos_score >= threshold_pos:
            return 'Positive'
        elif neg_score >= threshold_neg:
            return 'Negative'
        else:
            return 'Neutral'

    except Exception as e:
        print(f"Error in advanced sentiment: {e}")
        return 'Neutral'

# =============================
# TOPIC MODELING
# =============================

def perform_topic_modeling(df, sentiment_type='all', max_samples=1000):
    """Perform BERTopic analysis"""
    if not ADVANCED_COMPONENTS_AVAILABLE:
        print("BERTopic not available, skipping topic modeling")
        return None

    try:
        # Filter data
        if sentiment_type == 'all':
            texts = df['pros&cons'].dropna().tolist()
        else:
            texts = df[df['sentiment'] == sentiment_type]['pros&cons'].dropna().tolist()

        texts = texts[:max_samples]

        # Create and fit topic model
        topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=False)
        topics, probabilities = topic_model.fit_transform(texts)

        return {
            'model': topic_model,
            'topics': topics,
            'probabilities': probabilities,
            'topic_info': topic_model.get_topic_info(),
            'texts': texts
        }

    except Exception as e:
        print(f"Topic modeling failed: {e}")
        return None

# =============================
# CHURN PREDICTION
# =============================

def create_churn_labels(df, sentiment_column='sentiment', time_window=30):
    """Create churn labels based on sentiment patterns"""
    # Group by employee and get recent sentiment history
    employee_sentiment = df.groupby('employee_id')[sentiment_column].agg(list).reset_index()

    def calculate_churn_risk(sentiments):
        if not sentiments:
            return 0
        # Take last time_window days of sentiment
        recent_sentiments = sentiments[-time_window:] if len(sentiments) >= time_window else sentiments
        negative_ratio = sum(1 for s in recent_sentiments if s == 'Negative') / len(recent_sentiments)
        return 1 if negative_ratio > 0.6 else 0  # High churn risk if >60% negative

    employee_sentiment['churn_risk'] = employee_sentiment[sentiment_column].apply(calculate_churn_risk)
    return employee_sentiment[['employee_id', 'churn_risk']]

def engineer_features(df):
    """Create features for churn prediction model"""
    features = []

    for employee_id in df['employee_id'].unique():
        employee_data = df[df['employee_id'] == employee_id].copy()

        if len(employee_data) < 3:  # Minimum reviews
            continue

        # Basic sentiment features
        sentiment_counts = employee_data['sentiment'].value_counts()
        total_reviews = len(employee_data)

        # Sentiment ratios
        pos_ratio = sentiment_counts.get('Positive', 0) / total_reviews
        neg_ratio = sentiment_counts.get('Negative', 0) / total_reviews
        neu_ratio = sentiment_counts.get('Neutral', 0) / total_reviews

        # Sentiment trends
        mid_point = len(employee_data) // 2
        recent_sentiment = employee_data['sentiment'].iloc[mid_point:].value_counts()
        older_sentiment = employee_data['sentiment'].iloc[:mid_point].value_counts()

        recent_neg_ratio = recent_sentiment.get('Negative', 0) / max(1, recent_sentiment.sum())
        older_neg_ratio = older_sentiment.get('Negative', 0) / max(1, older_sentiment.sum())
        sentiment_trend = recent_neg_ratio - older_neg_ratio

        # Frequency features
        review_frequency = total_reviews / max(1, (employee_data['dates'].max() - employee_data['dates'].min()).days)

        # Text length features
        avg_message_length = employee_data['message_length'].mean()
        avg_word_count = employee_data['word_count'].mean()

        # Rating features
        avg_rating = employee_data['overall-ratings'].mean()
        rating_volatility = employee_data['overall-ratings'].std()

        # Time-based features
        first_review = employee_data['dates'].min()
        last_review = employee_data['dates'].max()
        tenure_days = (last_review - first_review).days

        feature_dict = {
            'employee_id': employee_id,
            'total_reviews': total_reviews,
            'positive_ratio': pos_ratio,
            'negative_ratio': neg_ratio,
            'neutral_ratio': neu_ratio,
            'sentiment_trend': sentiment_trend,
            'review_frequency': review_frequency,
            'avg_message_length': avg_message_length,
            'avg_word_count': avg_word_count,
            'avg_rating': avg_rating,
            'rating_volatility': rating_volatility,
            'tenure_days': tenure_days
        }

        features.append(feature_dict)

    return pd.DataFrame(features)

def train_churn_model(features_df, churn_labels, test_size=0.2, random_state=42):
    """Train and evaluate churn prediction model"""
    # Merge features with churn labels
    model_data = features_df.merge(churn_labels, on='employee_id', how='left')
    model_data = model_data.fillna(0)

    # Prepare features and target
    feature_cols = [col for col in model_data.columns if col not in ['employee_id', 'churn_risk']]
    X = model_data[feature_cols]
    y = model_data['churn_risk']

    # Handle class imbalance
    if y.sum() < len(y) * 0.1:
        print("Warning: Low churn rate detected.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    # Evaluation
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    return {
        'model': rf_model,
        'scaler': scaler,
        'selected_features': feature_cols,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'classification_report': report,
        'auc_score': auc_score,
        'X_test': X_test,
        'y_test': y_test
    }

# =============================
# ANOMALY DETECTION
# =============================

def detect_sentiment_anomalies(df, sentiment_column='sentiment', date_column='dates', window_size=30):
    """Detect sentiment anomalies using statistical methods"""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    df = df.sort_values(date_column)

    # Create daily sentiment counts
    daily_sentiment = df.groupby(df[date_column].dt.date)[sentiment_column].value_counts().unstack(fill_value=0)
    daily_sentiment['total_reviews'] = daily_sentiment.sum(axis=1)
    daily_sentiment['negative_ratio'] = daily_sentiment.get('Negative', 0) / daily_sentiment['total_reviews']

    # Rolling statistics
    daily_sentiment['neg_ratio_ma'] = daily_sentiment['negative_ratio'].rolling(window=window_size, center=True).mean()
    daily_sentiment['neg_ratio_std'] = daily_sentiment['negative_ratio'].rolling(window=window_size, center=True).std()
    daily_sentiment['neg_ratio_zscore'] = (daily_sentiment['negative_ratio'] - daily_sentiment['neg_ratio_ma']) / daily_sentiment['neg_ratio_std']
    daily_sentiment['is_anomaly'] = abs(daily_sentiment['neg_ratio_zscore']) > 3

    return daily_sentiment

def ml_anomaly_detection(df, features=['negative_ratio', 'total_reviews'], contamination=0.1):
    """ML-based anomaly detection"""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    feature_data = df[features].copy()
    feature_data = feature_data.fillna(feature_data.mean())

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_scores = iso_forest.fit_predict(scaled_features)

    df = df.copy()
    df['ml_anomaly_score'] = iso_forest.decision_function(scaled_features)
    df['is_ml_anomaly'] = anomaly_scores == -1

    return df, iso_forest, scaler

# =============================
# VISUALIZATION FUNCTIONS
# =============================

def create_sentiment_distribution_pie(df):
    """Create sentiment distribution pie chart"""
    sentiment_dist = df['sentiment'].value_counts()
    colors = ['#4CAF50', '#F44336', '#FFC107']

    plt.figure(figsize=(10, 8))
    plt.pie(sentiment_dist.values, labels=sentiment_dist.index, colors=colors,
            autopct='%1.1f%%', startangle=90, shadow=True, explode=(0.05, 0, 0))
    plt.title('Employee Sentiment Distribution', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.savefig('visualizations/sentiment_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def create_sentiment_trends(df):
    """Create sentiment trends over time"""
    monthly_sentiment = df.groupby(['dates', 'sentiment']).size().unstack(fill_value=0)
    monthly_sentiment = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0)

    plt.figure(figsize=(15, 8))
    for sentiment in monthly_sentiment.columns:
        plt.plot(monthly_sentiment.index, monthly_sentiment[sentiment], marker='o', linewidth=2, label=sentiment)

    plt.title('Sentiment Trends Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_trends_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def create_employee_rankings_table(monthly_scores):
    """Create employee rankings visualization"""
    recent_months = sorted(monthly_scores['month'].unique())[-6:]
    top_positive = []
    top_negative = []

    for month in recent_months:
        month_data = monthly_scores[monthly_scores['month'] == month]
        top_pos = month_data.nlargest(3, 'monthly_score')[['employee_id', 'monthly_score']]
        top_neg = month_data.nsmallest(3, 'monthly_score')[['employee_id', 'monthly_score']]
        top_positive.append(top_pos.assign(month=month))
        top_negative.append(top_neg.assign(month=month))

    top_positive_df = pd.concat(top_positive)
    top_negative_df = pd.concat(top_negative)

    # Create table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    table_data = []
    for month in recent_months:
        pos_data = top_positive_df[top_positive_df['month'] == month]
        neg_data = top_negative_df[top_negative_df['month'] == month]

        table_data.append([month, 'Top Positive', pos_data.iloc[0]['employee_id'] if len(pos_data) > 0 else 'N/A',
                          pos_data.iloc[0]['monthly_score'] if len(pos_data) > 0 else 0])
        if len(pos_data) > 1:
            table_data.append(['', '', pos_data.iloc[1]['employee_id'], pos_data.iloc[1]['monthly_score']])
        if len(pos_data) > 2:
            table_data.append(['', '', pos_data.iloc[2]['employee_id'], pos_data.iloc[2]['monthly_score']])

        table_data.append(['', 'Top Negative', neg_data.iloc[0]['employee_id'] if len(neg_data) > 0 else 'N/A',
                          neg_data.iloc[0]['monthly_score'] if len(neg_data) > 0 else 0])
        if len(neg_data) > 1:
            table_data.append(['', '', neg_data.iloc[1]['employee_id'], neg_data.iloc[1]['monthly_score']])
        if len(neg_data) > 2:
            table_data.append(['', '', neg_data.iloc[2]['employee_id'], neg_data.iloc[2]['monthly_score']])

    table = ax.table(cellText=table_data, colLabels=['Month', 'Category', 'Employee ID', 'Score'],
                    cellLoc='center', loc='center', colWidths=[0.2, 0.3, 0.4, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title('Employee Rankings by Month', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('visualizations/employee_rankings_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_word_clouds(df):
    """Create sentiment word clouds"""
    def create_sentiment_wordcloud(sentiment_type, color_map):
        text = ' '.join(df[df['sentiment'] == sentiment_type]['pros&cons'].dropna())
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                colormap=color_map, max_words=100,
                                contour_width=1, contour_color='steelblue').generate(text)

            plt.figure(figsize=(16, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'{sentiment_type} Sentiment Word Cloud', fontsize=20, fontweight='bold', pad=20)
            plt.savefig(f'visualizations/{sentiment_type.lower()}_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

    create_sentiment_wordcloud('Positive', 'Greens')
    create_sentiment_wordcloud('Negative', 'Reds')
    create_sentiment_wordcloud('Neutral', 'Blues')

# =============================
# MAIN ANALYSIS PIPELINE
# =============================

def main():
    """Main analysis pipeline"""
    print("🚀 Starting Employee Sentiment Analysis...")

    # Load and preprocess data
    df = pd.read_csv('employee_reviews.csv', encoding='unicode_escape')
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

    # Basic cleaning
    df['curr/ex-flg'] = df['job-title'].str.split('-', expand=True)[0]
    df['job-title'] = df['job-title'].str.split('-', n=1).str[1]
    df['dates'] = df['dates'].str.strip()
    df['dates'] = pd.to_datetime(df['dates'], errors='coerce')
    df['dates'] = df['dates'].dt.strftime('%Y-%m')
    df.dropna(how='all', inplace=True)
    df.drop(index=54743, errors='ignore', inplace=True)

    # Process text and filter English
    df['pros&cons'] = df['pros&cons'].apply(remove_special_char)
    df['is_english'] = df['pros&cons'].apply(is_english)
    df_english = df[df['is_english'] == True].copy()

    # Basic sentiment analysis
    df_english['sentiment'] = df_english['pros&cons'].apply(sentiment_label)

    # Create employee IDs and additional features
    df_english = create_employee_id(df_english)
    df_english['message_length'] = df_english['pros&cons'].str.len()
    df_english['word_count'] = df_english['pros&cons'].str.split().str.len()

    print(f"Processed {len(df_english)} English reviews")

    # =============================
    # BASIC ANALYSIS
    # =============================

    print("\n📊 Performing Basic Analysis...")

    # Task 1: Create sentiment distribution pie chart
    print("\n=== Task 1: Sentiment Distribution Visualization ===")
    create_sentiment_distribution_pie(df_english)
    print("Sentiment distribution pie chart created and displayed")

    # Task 2: Create sentiment trends over time
    print("\n=== Task 2: Exploratory Data Analysis ===")
    create_sentiment_trends(df_english)
    print("Dataset structure:")
    print(f"Total reviews: {len(df_english)}")
    # Get date range safely
    valid_dates = pd.to_datetime(df_english['dates'], errors='coerce').dropna()
    if len(valid_dates) > 0:
        print(f"Date range: {valid_dates.min().strftime('%Y-%m')} to {valid_dates.max().strftime('%Y-%m')}")
    else:
        print("Date range: Unable to determine")
    print("Sentiment trends chart created and displayed")

    # Task 3: Create word clouds
    print("\n=== Task 3: Employee Score Calculation ===")
    create_word_clouds(df_english)
    print("Sentiment word clouds created and displayed")

    # Task 4: Employee scoring
    sentiment_scores = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    df_english['sentiment_score'] = df_english['sentiment'].map(sentiment_scores)
    monthly_scores = df_english.groupby(['employee_id', 'dates'])['sentiment_score'].sum().reset_index()
    monthly_scores.columns = ['employee_id', 'month', 'monthly_score']
    print(f"Monthly scores calculated for {monthly_scores['employee_id'].nunique()} unique employees")
    print("\nSample monthly scores:")
    print(monthly_scores.head(10).to_string())

    # Task 5: Employee rankings
    print("\n=== Task 4: Employee Ranking ===")
    create_employee_rankings_table(monthly_scores)
    print("Employee Rankings by Month:")
    recent_months = sorted(monthly_scores['month'].unique())[-6:]
    top_positive = []
    top_negative = []

    for month in recent_months:
        month_data = monthly_scores[monthly_scores['month'] == month]
        top_pos = month_data.nlargest(3, 'monthly_score')[['employee_id', 'monthly_score']]
        top_neg = month_data.nsmallest(3, 'monthly_score')[['employee_id', 'monthly_score']]
        top_positive.append(top_pos.assign(month=month))
        top_negative.append(top_neg.assign(month=month))

    top_positive_df = pd.concat(top_positive)
    top_negative_df = pd.concat(top_negative)

    for month in recent_months:
        print(f"\n{month}:")
        pos_data = top_positive_df[top_positive_df['month'] == month]
        neg_data = top_negative_df[top_negative_df['month'] == month]

        print("Top 3 Positive:")
        for i in range(min(3, len(pos_data))):
            print(f"  {pos_data.iloc[i]['employee_id']}: {int(pos_data.iloc[i]['monthly_score'])}")

        print("Top 3 Negative:")
        for i in range(min(3, len(neg_data))):
            print(f"  {neg_data.iloc[i]['employee_id']}: {int(neg_data.iloc[i]['monthly_score'])}")

    # Task 6: Flight risk analysis
    print("\n=== Task 5: Flight Risk Identification ===")
    monthly_negative = df_english[df_english['sentiment'] == 'Negative'].groupby(['employee_id', 'dates']).size().reset_index(name='negative_count')
    monthly_negative['negative_count_30d'] = monthly_negative.groupby('employee_id')['negative_count'].rolling(window=1, min_periods=1).sum().reset_index(0, drop=True)
    flight_risks = monthly_negative[monthly_negative['negative_count_30d'] >= 4]['employee_id'].unique()
    print(f"Flight Risk Employees: {len(flight_risks)}")
    print("\nFlight risk employee IDs:")
    for i, emp_id in enumerate(flight_risks[:10]):
        print(f"  {emp_id}")
    if len(flight_risks) > 10:
        print(f"  ... and {len(flight_risks) - 10} more")

    # =============================
    # ADVANCED ANALYSIS (DISABLED FOR SPEED)
    # =============================

    advanced_results = {}

    # Skip advanced analysis to speed up execution
    # if ADVANCED_COMPONENTS_AVAILABLE:
    #     print("\n🤖 Performing Advanced Analysis...")
    #
    #     # Skip advanced sentiment analysis to speed up execution
    #     # sample_df = df_english.sample(n=min(1000, len(df_english)), random_state=42)
    #     # sample_df['advanced_sentiment'] = sample_df['pros&cons'].apply(advanced_sentiment_analysis)
    #     # agreement = (sample_df['sentiment'] == sample_df['advanced_sentiment']).mean()
    #     # print(f"VADER vs RoBERTa agreement: {agreement:.2%}")
    #
    #     # Topic modeling
    #     # topic_results = perform_topic_modeling(df_english, max_samples=1000)
    #     # if topic_results:
    #     #     print(f"Found {len(topic_results['topic_info'])} topics")
    #
    #     # Churn prediction
    #     # churn_labels = create_churn_labels(df_english)
    #     # features_df = engineer_features(df_english)
    #     # churn_model_results = train_churn_model(features_df, churn_labels)
    #
    #     # print(f"Churn model AUC: {churn_model_results['auc_score']:.3f}")
    #     # Anomaly detection
    #     # daily_sentiment = detect_sentiment_anomalies(df_english)
    #     # ml_anomalies, _, _ = ml_anomaly_detection(daily_sentiment)
    #
    #     # anomaly_count = daily_sentiment['is_anomaly'].sum()
    #     # ml_anomaly_count = ml_anomalies['is_ml_anomaly'].sum()
    #
    #     # print(f"Found {anomaly_count} statistical anomalies and {ml_anomaly_count} ML anomalies")
    #
    #     # advanced_results = {
    #     #     # 'sentiment_agreement': agreement,
    #     #     'topics_found': len(topic_results['topic_info']) if topic_results else 0,
    #     #     'churn_model_auc': churn_model_results['auc_score'],
    #     #     'statistical_anomalies': int(anomaly_count),
    #     #     'ml_anomalies': int(ml_anomaly_count)
    #     # }

    # Save results
    with open('analysis_results.json', 'w') as f:
        json.dump({
            'basic_stats': {
                'total_reviews': len(df_english),
                'sentiment_distribution': df_english['sentiment'].value_counts().to_dict(),
                'unique_employees': monthly_scores['employee_id'].nunique(),
                'flight_risks': len(flight_risks)
            },
            'advanced_results': advanced_results
        }, f, indent=2)

    print("\n✅ Analysis Complete!")
    print("📁 Generated visualizations in 'visualizations/' folder")
    print("📄 Results saved to 'analysis_results.json'")

    return df_english, monthly_scores, flight_risks

if __name__ == "__main__":
    # Run main analysis
    df_processed, monthly_scores, flight_risks = main()

    print("\n🎉 Employee Sentiment Analysis completed successfully!")
    print(f"📊 Processed {len(df_processed)} reviews")
    print(f"👥 Analyzed {monthly_scores['employee_id'].nunique()} employees")
    print(f"⚠️ Identified {len(flight_risks)} flight risk employees")
