#!/usr/bin/env python3
"""
Employee Sentiment Analysis - Main Analysis Script
Analyzes employee messages from test.xlsx dataset

Tasks:
1. Sentiment Labeling - Label messages as Positive/Negative/Neutral
2. Exploratory Data Analysis (EDA)
3. Employee Score Calculation
4. Employee Ranking
5. Flight Risk Identification
6. Predictive Modeling (Linear Regression)
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
from datetime import datetime, timedelta
from collections import Counter

# NLTK & Text Processing
import nltk
from nltk.corpus import words
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ML & Stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('words', quiet=True)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create visualizations folder
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

print("=" * 60)
print("EMPLOYEE SENTIMENT ANALYSIS")
print("=" * 60)
print("\nLibraries imported successfully!")

# =============================
# UTILITY FUNCTIONS
# =============================

def remove_special_char(text):
    """Remove special characters from text"""
    if pd.isnull(text): return ""
    text = str(text)
    return re.sub(r"[^A-Za-z0-9\s.,!?\'\"-]", ' ', text)

def sentiment_label(text):
    """
    Task 1: Sentiment Labeling
    Uses VADER sentiment analyzer to classify messages
    Returns: 'Positive', 'Negative', or 'Neutral'
    """
    if pd.isnull(text) or str(text).strip() == "":
        return 'Neutral'
    try:
        score = sia.polarity_scores(str(text))['compound']
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    except:
        return 'Neutral'

# =============================
# DATA LOADING & PREPROCESSING
# =============================

def load_and_preprocess_data(file_path='test.xlsx'):
    """Load data from test.xlsx and preprocess for analysis"""
    print("\n" + "=" * 60)
    print("TASK 1: DATA LOADING & PREPROCESSING")
    print("=" * 60)
    
    df = pd.read_excel(file_path)
    print(f"\nDataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    
    print(f"\n--- Initial Data Structure ---")
    print(f"Total records: {len(df)}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    
    # Combine Subject and body for full text analysis
    df['Subject'] = df['Subject'].fillna('').astype(str)
    df['body'] = df['body'].fillna('').astype(str)
    df['full_text'] = df['Subject'] + ' ' + df['body']
    
    # Clean text
    df['clean_text'] = df['full_text'].apply(remove_special_char)
    
    # Parse dates - handle various formats
    df['parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract year-month for monthly aggregation
    df['month'] = df['parsed_date'].dt.to_period('M').astype(str)
    
    # Employee identifier from 'from' column
    df['employee_id'] = df['from'].fillna('unknown@enron.com')
    
    # Filter valid records
    df_valid = df[(df['clean_text'].str.len() > 10) & (df['parsed_date'].notna())].copy()
    
    print(f"\nAfter filtering valid records: {len(df_valid)} rows")
    
    valid_dates = df_valid['parsed_date'].dropna()
    if len(valid_dates) > 0:
        print(f"\n  Date range: {valid_dates.min().strftime('%Y-%m-%d')} to {valid_dates.max().strftime('%Y-%m-%d')}")
    
    return df_valid

# =============================
# TASK 1: SENTIMENT LABELING
# =============================

def perform_sentiment_labeling(df):
    """Task 1: Sentiment Labeling"""
    print("\n" + "=" * 60)
    print("TASK 1: SENTIMENT LABELING")
    print("=" * 60)
    
    print("\nUsing VADER Sentiment Analysis for labeling...")
    print("  - Positive: compound score >= 0.05")
    print("  - Negative: compound score <= -0.05")
    print("  - Neutral: compound score between -0.05 and 0.05")
    
    df['sentiment'] = df['clean_text'].apply(sentiment_label)
    sentiment_dist = df['sentiment'].value_counts()
    
    print(f"\nSentiment labeling complete!")
    print(f"\n--- Sentiment Distribution ---")
    for sent, count in sentiment_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {sent}: {count} ({pct:.1f}%)")
    
    # Add sentiment scores for Task 3
    sentiment_scores = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
    df['sentiment_score'] = df['sentiment'].map(sentiment_scores)
    
    return df

# =============================
# TASK 2: EXPLORATORY DATA ANALYSIS
# =============================

def perform_eda(df):
    """Task 2: EDA"""
    print("\n" + "=" * 60)
    print("TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    print("\n--- 2.1 Data Structure Analysis ---")
    print(f"  Total messages: {len(df)}")
    print(f"  Unique employees: {df['employee_id'].nunique()}")
    print(f"  Date range: {df['month'].min()} to {df['month'].max()}")
    print(f"  Unique months: {df['month'].nunique()}")
    
    print("\n--- 2.2 Sentiment Distribution ---")
    
    # Pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    sentiment_dist = df['sentiment'].value_counts()
    colors = ['#4CAF50', '#F44336', '#FFC107']
    explode = (0.05, 0.05, 0.05)
    
    ax.pie(sentiment_dist.values, labels=sentiment_dist.index, colors=colors,
           autopct='%1.1f%%', startangle=90, explode=explode, shadow=True)
    ax.set_title('Employee Sentiment Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Saved: visualizations/sentiment_distribution_pie.png")
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sentiment_dist.index, sentiment_dist.values, color=colors, edgecolor='black')
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Number of Messages', fontsize=12)
    ax.set_title('Sentiment Distribution by Count', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, sentiment_dist.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(val), ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_distribution_bar.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("  Saved: visualizations/sentiment_distribution_bar.png")
    
    print("\n--- 2.3 Time-based Trends ---")
    
    monthly_sentiment = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
    monthly_total = monthly_sentiment.sum(axis=1)
    monthly_pct = monthly_sentiment.div(monthly_total, axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors_line = {'Positive': '#4CAF50', 'Negative': '#F44336', 'Neutral': '#FFC107'}
    
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment in monthly_pct.columns:
            ax.plot(monthly_pct.index, monthly_pct[sentiment], 
                   marker='o', linewidth=2, label=sentiment, color=colors_line[sentiment])
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Sentiment Trends Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_trends_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: visualizations/sentiment_trends_over_time.png")
    
    print("\n--- 2.4 Word Clouds by Sentiment ---")
    
    def create_sentiment_wordcloud(sentiment_type, color_map):
        text = ' '.join(df[df['sentiment'] == sentiment_type]['clean_text'].dropna())
        if len(text.strip()) > 0:
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                colormap=color_map, max_words=100,
                                min_font_size=10).generate(text)
            
            plt.figure(figsize=(16, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'{sentiment_type} Sentiment Word Cloud', fontsize=16, fontweight='bold')
            plt.savefig(f'visualizations/{sentiment_type.lower()}_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
            return True
        return False
    
    create_sentiment_wordcloud('Positive', 'Greens')
    create_sentiment_wordcloud('Negative', 'Reds')
    create_sentiment_wordcloud('Neutral', 'Blues')
    print("  Saved: word clouds for each sentiment type")
    
    print("\n--- 2.5 Additional Patterns ---")
    
    df['message_length'] = df['clean_text'].str.len()
    df['word_count'] = df['clean_text'].str.split().str.len()
    
    print(f"  Average message length: {df['message_length'].mean():.1f} characters")
    print(f"  Average word count: {df['word_count'].mean():.1f} words")
    
    msg_per_employee = df['employee_id'].value_counts()
    print(f"  Messages per employee: min={msg_per_employee.min()}, max={msg_per_employee.max()}, median={msg_per_employee.median():.0f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df['message_length'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Message Length (characters)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Message Length Distribution')
    axes[0].axvline(df['message_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["message_length"].mean():.0f}')
    axes[0].legend()
    
    top_employees = msg_per_employee.head(20)
    axes[1].barh(range(len(top_employees)), top_employees.values, color='teal')
    axes[1].set_yticks(range(len(top_employees)))
    axes[1].set_yticklabels([e.split('@')[0][:15] for e in top_employees.index], fontsize=8)
    axes[1].set_xlabel('Number of Messages')
    axes[1].set_title('Top 20 Employees by Message Count')
    
    plt.tight_layout()
    plt.savefig('visualizations/data_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: visualizations/data_patterns.png")
    
    print("\nEDA complete!")
    return df

# =============================
# TASK 3: EMPLOYEE SCORE CALCULATION
# =============================

def calculate_employee_scores(df):
    """Task 3: Employee Score Calculation"""
    print("\n" + "=" * 60)
    print("TASK 3: EMPLOYEE SCORE CALCULATION")
    print("=" * 60)
    
    print("\nScoring Method:")
    print("  - Positive Message: +1 point")
    print("  - Negative Message: -1 point")
    print("  - Neutral Message: 0 points (no effect)")
    print("  - Monthly aggregation: Sum of scores per employee per month")
    
    monthly_scores = df.groupby(['employee_id', 'month'])['sentiment_score'].sum().reset_index()
    monthly_scores.columns = ['employee_id', 'month', 'monthly_score']
    monthly_scores = monthly_scores.sort_values(['month', 'monthly_score'], ascending=[True, False])
    
# =============================
# TASK 3: EMPLOYEE SCORE CALCULATION
# =============================

def calculate_employee_scores(df):
    """Task 3: Employee Score Calculation"""
    print("\n" + "=" * 60)
    print("TASK 3: EMPLOYEE SCORE CALCULATION")
    print("=" * 60)
    
    print("\nScoring Method:")
    print("  - Positive Message: +1 point")
    print("  - Negative Message: -1 point")
    print("  - Neutral Message: 0 points (no effect)")
    print("  - Monthly aggregation: Sum of scores per employee per month")
    
    monthly_scores = df.groupby(['employee_id', 'month'])['sentiment_score'].sum().reset_index()
    monthly_scores.columns = ['employee_id', 'month', 'monthly_score']
    monthly_scores = monthly_scores.sort_values(['month', 'monthly_score'], ascending=[True, False])
    
    print(f"\n✓ Monthly scores calculated")
    print(f"  Total employee-month combinations: {len(monthly_scores)}")
    print(f"  Unique employees scored: {monthly_scores['employee_id'].nunique()}")
    
    print(f"\n--- Sample Monthly Scores ---")
    print(monthly_scores.head(10).to_string(index=False))
    
    print(f"\n--- Score Statistics ---")
    print(f"  Highest score: {monthly_scores['monthly_score'].max()}")
    print(f"  Lowest score: {monthly_scores['monthly_score'].min()}")
    print(f"  Average score: {monthly_scores['monthly_score'].mean():.2f}")
    
    return monthly_scores

# =============================
# TASK 4: EMPLOYEE RANKING
# =============================

def rank_employees(monthly_scores):
    """Task 4: Employee Ranking"""
    print("\n" + "=" * 60)
    print("TASK 4: EMPLOYEE RANKING")
    print("=" * 60)
    
    months = sorted(monthly_scores['month'].unique())
    rankings = []
    
    print("\n--- Employee Rankings by Month ---")
    
    for month in months:
        month_data = monthly_scores[monthly_scores['month'] == month].copy()
        month_data_sorted = month_data.sort_values(
            by=['monthly_score', 'employee_id'], 
            ascending=[False, True]
        )
        
        top_positive = month_data_sorted.head(3)
        top_negative = month_data_sorted.tail(3).sort_values(by='monthly_score', ascending=True)
        
        print(f"\n{month}:")
        print("  Top 3 Positive:")
        for i, (_, row) in enumerate(top_positive.iterrows(), 1):
            emp_name = row['employee_id'].split('@')[0]
            print(f"    {i}. {emp_name}: {int(row['monthly_score'])} points")
            rankings.append({
                'month': month,
                'rank_type': 'Top Positive',
                'rank_position': i,
                'employee_id': row['employee_id'],
                'score': row['monthly_score']
            })
        
        print("  Top 3 Negative:")
        for i, (_, row) in enumerate(top_negative.iterrows(), 1):
            emp_name = row['employee_id'].split('@')[0]
            print(f"    {i}. {emp_name}: {int(row['monthly_score'])} points")
            rankings.append({
                'month': month,
                'rank_type': 'Top Negative',
                'rank_position': i,
                'employee_id': row['employee_id'],
                'score': row['monthly_score']
            })
    
    rankings_df = pd.DataFrame(rankings)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    table_data = []
    for month in months:
        month_ranks = rankings_df[rankings_df['month'] == month]
        pos_ranks = month_ranks[month_ranks['rank_type'] == 'Top Positive']
        neg_ranks = month_ranks[month_ranks['rank_type'] == 'Top Negative']
        
        row = [month]
        pos_str = '\n'.join([f"{r['employee_id'].split('@')[0]}: {int(r['score'])}" 
                            for _, r in pos_ranks.iterrows()])
        row.append(pos_str)
        neg_str = '\n'.join([f"{r['employee_id'].split('@')[0]}: {int(r['score'])}" 
                            for _, r in neg_ranks.iterrows()])
        row.append(neg_str)
        table_data.append(row)
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Month', 'Top 3 Positive', 'Top 3 Negative'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.4, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    plt.title('Employee Rankings by Month', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('visualizations/employee_rankings_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n  ✓ Saved: visualizations/employee_rankings_table.png")
    
    return rankings_df

# =============================
# TASK 5: FLIGHT RISK IDENTIFICATION
# =============================

def identify_flight_risks(df):
    """Task 5: Flight Risk Identification"""
    print("\n" + "=" * 60)
    print("TASK 5: FLIGHT RISK IDENTIFICATION")
    print("=" * 60)
    
    print("\nCriteria:")
    print("  - Flight risk = 4 or more negative messages")
    print("  - Within a 30-day rolling window (irrespective of month)")
    
    negative_df = df[df['sentiment'] == 'Negative'].copy()
    negative_df = negative_df.sort_values(['employee_id', 'parsed_date'])
    
    print(f"\n  Total negative messages: {len(negative_df)}")
    print(f"  Unique employees with negative messages: {negative_df['employee_id'].nunique()}")
    
    flight_risks = set()
    
    for employee_id in negative_df['employee_id'].unique():
        emp_negative = negative_df[negative_df['employee_id'] == employee_id].copy()
        emp_negative = emp_negative.sort_values('parsed_date')
        
        if len(emp_negative) < 4:
            continue
        
        dates = emp_negative['parsed_date'].tolist()
        
        for i, start_date in enumerate(dates):
            end_date = start_date + timedelta(days=30)
            count = sum(1 for d in dates if start_date <= d <= end_date)
            
            if count >= 4:
                flight_risks.add(employee_id)
                break
    
    flight_risk_list = sorted(list(flight_risks))
    
    print(f"\n✓ Flight Risk Analysis Complete")
    print(f"  Employees identified as flight risks: {len(flight_risk_list)}")
    
    if flight_risk_list:
        print(f"\n--- Flight Risk Employee List ---")
        for emp in flight_risk_list[:20]:
            emp_name = emp.split('@')[0]
            neg_count = len(negative_df[negative_df['employee_id'] == emp])
            print(f"  - {emp_name}: {neg_count} negative messages")
        
        if len(flight_risk_list) > 20:
            print(f"  ... and {len(flight_risk_list) - 20} more")
    
    flight_risk_df = pd.DataFrame({
        'employee_id': flight_risk_list,
        'employee_name': [e.split('@')[0] for e in flight_risk_list],
        'negative_message_count': [len(negative_df[negative_df['employee_id'] == e]) for e in flight_risk_list]
    })
    flight_risk_df.to_csv('visualizations/flight_risk_employees.csv', index=False)
    print(f"\n  ✓ Saved: visualizations/flight_risk_employees.csv")
    
    return flight_risk_list

# =============================
# TASK 6: PREDICTIVE MODELING
# =============================

def build_predictive_model(df):
    """Task 6: Predictive Modeling"""
    print("\n" + "=" * 60)
    print("TASK 6: PREDICTIVE MODELING")
    print("=" * 60)
    
    print("\nFeatures selected for prediction:")
    print("  - message_length: Length of the message in characters")
    print("  - word_count: Number of words in the message")
    print("  - month_msg_count: Number of messages the employee sent that month")
    print("  - employee_avg_length: Employee's average message length historically")
    
    monthly_msg_count = df.groupby(['employee_id', 'month']).size().reset_index(name='month_msg_count')
    df = df.merge(monthly_msg_count, on=['employee_id', 'month'], how='left')
    
    emp_avg_length = df.groupby('employee_id')['message_length'].mean().reset_index(name='employee_avg_length')
    df = df.merge(emp_avg_length, on='employee_id', how='left')
    
    feature_cols = ['message_length', 'word_count', 'month_msg_count', 'employee_avg_length']
    
    model_df = df[feature_cols + ['sentiment_score']].dropna()
    
    X = model_df[feature_cols]
    y = model_df['sentiment_score']
    
    print(f"\n  Dataset size for modeling: {len(X)} samples")
    print(f"  Features: {feature_cols}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n✓ Linear Regression Model Trained")
    print(f"\n--- Model Performance ---")
    print(f"  Training Set:")
    print(f"    - R² Score: {train_r2:.4f}")
    print(f"    - RMSE: {train_rmse:.4f}")
    print(f"    - MAE: {train_mae:.4f}")
    print(f"\n  Test Set:")
    print(f"    - R² Score: {test_r2:.4f}")
    print(f"    - RMSE: {test_rmse:.4f}")
    print(f"    - MAE: {test_mae:.4f}")
    
    print(f"\n--- Feature Coefficients ---")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"  {feat}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_test, y_test_pred, alpha=0.5, color='steelblue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Sentiment Score')
    axes[0].set_ylabel('Predicted Sentiment Score')
    axes[0].set_title(f'Actual vs Predicted (Test R² = {test_r2:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=True)
    
    colors = ['#F44336' if c < 0 else '#4CAF50' for c in importance_df['Coefficient']]
    axes[1].barh(importance_df['Feature'], importance_df['Coefficient'], color=colors)
    axes[1].set_xlabel('Coefficient Value')
    axes[1].set_title('Feature Importance (Linear Regression Coefficients)')
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('visualizations/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Saved: visualizations/model_performance.png")
    
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'coefficients': dict(zip(feature_cols, model.coef_)),
        'intercept': model.intercept_
    }

# =============================
# MAIN EXECUTION
# =============================

def main():
    """Main analysis pipeline"""
    print("\n" + "=" * 60)
    print("STARTING EMPLOYEE SENTIMENT ANALYSIS")
    print("=" * 60)
    
    # Task 1: Load and preprocess data
    df = load_and_preprocess_data('test.xlsx')
    
    # Task 1: Sentiment Labeling
    df = perform_sentiment_labeling(df)
    
    # Task 2: EDA
    df = perform_eda(df)
    
    # Task 3: Employee Score Calculation
    monthly_scores = calculate_employee_scores(df)
    
    # Task 4: Employee Ranking
    rankings_df = rank_employees(monthly_scores)
    
    # Task 5: Flight Risk Identification
    flight_risks = identify_flight_risks(df)
    
    # Task 6: Predictive Modeling
    model_metrics = build_predictive_model(df)
    
    # =============================
    # SAVE RESULTS
    # =============================
    
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    df.to_csv('processed_employee_data.csv', index=False)
    print("\n✓ Saved: processed_employee_data.csv")
    
    monthly_scores.to_csv('monthly_employee_scores.csv', index=False)
    print("✓ Saved: monthly_employee_scores.csv")
    
    rankings_df.to_csv('employee_rankings.csv', index=False)
    print("✓ Saved: employee_rankings.csv")
    
    results = {
        'analysis_summary': {
            'total_messages': len(df),
            'unique_employees': df['employee_id'].nunique(),
            'date_range': f"{df['month'].min()} to {df['month'].max()}",
            'unique_months': df['month'].nunique()
        },
        'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
        'flight_risks_count': len(flight_risks),
        'flight_risk_employees': list(flight_risks),
        'model_performance': model_metrics
    }
    
    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("✓ Saved: analysis_results.json")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Total messages analyzed: {len(df)}")
    print(f"👥 Unique employees: {df['employee_id'].nunique()}")
    print(f"⚠️ Flight risk employees identified: {len(flight_risks)}")
    print(f"📁 All outputs saved in 'visualizations/' folder")
    
    return df, monthly_scores, rankings_df, flight_risks

if __name__ == "__main__":
    df_processed, monthly_scores, rankings, flight_risks = main()
    print("\n🎉 Employee Sentiment Analysis completed successfully!")
 