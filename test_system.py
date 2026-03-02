#!/usr/bin/env python3
"""
System Validation Test Script
Tests core components of the Employee Sentiment Analysis System
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime

# Test results tracking
test_results = []

def log_test(test_name, status, message=""):
    """Log test results"""
    result = f"{'✅' if status else '❌'} {test_name}: {message}"
    test_results.append((test_name, status, message))
    print(result)
    return status

def test_data_loading():
    """Test data loading and basic preprocessing"""
    try:
        # Load data
        df = pd.read_csv('employee_reviews.csv', encoding='unicode_escape')
        log_test("Data Loading", True, f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Basic cleaning
        df['curr/ex-flg'] = df['job-title'].str.split('-', expand=True)[0]
        df['job-title'] = df['job-title'].str.split('-', n=1).str[1]
        df['dates'] = df['dates'].str.strip()
        df['dates'] = pd.to_datetime(df['dates'], errors='coerce')
        df['dates'] = df['dates'].dt.strftime('%Y-%m')

        log_test("Data Cleaning", True, "Basic preprocessing completed")
        return df
    except Exception as e:
        log_test("Data Loading", False, str(e))
        return None

def test_sentiment_analysis(df):
    """Test sentiment analysis functionality"""
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        # Initialize VADER
        sia = SentimentIntensityAnalyzer()

        # Test on sample data
        sample_reviews = df['pros&cons'].dropna().head(100)

        sentiments = []
        for review in sample_reviews:
            score = sia.polarity_scores(str(review))['compound']
            if score >= 0.05:
                sentiments.append('Positive')
            elif score <= -0.05:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')

        sentiment_dist = pd.Series(sentiments).value_counts()
        # Add sentiment column to df for the sample
        df.loc[sample_reviews.index, 'sentiment'] = sentiments
        log_test("Sentiment Analysis", True, f"Processed {len(sentiments)} reviews: {dict(sentiment_dist)}")
        return True
    except Exception as e:
        log_test("Sentiment Analysis", False, str(e))
        return False

def test_employee_scoring(df):
    """Test employee scoring functionality"""
    try:
        # Create employee IDs
        df_test = df.copy()
        df_test['employee_id'] = df_test['job-title'].fillna('Unknown') + '_' + df_test['curr/ex-flg']
        df_test['employee_id'] = df_test['employee_id'].str.replace(' ', '_')

        # Calculate scores
        sentiment_scores = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
        df_test['sentiment_score'] = df_test['sentiment'].map(sentiment_scores)

        monthly_scores = df_test.groupby(['employee_id', 'dates'])['sentiment_score'].sum().reset_index()
        monthly_scores.columns = ['employee_id', 'month', 'monthly_score']

        log_test("Employee Scoring", True, f"Created scores for {monthly_scores['employee_id'].nunique()} employees")
        return True
    except Exception as e:
        log_test("Employee Scoring", False, str(e))
        return False

def test_visualizations():
    """Test visualization generation"""
    try:
        import matplotlib.pyplot as plt
        import plotly.express as px

        # Create test data
        test_data = pd.DataFrame({
            'sentiment': ['Positive', 'Negative', 'Neutral'] * 10,
            'count': np.random.randint(1, 100, 30)
        })

        # Test matplotlib
        plt.figure(figsize=(8, 6))
        test_data.groupby('sentiment')['count'].sum().plot(kind='bar')
        plt.title('Test Sentiment Distribution')
        plt.savefig('test_plot.png')
        plt.close()

        # Test plotly
        fig = px.pie(values=[30, 20, 10], names=['Positive', 'Negative', 'Neutral'])
        fig.write_image('test_pie.png')

        log_test("Visualizations", True, "Generated test plots successfully")
        return True
    except Exception as e:
        log_test("Visualizations", False, str(e))
        return False

def test_dependencies():
    """Test advanced dependencies availability"""
    dependencies = {
        'transformers': 'Advanced sentiment analysis',
        'bertopic': 'Topic modeling',
        'scipy': 'Statistical analysis',
        'sklearn': 'Machine learning',
        'streamlit': 'Dashboard',
        'plotly': 'Interactive visualizations'
    }

    available_deps = []
    missing_deps = []

    for dep, purpose in dependencies.items():
        try:
            __import__(dep)
            available_deps.append(f"{dep} ({purpose})")
        except ImportError:
            missing_deps.append(f"{dep} ({purpose})")

    if available_deps:
        log_test("Available Dependencies", True, f"{len(available_deps)} found: {', '.join(available_deps[:3])}...")

    if missing_deps:
        log_test("Missing Dependencies", len(missing_deps) == 0, f"{len(missing_deps)} missing: {', '.join(missing_deps)}")

    return len(missing_deps) == 0

def test_streamlit_app():
    """Test Streamlit app accessibility"""
    try:
        import streamlit as st
        import subprocess
        import time

        # Check if app.py exists and is valid Python
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()

        if 'streamlit' in content.lower() and 'def main' in content:
            log_test("Streamlit App", True, "App file exists and contains Streamlit code")
            return True
        else:
            log_test("Streamlit App", False, "App file missing or invalid")
            return False

    except Exception as e:
        log_test("Streamlit App", False, str(e))
        return False

def run_performance_test():
    """Run basic performance test"""
    try:
        start_time = time.time()

        # Load and process sample data
        df = pd.read_csv('employee_reviews.csv', encoding='unicode_escape', nrows=1000)
        df['dates'] = pd.to_datetime(df['dates'], errors='coerce')

        end_time = time.time()
        processing_time = end_time - start_time

        log_test("Performance Test", True, f"Processed 1000 rows in {processing_time:.2f} seconds")
        return True
    except Exception as e:
        log_test("Performance Test", False, str(e))
        return False

def main():
    """Run all system validation tests"""
    print("🚀 Employee Sentiment Analysis System - Validation Tests")
    print("=" * 60)

    # Run tests
    df = test_data_loading()

    if df is not None:
        test_sentiment_analysis(df)
        test_employee_scoring(df)

    test_visualizations()
    test_dependencies()
    test_streamlit_app()
    run_performance_test()

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")

    passed = sum(1 for _, status, _ in test_results if status)
    total = len(test_results)

    print(f"Tests Passed: {passed}/{total}")

    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    # Detailed results
    print("\n📋 DETAILED RESULTS:")
    for test_name, status, message in test_results:
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {test_name}")
        if message:
            print(f"    {message}")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
