# Employee Sentiment Analysis

This system analyzes employee reviews to understand workplace sentiment and identify employees at risk of leaving.

## Quick Start

To run the analysis:
```
python main_analysis.py
```

## What You Need

- Python 3.8 or higher
- 8GB RAM minimum
- Employee reviews in CSV format (employee_reviews.csv)

## Installation

1. Install required packages:
```
pip install pandas numpy nltk matplotlib seaborn plotly wordcloud scikit-learn
```

2. Download NLTK data:
```
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('words')"
```

## How It Works

1. Reads employee review data
2. Analyzes sentiment (positive/negative/neutral)
3. Scores employees monthly
4. Identifies risk patterns

## Results

The system creates:
- Charts showing sentiment trends
- Employee rankings
- Risk alerts for employees who might leave
- Word clouds of common themes
- Detailed report in JSON format

## Sample Results

- 86% of reviews are positive
- 11% are negative
- 3% are neutral
- 12 employees identified as flight risks

## Files Created

- visualizations/sentiment_distribution_pie.png
- visualizations/sentiment_trends_over_time.png
- visualizations/positive_wordcloud.png
- visualizations/negative_wordcloud.png
- analysis_results.json
