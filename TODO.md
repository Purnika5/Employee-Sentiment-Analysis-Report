# TODO: Employee Sentiment Analysis - Implementation Plan

## Phase 1: Data Loading & Preprocessing
- [ ] 1.1 Create new main_analysis.py to read test.xlsx
- [ ] 1.2 Map columns: body+Subject → text, from → employee_id, date → dates
- [ ] 1.3 Handle date parsing and missing values
- [ ] 1.4 Basic text cleaning

## Phase 2: Task 1 - Sentiment Labeling
- [ ] 2.1 Implement VADER sentiment analysis
- [ ] 2.2 Label each message as Positive/Negative/Neutral
- [ ] 2.3 Add sentiment column to dataframe

## Phase 3: Task 2 - Exploratory Data Analysis
- [ ] 3.1 Analyze data structure (records, types, missing values)
- [ ] 3.2 Sentiment distribution visualization (pie chart)
- [ ] 3.3 Time-based trends visualization
- [ ] 3.4 Word clouds by sentiment

## Phase 4: Task 3 - Employee Score Calculation
- [ ] 4.1 Assign scores: Positive=+1, Negative=-1, Neutral=0
- [ ] 4.2 Calculate monthly aggregated scores per employee
- [ ] 4.3 Document methodology

## Phase 5: Task 4 - Employee Ranking
- [ ] 5.1 Create top 3 positive employees per month
- [ ] 5.2 Create top 3 negative employees per month
- [ ] 5.3 Sort by score (descending), then alphabetically

## Phase 6: Task 5 - Flight Risk Identification
- [ ] 6.1 Implement 30-day rolling window for negative messages
- [ ] 6.2 Flag employees with 4+ negative emails in 30 days
- [ ] 6.3 Extract flight risk employee list

## Phase 7: Task 6 - Predictive Modeling
- [ ] 7.1 Feature engineering (message frequency, length, word count)
- [ ] 7.2 Train/test split
- [ ] 7.3 Build Linear Regression model
- [ ] 7.4 Evaluate with metrics (R², MSE)

## Phase 8: Final Deliverables
- [ ] 8.1 Generate all visualizations
- [ ] 8.2 Save analysis results to JSON
- [ ] 8.3 Create README.md summary

