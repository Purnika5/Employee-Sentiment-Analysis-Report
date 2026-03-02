#!/usr/bin/env python3
"""
Deployment Validation Script
Validates that the Employee Sentiment Analysis System is ready for production deployment
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {'Found' if exists else 'Missing'}")
    return exists

def check_directory_structure():
    """Validate directory structure"""
    print("\n📁 Checking Directory Structure...")

    required_files = [
        ('employee_reviews.csv', 'Main dataset'),
        ('employee_sentiment_analysis.py', 'Main analysis script'),
        ('app.py', 'Streamlit dashboard'),
        ('comprehensive_visualizations.py', 'Visualization module'),
        ('advanced_sentiment.py', 'Advanced sentiment analysis'),
        ('topic_modeling.py', 'Topic modeling module'),
        ('README.md', 'Documentation'),
        ('TODO.md', 'Project roadmap')
    ]

    directories = [
        ('visualizations', 'Output visualizations directory'),
        ('iframe_figures', 'Interactive figures directory')
    ]

    all_good = True

    for filename, description in required_files:
        if not check_file_exists(filename, description):
            all_good = False

    for dirname, description in directories:
        exists = os.path.exists(dirname)
        status = "✅" if exists else "❌"
        print(f"{status} {description}: {'Found' if exists else 'Missing'}")
        if not exists:
            all_good = False

    return all_good

def check_python_dependencies():
    """Check if required Python packages are installed"""
    print("\n🐍 Checking Python Dependencies...")

    required_packages = [
        'pandas', 'numpy', 'nltk', 'matplotlib', 'seaborn',
        'plotly', 'streamlit', 'scikit-learn', 'transformers',
        'bertopic', 'torch', 'wordcloud'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
        return False

    print("✅ All required packages are installed")
    return True

def validate_streamlit_app():
    """Validate Streamlit app structure"""
    print("\n🌐 Validating Streamlit App...")

    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for required Streamlit components
        required_elements = [
            'import streamlit',
            'st.title',
            'st.sidebar',
            'def main',
            'if __name__'
        ]

        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)

        if missing_elements:
            print(f"❌ Missing Streamlit elements: {missing_elements}")
            return False

        print("✅ Streamlit app structure is valid")
        return True

    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False

def check_data_integrity():
    """Check data file integrity"""
    print("\n📊 Checking Data Integrity...")

    try:
        import pandas as pd

        # Check CSV file
        df = pd.read_csv('employee_reviews.csv', encoding='unicode_escape')

        required_columns = ['job-title', 'pros&cons', 'overall-ratings', 'dates']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False

        print(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"✅ Required columns present: {required_columns}")

        # Check for data quality
        null_pros_cons = df['pros&cons'].isnull().sum()
        if null_pros_cons > 0:
            print(f"⚠️  {null_pros_cons} null values in pros&cons column")

        return True

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without full analysis"""
    print("\n🧪 Testing Basic Functionality...")

    try:
        import pandas as pd
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk

        # Load small sample
        df = pd.read_csv('employee_reviews.csv', encoding='unicode_escape', nrows=100)

        # Test sentiment analysis
        sia = SentimentIntensityAnalyzer()
        sample_text = df['pros&cons'].dropna().iloc[0]
        score = sia.polarity_scores(str(sample_text))

        print("✅ Sentiment analysis working")
        print(f"Sample sentiment score: {score['compound']:.3f}")
        # Test basic plotting
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        df['overall-ratings'].value_counts().plot(kind='bar')
        plt.title('Test Plot')
        plt.savefig('deployment_test_plot.png')
        plt.close()

        print("✅ Basic plotting working")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def check_documentation():
    """Check documentation completeness"""
    print("\n📚 Checking Documentation...")

    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()

        required_sections = [
            'Employee Sentiment Analysis',
            'Installation',
            'Usage',
            'Features',
            'Requirements'
        ]

        missing_sections = []
        for section in required_sections:
            if section.lower() not in readme_content.lower():
                missing_sections.append(section)

        if missing_sections:
            print(f"⚠️  Missing documentation sections: {missing_sections}")
            return False

        print("✅ Documentation is comprehensive")
        return True

    except Exception as e:
        print(f"❌ Error reading README: {e}")
        return False

def generate_deployment_report():
    """Generate deployment readiness report"""
    print("\n📋 Generating Deployment Report...")

    report = {
        "timestamp": str(pd.Timestamp.now()),
        "system_status": "READY" if all([
            check_directory_structure(),
            check_python_dependencies(),
            validate_streamlit_app(),
            check_data_integrity(),
            test_basic_functionality(),
            check_documentation()
        ]) else "NOT READY",
        "checks": {
            "directory_structure": check_directory_structure(),
            "python_dependencies": check_python_dependencies(),
            "streamlit_app": validate_streamlit_app(),
            "data_integrity": check_data_integrity(),
            "basic_functionality": test_basic_functionality(),
            "documentation": check_documentation()
        }
    }

    with open('deployment_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("✅ Deployment report saved to deployment_report.json")
    return report

def main():
    """Run deployment validation"""
    print("🚀 Employee Sentiment Analysis - Deployment Validation")
    print("=" * 60)

    # Run all checks
    checks_passed = 0
    total_checks = 6

    if check_directory_structure():
        checks_passed += 1

    if check_python_dependencies():
        checks_passed += 1

    if validate_streamlit_app():
        checks_passed += 1

    if check_data_integrity():
        checks_passed += 1

    if test_basic_functionality():
        checks_passed += 1

    if check_documentation():
        checks_passed += 1

    # Generate report
    report = generate_deployment_report()

    # Final summary
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT VALIDATION SUMMARY")
    print(f"Checks Passed: {checks_passed}/{total_checks}")

    if checks_passed == total_checks:
        print("🎉 SYSTEM IS READY FOR DEPLOYMENT!")
        print("\n🚀 Deployment Commands:")
        print("1. Start Streamlit Dashboard: streamlit run app.py")
        print("2. Run Full Analysis: python employee_sentiment_analysis.py")
        print("3. Access Dashboard: http://localhost:8501")
    else:
        print("⚠️  SYSTEM NEEDS ATTENTION BEFORE DEPLOYMENT")
        print("Check the deployment_report.json for detailed results")

    return checks_passed == total_checks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
