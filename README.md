# CredXplain - Real-Time Credit Scoring System 🏦

A comprehensive real-time credit scoring system that leverages machine learning, sentiment analysis, and financial data to provide transparent and explainable credit risk assessments.

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

- **Multi-source Data Ingestion**: Yahoo Finance, World Bank/FRED, Financial news RSS
- **Advanced Feature Engineering**: Financial ratios, sentiment analysis, trend indicators
- **Interpretable ML Models**: Decision Tree and Logistic Regression with SHAP explanations
- **Real-time Dashboard**: Interactive Streamlit interface with alerts and visualizations
- **Automated Pipeline**: Scheduled data updates and model retraining

## Project Structure

```
credit-scoring-system/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py          # Data fetching from multiple sources
│   │   ├── preprocessing.py      # Data cleaning and normalization
│   │   └── database.py          # Database operations
│   ├── features/
│   │   ├── __init__.py
│   │   ├── financial_ratios.py   # Financial ratio calculations
│   │   ├── sentiment_analysis.py # News sentiment processing
│   │   └── trend_indicators.py   # Technical indicators
│   ├── models/
│   │   ├── __init__.py
│   │   ├── scoring_engine.py     # ML models for credit scoring
│   │   └── explainability.py    # SHAP and interpretability
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py               # Main Streamlit application
│   │   └── components.py        # Dashboard components
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration settings
│       └── helpers.py           # Utility functions
├── data/
│   ├── raw/                     # Raw data storage
│   ├── processed/               # Processed data
│   └── database.db             # SQLite database
├── models/
│   └── trained/                # Saved model files
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── automation/
    └── scheduler.py            # Data refresh automation
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download spaCy model: `python -m spacy download en_core_web_sm`
4. Run the dashboard: `streamlit run src/dashboard/app.py`

## Usage

### Data Pipeline

The system automatically fetches and processes data from multiple sources, storing it in a normalized format.

### Credit Scoring

Uses ensemble methods combining financial ratios and sentiment analysis to generate interpretable credit scores (0-100).

### Dashboard

Interactive interface showing:

- Real-time credit scores
- Feature importance analysis
- News sentiment impact
- Historical trends and alerts

## API Endpoints

- `/score`: Get credit score for a company
- `/features`: Get feature importance breakdown
- `/news`: Get recent news sentiment analysis
- `/trends`: Get historical score trends

## Deployment

The system is containerized with Docker and can be deployed to cloud platforms like Heroku, Render, or AWS.
