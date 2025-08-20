# Configuration settings for the credit scoring system
import os
from typing import Dict, List


class Config:
    # Database settings
    DATABASE_URL = "sqlite:///data/database.db"

    # API Keys (set as environment variables)
    FRED_API_KEY = os.getenv("FRED_API_KEY", "your_fred_api_key_here")

    # Data sources
    COMPANIES = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "JPM",
        "BAC",
        "WMT",
        "V",
        "JNJ",
    ]

    # Financial ratios to calculate
    FINANCIAL_RATIOS = [
        "debt_to_equity",
        "current_ratio",
        "quick_ratio",
        "profit_margin",
        "roe",
        "roa",
        "cash_flow_ratio",
        "interest_coverage",
    ]

    # News sources RSS feeds
    NEWS_SOURCES = {
        "reuters": "https://www.reuters.com/business/finance/rss",
        "bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
        "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "financial_times": "https://www.ft.com/companies?format=rss",
    }

    # Model parameters
    MODEL_CONFIG = {
        "scoring_threshold": 0.5,
        "retrain_frequency_days": 30,
        "feature_importance_threshold": 0.05,
    }

    # Dashboard settings
    DASHBOARD_CONFIG = {
        "refresh_interval_minutes": 15,
        "alert_threshold": 10,  # Score change percentage
        "max_news_items": 50,
    }

    # Economic indicators from FRED
    ECONOMIC_INDICATORS = [
        "GDP",
        "UNRATE",  # Unemployment rate
        "FEDFUNDS",  # Federal funds rate
        "CPIAUCSL",  # Consumer Price Index
        "DEXUSEU",  # USD/EUR exchange rate
    ]
