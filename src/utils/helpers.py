# Utility helper functions
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_score(
    value: float,
    min_val: float,
    max_val: float,
    target_min: float = 0,
    target_max: float = 100,
) -> float:
    """Normalize a value to a target range (default 0-100)"""
    if max_val == min_val:
        return target_min

    normalized = (value - min_val) / (max_val - min_val)
    return target_min + normalized * (target_max - target_min)


def calculate_percentage_change(current: float, previous: float) -> float:
    """Calculate percentage change between two values"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Safely divide two numbers, return default if division by zero"""
    if denominator == 0:
        return default
    return numerator / denominator


def get_date_range(days_back: int = 365) -> tuple:
    """Get start and end dates for data fetching"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date


def clean_text(text: str) -> str:
    """Clean text for sentiment analysis"""
    import re

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove special characters but keep spaces and punctuation for sentiment
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()


def handle_missing_data(df: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
    """Handle missing data in DataFrame"""
    if method == "forward_fill":
        return df.fillna(method="ffill").fillna(method="bfill")
    elif method == "interpolate":
        return df.interpolate()
    elif method == "drop":
        return df.dropna()
    else:
        return df.fillna(0)


def validate_financial_data(data: Dict) -> bool:
    """Validate financial data completeness"""
    required_fields = [
        "revenue",
        "total_debt",
        "total_equity",
        "current_assets",
        "current_liabilities",
    ]
    return all(field in data and data[field] is not None for field in required_fields)


def format_currency(amount: float) -> str:
    """Format currency amounts for display"""
    if abs(amount) >= 1e9:
        return f"${amount/1e9:.2f}B"
    elif abs(amount) >= 1e6:
        return f"${amount/1e6:.2f}M"
    elif abs(amount) >= 1e3:
        return f"${amount/1e3:.2f}K"
    else:
        return f"${amount:.2f}"


def get_risk_level(score: float) -> str:
    """Convert numerical score to risk level"""
    if score >= 80:
        return "Low Risk"
    elif score >= 60:
        return "Medium Risk"
    elif score >= 40:
        return "High Risk"
    else:
        return "Very High Risk"


def get_rating_scale(score: float) -> str:
    """Convert score to traditional credit rating scale"""
    if score >= 90:
        return "AAA"
    elif score >= 85:
        return "AA"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "BBB"
    elif score >= 60:
        return "BB"
    elif score >= 50:
        return "B"
    elif score >= 40:
        return "CCC"
    else:
        return "D"


def create_feature_summary(
    features: Dict[str, float], importance: Dict[str, float]
) -> str:
    """Create human-readable feature importance summary"""
    sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)

    summary_parts = []
    for feature, imp in sorted_features[:5]:  # Top 5 features
        impact = "increased" if imp > 0 else "decreased"
        value = features.get(feature, 0)
        summary_parts.append(
            f"{feature.replace('_', ' ').title()}: {value:.2f} ({impact} score)"
        )

    return "; ".join(summary_parts)


def detect_anomalies(data: pd.Series, threshold: float = 2.0) -> pd.Series:
    """Detect anomalies using z-score method"""
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold


def calculate_volatility(prices: pd.Series, window: int = 30) -> float:
    """Calculate price volatility"""
    returns = prices.pct_change().dropna()
    return returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)  # Annualized
