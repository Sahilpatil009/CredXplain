# Data preprocessing and cleaning
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.utils.helpers import handle_missing_data, validate_financial_data, clean_text

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data cleaning, normalization, and preprocessing"""

    def __init__(self):
        pass

    def clean_financial_data(self, financial_data: Dict) -> Dict:
        """Clean and normalize financial data from Yahoo Finance"""
        try:
            logger.info("Cleaning financial data")

            cleaned_data = {
                "symbol": financial_data.get("symbol", ""),
                "timestamp": financial_data.get("timestamp", datetime.now()),
            }

            # Process balance sheet data
            balance_sheet = financial_data.get("balance_sheet", {})
            if balance_sheet:
                cleaned_data["balance_sheet"] = self._process_financial_statement(
                    balance_sheet
                )

            # Process income statement data
            income_stmt = financial_data.get("income_statement", {})
            if income_stmt:
                cleaned_data["income_statement"] = self._process_financial_statement(
                    income_stmt
                )

            # Process cash flow data
            cash_flow = financial_data.get("cash_flow", {})
            if cash_flow:
                cleaned_data["cash_flow"] = self._process_financial_statement(cash_flow)

            # Process price history
            price_history = financial_data.get("price_history", {})
            if price_history:
                cleaned_data["price_history"] = self._process_price_data(price_history)

            # Process company info
            info = financial_data.get("info", {})
            if info:
                cleaned_data["company_info"] = self._process_company_info(info)

            return cleaned_data

        except Exception as e:
            logger.error(f"Error cleaning financial data: {e}")
            return {}

    def _process_financial_statement(self, statement_data: Dict) -> pd.DataFrame:
        """Process financial statement data into clean DataFrame"""
        try:
            if not statement_data:
                return pd.DataFrame()

            df = pd.DataFrame(statement_data)

            # Handle missing values
            df = handle_missing_data(df, method="forward_fill")

            # Convert to numeric where possible
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Sort by date (most recent first)
            df = df.sort_index(axis=1, ascending=False)

            return df

        except Exception as e:
            logger.error(f"Error processing financial statement: {e}")
            return pd.DataFrame()

    def _process_price_data(self, price_data: Dict) -> pd.DataFrame:
        """Process stock price data"""
        try:
            df = pd.DataFrame(price_data)

            if df.empty:
                return df

            # Ensure numeric columns
            numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Handle missing values
            df = handle_missing_data(df, method="interpolate")

            # Calculate additional metrics
            if "Close" in df.columns:
                df["Daily_Return"] = df["Close"].pct_change()
                df["Volatility_20d"] = df["Daily_Return"].rolling(window=20).std()
                df["SMA_50"] = df["Close"].rolling(window=50).mean()
                df["SMA_200"] = df["Close"].rolling(window=200).mean()

            return df

        except Exception as e:
            logger.error(f"Error processing price data: {e}")
            return pd.DataFrame()

    def _process_company_info(self, info_data: Dict) -> Dict:
        """Process company information data"""
        try:
            # Extract key metrics with fallback values
            processed_info = {
                "market_cap": info_data.get("marketCap", 0),
                "enterprise_value": info_data.get("enterpriseValue", 0),
                "pe_ratio": info_data.get("trailingPE", 0),
                "pb_ratio": info_data.get("priceToBook", 0),
                "debt_to_equity": info_data.get("debtToEquity", 0),
                "current_ratio": info_data.get("currentRatio", 0),
                "roe": info_data.get("returnOnEquity", 0),
                "roa": info_data.get("returnOnAssets", 0),
                "profit_margin": info_data.get("profitMargins", 0),
                "revenue_growth": info_data.get("revenueGrowth", 0),
                "earnings_growth": info_data.get("earningsGrowth", 0),
                "beta": info_data.get("beta", 1.0),
                "dividend_yield": info_data.get("dividendYield", 0),
                "sector": info_data.get("sector", "Unknown"),
                "industry": info_data.get("industry", "Unknown"),
                "country": info_data.get("country", "Unknown"),
                "full_time_employees": info_data.get("fullTimeEmployees", 0),
            }

            # Convert None values to 0 or appropriate defaults
            for key, value in processed_info.items():
                if value is None:
                    if key in ["sector", "industry", "country"]:
                        processed_info[key] = "Unknown"
                    else:
                        processed_info[key] = 0

            return processed_info

        except Exception as e:
            logger.error(f"Error processing company info: {e}")
            return {}

    def clean_news_data(self, news_data: List[Dict]) -> pd.DataFrame:
        """Clean and process news articles data"""
        try:
            logger.info(f"Cleaning {len(news_data)} news articles")

            if not news_data:
                return pd.DataFrame()

            df = pd.DataFrame(news_data)

            # Clean text fields
            if "headline" in df.columns:
                df["headline_clean"] = df["headline"].apply(
                    lambda x: clean_text(str(x))
                )

            if "summary" in df.columns:
                df["summary_clean"] = df["summary"].apply(lambda x: clean_text(str(x)))

            # Ensure datetime format for published_date
            if "published_date" in df.columns:
                df["published_date"] = pd.to_datetime(
                    df["published_date"], errors="coerce"
                )

            # Remove duplicates based on headline
            df = df.drop_duplicates(subset=["headline"], keep="first")

            # Sort by publication date (most recent first)
            if "published_date" in df.columns:
                df = df.sort_values("published_date", ascending=False)

            return df

        except Exception as e:
            logger.error(f"Error cleaning news data: {e}")
            return pd.DataFrame()

    def clean_economic_data(self, economic_data: Dict) -> pd.DataFrame:
        """Clean and process economic indicators data"""
        try:
            logger.info("Cleaning economic data")

            indicators = economic_data.get("indicators", {})
            if not indicators:
                return pd.DataFrame()

            # Convert to DataFrame
            dfs = []
            for indicator, data in indicators.items():
                if data:
                    series = pd.Series(data, name=indicator)
                    series.index = pd.to_datetime(series.index)
                    dfs.append(series)

            if not dfs:
                return pd.DataFrame()

            df = pd.concat(dfs, axis=1, sort=True)

            # Handle missing values
            df = handle_missing_data(df, method="interpolate")

            # Calculate percentage changes
            for col in df.columns:
                df[f"{col}_pct_change"] = df[col].pct_change()

            return df

        except Exception as e:
            logger.error(f"Error cleaning economic data: {e}")
            return pd.DataFrame()

    def normalize_all_data(self, raw_data: Dict) -> Dict:
        """Main function to clean and normalize all data types"""
        try:
            logger.info("Normalizing all data")

            normalized_data = {
                "symbol": raw_data.get("financial_data", {}).get("symbol", ""),
                "timestamp": raw_data.get("timestamp", datetime.now()),
            }

            # Clean financial data
            if "financial_data" in raw_data:
                normalized_data["financial"] = self.clean_financial_data(
                    raw_data["financial_data"]
                )

            # Clean news data
            if "news_data" in raw_data:
                normalized_data["news"] = self.clean_news_data(raw_data["news_data"])

            # Clean economic data
            if "economic_data" in raw_data:
                normalized_data["economic"] = self.clean_economic_data(
                    raw_data["economic_data"]
                )

            return normalized_data

        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return {}

    def create_feature_matrix(self, normalized_data: Dict) -> pd.DataFrame:
        """Create a feature matrix ready for machine learning"""
        try:
            logger.info("Creating feature matrix")

            features = {}

            # Extract financial features
            if "financial" in normalized_data:
                financial = normalized_data["financial"]
                if "company_info" in financial:
                    info = financial["company_info"]
                    for key, value in info.items():
                        if isinstance(value, (int, float)):
                            features[f"financial_{key}"] = value

            # Add economic features (latest values)
            if "economic" in normalized_data:
                economic_df = normalized_data["economic"]
                if not economic_df.empty:
                    latest_economic = economic_df.iloc[-1]
                    for col in economic_df.columns:
                        if not col.endswith("_pct_change"):
                            features[f"economic_{col}"] = latest_economic[col]

            # Convert to DataFrame
            if features:
                feature_df = pd.DataFrame([features])
                feature_df = feature_df.fillna(0)  # Fill remaining NaN values
                return feature_df
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error creating feature matrix: {e}")
            return pd.DataFrame()
