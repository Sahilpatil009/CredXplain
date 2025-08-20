import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from fredapi import Fred
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.utils.config import Config
from src.utils.helpers import get_date_range, clean_text, handle_missing_data

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data ingestion from structured and unstructured sources"""

    def __init__(self):
        self.config = Config()
        self.fred = (
            Fred(api_key=self.config.FRED_API_KEY)
            if self.config.FRED_API_KEY != "your_fred_api_key_here"
            else None
        )

    def fetch_yahoo_finance_data(self, symbol: str, period: str = "1y") -> Dict:
        """Fetch financial data from Yahoo Finance"""
        try:
            logger.info(f"Fetching Yahoo Finance data for {symbol}")
            ticker = yf.Ticker(symbol)

            # Get financial statements
            balance_sheet = ticker.balance_sheet
            income_stmt = ticker.financials
            cash_flow = ticker.cashflow

            # Get stock price data
            hist = ticker.history(period=period)

            # Get key statistics
            info = ticker.info

            return {
                "symbol": symbol,
                "balance_sheet": (
                    balance_sheet.to_dict() if not balance_sheet.empty else {}
                ),
                "income_statement": (
                    income_stmt.to_dict() if not income_stmt.empty else {}
                ),
                "cash_flow": cash_flow.to_dict() if not cash_flow.empty else {},
                "price_history": hist.to_dict(),
                "info": info,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return {}

    def fetch_economic_indicators(self, days_back: int = 365) -> Dict:
        """Fetch economic indicators from FRED"""
        if not self.fred:
            logger.warning("FRED API key not configured, using mock data")
            return self._get_mock_economic_data()

        try:
            logger.info("Fetching economic indicators from FRED")
            start_date, end_date = get_date_range(days_back)

            indicators = {}
            for indicator in self.config.ECONOMIC_INDICATORS:
                try:
                    data = self.fred.get_series(
                        indicator, start=start_date, end=end_date
                    )
                    indicators[indicator] = data.to_dict()
                except Exception as e:
                    logger.warning(f"Could not fetch {indicator}: {e}")
                    continue

            return {"indicators": indicators, "timestamp": datetime.now()}
        except Exception as e:
            logger.error(f"Error fetching economic indicators: {e}")
            return self._get_mock_economic_data()

    def _get_mock_economic_data(self) -> Dict:
        """Generate mock economic data when FRED is not available"""
        logger.info("Generating mock economic data")
        dates = pd.date_range(end=datetime.now(), periods=365, freq="D")

        mock_data = {
            "indicators": {
                "GDP": {
                    date: 20000 + np.random.normal(0, 500) for date in dates[::30]
                },  # Monthly
                "UNRATE": {
                    date: 4.0 + np.random.normal(0, 0.5) for date in dates[::30]
                },
                "FEDFUNDS": {
                    date: 2.5 + np.random.normal(0, 0.3) for date in dates[::30]
                },
                "CPIAUCSL": {
                    date: 250 + np.random.normal(0, 5) for date in dates[::30]
                },
                "DEXUSEU": {date: 1.1 + np.random.normal(0, 0.05) for date in dates},
            },
            "timestamp": datetime.now(),
        }
        return mock_data

    def fetch_news_data(self, company_name: str, max_articles: int = 50) -> List[Dict]:
        """Fetch news articles from RSS feeds"""
        try:
            logger.info(f"Fetching news data for {company_name}")
            articles = []

            for source_name, rss_url in self.config.NEWS_SOURCES.items():
                try:
                    # For demo purposes, we'll create mock news data
                    # In production, you would parse actual RSS feeds
                    mock_articles = self._generate_mock_news(
                        company_name, source_name, 10
                    )
                    articles.extend(mock_articles)

                    if len(articles) >= max_articles:
                        break

                except Exception as e:
                    logger.warning(f"Error fetching from {source_name}: {e}")
                    continue

            return articles[:max_articles]

        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return []

    def _generate_mock_news(
        self, company_name: str, source: str, count: int
    ) -> List[Dict]:
        """Generate mock news articles for demonstration"""
        import random

        # Sample headlines with varying sentiment
        positive_templates = [
            f"{company_name} reports strong quarterly earnings",
            f"{company_name} announces new strategic partnership",
            f"{company_name} stock reaches new highs amid positive outlook",
            f"{company_name} expands market presence with new product launch",
        ]

        negative_templates = [
            f"{company_name} faces regulatory challenges",
            f"{company_name} reports declining revenues",
            f"Analysts downgrade {company_name} stock rating",
            f"{company_name} CEO resignation sparks investor concerns",
        ]

        neutral_templates = [
            f"{company_name} announces quarterly board meeting",
            f"{company_name} updates corporate governance policies",
            f"{company_name} releases sustainability report",
            f"{company_name} schedules investor conference call",
        ]

        all_templates = positive_templates + negative_templates + neutral_templates
        articles = []

        for i in range(count):
            headline = random.choice(all_templates)
            pub_date = datetime.now() - timedelta(days=random.randint(0, 30))

            articles.append(
                {
                    "headline": headline,
                    "source": source,
                    "published_date": pub_date,
                    "url": f"https://{source}.com/article/{i}",
                    "summary": f"Article about {company_name} from {source}. {headline}.",
                }
            )

        return articles

    def fetch_all_company_data(self, symbol: str) -> Dict:
        """Fetch all available data for a company"""
        logger.info(f"Fetching all data for {symbol}")

        # Get company info to extract name
        ticker = yf.Ticker(symbol)
        info = ticker.info
        company_name = info.get("longName", symbol)

        return {
            "financial_data": self.fetch_yahoo_finance_data(symbol),
            "news_data": self.fetch_news_data(company_name),
            "economic_data": self.fetch_economic_indicators(),
            "timestamp": datetime.now(),
        }

    def batch_fetch_companies(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch data for multiple companies"""
        logger.info(f"Batch fetching data for {len(symbols)} companies")

        results = {}
        economic_data = self.fetch_economic_indicators()  # Fetch once for all companies

        for symbol in symbols:
            try:
                logger.info(f"Processing {symbol}")
                company_data = self.fetch_all_company_data(symbol)
                company_data["economic_data"] = economic_data  # Share economic data
                results[symbol] = company_data
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = {"error": str(e)}

        return results
