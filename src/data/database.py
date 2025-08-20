# Database operations for storing and retrieving data
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.utils.config import Config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handles database operations for the credit scoring system"""

    def __init__(self):
        self.config = Config()
        self.engine = create_engine(self.config.DATABASE_URL)
        self.init_database()

    def init_database(self):
        """Initialize database with required tables"""
        try:
            logger.info("Initializing database")

            # Create companies table
            companies_sql = """
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT,
                sector TEXT,
                industry TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """

            # Create financial_data table
            financial_sql = """
            CREATE TABLE IF NOT EXISTS financial_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER,
                data_date DATE,
                market_cap REAL,
                enterprise_value REAL,
                pe_ratio REAL,
                pb_ratio REAL,
                debt_to_equity REAL,
                current_ratio REAL,
                quick_ratio REAL,
                roe REAL,
                roa REAL,
                profit_margin REAL,
                revenue_growth REAL,
                earnings_growth REAL,
                beta REAL,
                dividend_yield REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
            """

            # Create news_data table
            news_sql = """
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER,
                headline TEXT,
                summary TEXT,
                source TEXT,
                published_date TIMESTAMP,
                sentiment_score REAL,
                sentiment_label TEXT,
                url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
            """

            # Create economic_data table
            economic_sql = """
            CREATE TABLE IF NOT EXISTS economic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_name TEXT,
                data_date DATE,
                value REAL,
                pct_change REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """

            # Create credit_scores table
            scores_sql = """
            CREATE TABLE IF NOT EXISTS credit_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER,
                score REAL,
                rating TEXT,
                risk_level TEXT,
                model_version TEXT,
                feature_importance TEXT,  -- JSON string
                explanation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
            """

            # Create price_data table
            price_sql = """
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER,
                data_date DATE,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                daily_return REAL,
                volatility_20d REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
            """

            with self.engine.connect() as conn:
                conn.execute(text(companies_sql))
                conn.execute(text(financial_sql))
                conn.execute(text(news_sql))
                conn.execute(text(economic_sql))
                conn.execute(text(scores_sql))
                conn.execute(text(price_sql))
                conn.commit()

            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def insert_company(
        self, symbol: str, name: str = None, sector: str = None, industry: str = None
    ) -> int:
        """Insert or update company information"""
        try:
            with self.engine.connect() as conn:
                # Check if company exists
                check_sql = "SELECT id FROM companies WHERE symbol = :symbol"
                result = conn.execute(text(check_sql), {"symbol": symbol}).fetchone()

                if result:
                    # Update existing
                    company_id = result[0]
                    update_sql = """
                    UPDATE companies 
                    SET name = COALESCE(:name, name),
                        sector = COALESCE(:sector, sector),
                        industry = COALESCE(:industry, industry),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :company_id
                    """
                    conn.execute(
                        text(update_sql),
                        {
                            "name": name,
                            "sector": sector,
                            "industry": industry,
                            "company_id": company_id,
                        },
                    )
                else:
                    # Insert new
                    insert_sql = """
                    INSERT INTO companies (symbol, name, sector, industry)
                    VALUES (:symbol, :name, :sector, :industry)
                    """
                    result = conn.execute(
                        text(insert_sql),
                        {
                            "symbol": symbol,
                            "name": name,
                            "sector": sector,
                            "industry": industry,
                        },
                    )
                    company_id = result.lastrowid

                conn.commit()
                return company_id

        except Exception as e:
            logger.error(f"Error inserting company {symbol}: {e}")
            return None

    def insert_financial_data(
        self, company_id: int, financial_data: Dict, data_date: datetime = None
    ):
        """Insert financial data for a company"""
        try:
            if data_date is None:
                data_date = datetime.now().date()

            insert_sql = """
            INSERT OR REPLACE INTO financial_data (
                company_id, data_date, market_cap, enterprise_value, pe_ratio, pb_ratio,
                debt_to_equity, current_ratio, quick_ratio, roe, roa, profit_margin,
                revenue_growth, earnings_growth, beta, dividend_yield
            ) VALUES (
                :company_id, :data_date, :market_cap, :enterprise_value, :pe_ratio, :pb_ratio,
                :debt_to_equity, :current_ratio, :quick_ratio, :roe, :roa, :profit_margin,
                :revenue_growth, :earnings_growth, :beta, :dividend_yield
            )
            """

            with self.engine.connect() as conn:
                conn.execute(
                    text(insert_sql),
                    {
                        "company_id": company_id,
                        "data_date": data_date,
                        **financial_data,
                    },
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Error inserting financial data: {e}")

    def insert_news_data(self, company_id: int, news_df: pd.DataFrame):
        """Insert news data for a company"""
        try:
            if news_df.empty:
                return

            # Prepare data for insertion
            news_records = []
            for _, row in news_df.iterrows():
                record = {
                    "company_id": company_id,
                    "headline": row.get("headline", ""),
                    "summary": row.get("summary", ""),
                    "source": row.get("source", ""),
                    "published_date": row.get("published_date"),
                    "sentiment_score": row.get("sentiment_score"),
                    "sentiment_label": row.get("sentiment_label"),
                    "url": row.get("url", ""),
                }
                news_records.append(record)

            insert_sql = """
            INSERT OR IGNORE INTO news_data (
                company_id, headline, summary, source, published_date,
                sentiment_score, sentiment_label, url
            ) VALUES (
                :company_id, :headline, :summary, :source, :published_date,
                :sentiment_score, :sentiment_label, :url
            )
            """

            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), news_records)
                conn.commit()

        except Exception as e:
            logger.error(f"Error inserting news data: {e}")

    def insert_credit_score(
        self,
        company_id: int,
        score: float,
        rating: str,
        risk_level: str,
        feature_importance: Dict,
        explanation: str,
        model_version: str = "v1.0",
    ):
        """Insert credit score for a company"""
        try:
            insert_sql = """
            INSERT INTO credit_scores (
                company_id, score, rating, risk_level, model_version,
                feature_importance, explanation
            ) VALUES (
                :company_id, :score, :rating, :risk_level, :model_version,
                :feature_importance, :explanation
            )
            """

            with self.engine.connect() as conn:
                conn.execute(
                    text(insert_sql),
                    {
                        "company_id": company_id,
                        "score": score,
                        "rating": rating,
                        "risk_level": risk_level,
                        "model_version": model_version,
                        "feature_importance": json.dumps(feature_importance),
                        "explanation": explanation,
                    },
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Error inserting credit score: {e}")

    def get_company_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get company information by symbol"""
        try:
            sql = "SELECT * FROM companies WHERE symbol = :symbol"
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), {"symbol": symbol}).fetchone()
                if result:
                    return {
                        "id": result[0],
                        "symbol": result[1],
                        "name": result[2],
                        "sector": result[3],
                        "industry": result[4],
                    }
                return None

        except Exception as e:
            logger.error(f"Error getting company {symbol}: {e}")
            return None

    def get_latest_score(self, company_id: int) -> Optional[Dict]:
        """Get the latest credit score for a company"""
        try:
            sql = """
            SELECT score, rating, risk_level, feature_importance, explanation, created_at
            FROM credit_scores
            WHERE company_id = :company_id
            ORDER BY created_at DESC
            LIMIT 1
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(sql), {"company_id": company_id}).fetchone()
                if result:
                    return {
                        "score": result[0],
                        "rating": result[1],
                        "risk_level": result[2],
                        "feature_importance": (
                            json.loads(result[3]) if result[3] else {}
                        ),
                        "explanation": result[4],
                        "timestamp": result[5],
                    }
                return None

        except Exception as e:
            logger.error(f"Error getting latest score: {e}")
            return None

    def get_score_history(self, company_id: int, days: int = 30) -> pd.DataFrame:
        """Get credit score history for a company"""
        try:
            sql = """
            SELECT score, rating, created_at
            FROM credit_scores
            WHERE company_id = :company_id
            AND created_at >= date('now', '-{} days')
            ORDER BY created_at
            """.format(
                days
            )

            return pd.read_sql(sql, self.engine, params={"company_id": company_id})

        except Exception as e:
            logger.error(f"Error getting score history: {e}")
            return pd.DataFrame()

    def get_all_companies(self) -> pd.DataFrame:
        """Get all companies in the database"""
        try:
            sql = "SELECT * FROM companies ORDER BY symbol"
            return pd.read_sql(sql, self.engine)

        except Exception as e:
            logger.error(f"Error getting all companies: {e}")
            return pd.DataFrame()
