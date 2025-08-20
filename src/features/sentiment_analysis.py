# Sentiment analysis for news data
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from datetime import datetime, timedelta
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.utils.helpers import clean_text

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment from news headlines and articles"""

    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.financial_keywords = self._load_financial_keywords()

    def _load_financial_keywords(self) -> Dict[str, List[str]]:
        """Load financial-specific keywords for sentiment analysis"""
        return {
            "positive": [
                "profit",
                "earnings",
                "growth",
                "revenue",
                "strong",
                "beat",
                "exceed",
                "outperform",
                "bullish",
                "upgrade",
                "buy",
                "increased",
                "expansion",
                "acquisition",
                "merger",
                "partnership",
                "innovation",
                "breakthrough",
                "record",
                "high",
                "surge",
                "rally",
                "optimistic",
                "confident",
            ],
            "negative": [
                "loss",
                "decline",
                "fall",
                "drop",
                "weak",
                "miss",
                "disappoint",
                "underperform",
                "bearish",
                "downgrade",
                "sell",
                "decreased",
                "cut",
                "layoffs",
                "bankruptcy",
                "lawsuit",
                "scandal",
                "investigation",
                "low",
                "crash",
                "plunge",
                "pessimistic",
                "concerned",
                "warning",
            ],
            "neutral": [
                "announce",
                "report",
                "release",
                "update",
                "meeting",
                "conference",
                "statement",
                "presentation",
                "discussion",
                "review",
                "analysis",
            ],
        }

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text using VADER"""
        try:
            if not text or pd.isna(text):
                return {
                    "compound": 0.0,
                    "positive": 0.0,
                    "negative": 0.0,
                    "neutral": 1.0,
                    "sentiment_label": "neutral",
                }

            # Clean the text
            clean_text_str = clean_text(str(text))

            # Get VADER scores
            scores = self.vader_analyzer.polarity_scores(clean_text_str)

            # Apply financial keyword weights
            financial_score = self._calculate_financial_sentiment(clean_text_str)

            # Combine VADER and financial keyword scores
            compound_score = (scores["compound"] * 0.7) + (financial_score * 0.3)

            # Determine sentiment label
            if compound_score >= 0.05:
                sentiment_label = "positive"
            elif compound_score <= -0.05:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            return {
                "compound": compound_score,
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
                "sentiment_label": sentiment_label,
                "financial_score": financial_score,
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sentiment_label": "neutral",
                "financial_score": 0.0,
            }

    def _calculate_financial_sentiment(self, text: str) -> float:
        """Calculate sentiment based on financial keywords"""
        try:
            text_lower = text.lower()
            positive_count = 0
            negative_count = 0

            # Count positive keywords
            for keyword in self.financial_keywords["positive"]:
                positive_count += len(re.findall(r"\b" + keyword + r"\b", text_lower))

            # Count negative keywords
            for keyword in self.financial_keywords["negative"]:
                negative_count += len(re.findall(r"\b" + keyword + r"\b", text_lower))

            # Calculate financial sentiment score
            total_keywords = positive_count + negative_count
            if total_keywords == 0:
                return 0.0

            return (positive_count - negative_count) / total_keywords

        except Exception as e:
            logger.error(f"Error calculating financial sentiment: {e}")
            return 0.0

    def analyze_news_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment for all news articles in a DataFrame"""
        try:
            logger.info(f"Analyzing sentiment for {len(news_df)} news articles")

            if news_df.empty:
                return news_df

            # Analyze headlines
            if "headline" in news_df.columns:
                headline_sentiment = news_df["headline"].apply(
                    self.analyze_text_sentiment
                )

                # Extract sentiment components
                news_df["headline_sentiment_compound"] = headline_sentiment.apply(
                    lambda x: x["compound"]
                )
                news_df["headline_sentiment_label"] = headline_sentiment.apply(
                    lambda x: x["sentiment_label"]
                )
                news_df["headline_financial_score"] = headline_sentiment.apply(
                    lambda x: x.get("financial_score", 0)
                )

            # Analyze summaries if available
            if "summary" in news_df.columns:
                summary_sentiment = news_df["summary"].apply(
                    self.analyze_text_sentiment
                )

                news_df["summary_sentiment_compound"] = summary_sentiment.apply(
                    lambda x: x["compound"]
                )
                news_df["summary_sentiment_label"] = summary_sentiment.apply(
                    lambda x: x["sentiment_label"]
                )
                news_df["summary_financial_score"] = summary_sentiment.apply(
                    lambda x: x.get("financial_score", 0)
                )

                # Combined sentiment score (headline weighted more heavily)
                news_df["combined_sentiment"] = (
                    news_df["headline_sentiment_compound"] * 0.7
                    + news_df["summary_sentiment_compound"] * 0.3
                )
            else:
                news_df["combined_sentiment"] = news_df.get(
                    "headline_sentiment_compound", 0
                )

            # Overall sentiment label based on combined score
            news_df["sentiment_label"] = news_df["combined_sentiment"].apply(
                lambda x: (
                    "positive"
                    if x >= 0.05
                    else ("negative" if x <= -0.05 else "neutral")
                )
            )

            # Final sentiment score for the article
            news_df["sentiment_score"] = news_df["combined_sentiment"]

            return news_df

        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return news_df

    def calculate_sentiment_metrics(
        self, news_df: pd.DataFrame, days_back: int = 30
    ) -> Dict[str, float]:
        """Calculate aggregate sentiment metrics"""
        try:
            if news_df.empty:
                return {
                    "overall_sentiment": 0.0,
                    "positive_ratio": 0.0,
                    "negative_ratio": 0.0,
                    "neutral_ratio": 0.0,
                    "sentiment_volatility": 0.0,
                    "recent_sentiment_trend": 0.0,
                    "total_articles": 0,
                }

            # Filter recent articles
            if "published_date" in news_df.columns:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                recent_news = news_df[news_df["published_date"] >= cutoff_date]
            else:
                recent_news = news_df

            if recent_news.empty:
                recent_news = news_df  # Fall back to all news if no recent news

            sentiment_scores = recent_news.get("sentiment_score", pd.Series([0]))
            sentiment_labels = recent_news.get(
                "sentiment_label", pd.Series(["neutral"])
            )

            # Overall sentiment (mean of sentiment scores)
            overall_sentiment = sentiment_scores.mean()

            # Sentiment distribution
            label_counts = sentiment_labels.value_counts()
            total_articles = len(recent_news)

            positive_ratio = label_counts.get("positive", 0) / max(total_articles, 1)
            negative_ratio = label_counts.get("negative", 0) / max(total_articles, 1)
            neutral_ratio = label_counts.get("neutral", 0) / max(total_articles, 1)

            # Sentiment volatility (standard deviation of scores)
            sentiment_volatility = (
                sentiment_scores.std() if len(sentiment_scores) > 1 else 0.0
            )

            # Recent sentiment trend (last 7 days vs previous period)
            recent_sentiment_trend = 0.0
            if "published_date" in recent_news.columns and len(recent_news) > 5:
                try:
                    recent_7d = recent_news[
                        recent_news["published_date"]
                        >= datetime.now() - timedelta(days=7)
                    ]
                    previous_7d = recent_news[
                        (
                            recent_news["published_date"]
                            < datetime.now() - timedelta(days=7)
                        )
                        & (
                            recent_news["published_date"]
                            >= datetime.now() - timedelta(days=14)
                        )
                    ]

                    if not recent_7d.empty and not previous_7d.empty:
                        recent_avg = recent_7d["sentiment_score"].mean()
                        previous_avg = previous_7d["sentiment_score"].mean()
                        recent_sentiment_trend = recent_avg - previous_avg
                except:
                    pass

            return {
                "overall_sentiment": overall_sentiment,
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "neutral_ratio": neutral_ratio,
                "sentiment_volatility": sentiment_volatility,
                "recent_sentiment_trend": recent_sentiment_trend,
                "total_articles": total_articles,
                "sentiment_strength": abs(
                    overall_sentiment
                ),  # How strong the sentiment is
                "sentiment_consistency": 1
                - sentiment_volatility,  # How consistent the sentiment is
            }

        except Exception as e:
            logger.error(f"Error calculating sentiment metrics: {e}")
            return {
                "overall_sentiment": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "sentiment_volatility": 0.0,
                "recent_sentiment_trend": 0.0,
                "total_articles": 0,
                "sentiment_strength": 0.0,
                "sentiment_consistency": 0.0,
            }

    def get_sentiment_impact_score(self, sentiment_metrics: Dict[str, float]) -> float:
        """Calculate overall sentiment impact on credit score"""
        try:
            # Weight different sentiment factors
            overall_weight = 0.4
            trend_weight = 0.3
            consistency_weight = 0.2
            volume_weight = 0.1

            # Overall sentiment impact
            overall_impact = (
                sentiment_metrics.get("overall_sentiment", 0) * overall_weight
            )

            # Trend impact
            trend_impact = (
                sentiment_metrics.get("recent_sentiment_trend", 0) * trend_weight
            )

            # Consistency impact (consistent sentiment is better)
            consistency_impact = (
                sentiment_metrics.get("sentiment_consistency", 0) * consistency_weight
            )

            # Volume impact (more articles = more reliable sentiment)
            total_articles = sentiment_metrics.get("total_articles", 0)
            volume_impact = (
                min(total_articles / 20, 1.0) * volume_weight
            )  # Normalize to max 20 articles

            # Combine all impacts
            total_impact = (
                overall_impact + trend_impact + consistency_impact + volume_impact
            )

            # Normalize to -1 to 1 range
            return max(-1, min(1, total_impact))

        except Exception as e:
            logger.error(f"Error calculating sentiment impact: {e}")
            return 0.0
