# Technical indicators and trend analysis
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.utils.helpers import calculate_percentage_change, calculate_volatility

logger = logging.getLogger(__name__)


class TrendIndicatorCalculator:
    """Calculate technical indicators and trend analysis"""

    def __init__(self):
        pass

    def calculate_price_indicators(self, price_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price-based technical indicators"""
        try:
            if price_df.empty or "Close" not in price_df.columns:
                return self._get_empty_price_indicators()

            logger.info("Calculating price indicators")

            close_prices = price_df["Close"].dropna()
            if len(close_prices) < 20:
                return self._get_empty_price_indicators()

            indicators = {}

            # Current price metrics
            current_price = close_prices.iloc[-1]
            indicators["current_price"] = current_price

            # Moving averages
            indicators["sma_20"] = close_prices.rolling(window=20).mean().iloc[-1]
            indicators["sma_50"] = (
                close_prices.rolling(window=50).mean().iloc[-1]
                if len(close_prices) >= 50
                else current_price
            )
            indicators["sma_200"] = (
                close_prices.rolling(window=200).mean().iloc[-1]
                if len(close_prices) >= 200
                else current_price
            )

            # Price relative to moving averages
            indicators["price_vs_sma20"] = (
                current_price - indicators["sma_20"]
            ) / indicators["sma_20"]
            indicators["price_vs_sma50"] = (
                current_price - indicators["sma_50"]
            ) / indicators["sma_50"]
            indicators["price_vs_sma200"] = (
                current_price - indicators["sma_200"]
            ) / indicators["sma_200"]

            # Price performance
            indicators["return_1d"] = close_prices.pct_change().iloc[-1]
            indicators["return_5d"] = (
                (current_price - close_prices.iloc[-6]) / close_prices.iloc[-6]
                if len(close_prices) >= 6
                else 0
            )
            indicators["return_20d"] = (
                (current_price - close_prices.iloc[-21]) / close_prices.iloc[-21]
                if len(close_prices) >= 21
                else 0
            )
            indicators["return_60d"] = (
                (current_price - close_prices.iloc[-61]) / close_prices.iloc[-61]
                if len(close_prices) >= 61
                else 0
            )

            # Volatility measures
            returns = close_prices.pct_change().dropna()
            indicators["volatility_20d"] = returns.rolling(window=20).std().iloc[
                -1
            ] * np.sqrt(252)
            indicators["volatility_60d"] = (
                returns.rolling(window=60).std().iloc[-1] * np.sqrt(252)
                if len(returns) >= 60
                else indicators["volatility_20d"]
            )

            # Price extremes
            high_52w = (
                close_prices.rolling(window=252).max().iloc[-1]
                if len(close_prices) >= 252
                else close_prices.max()
            )
            low_52w = (
                close_prices.rolling(window=252).min().iloc[-1]
                if len(close_prices) >= 252
                else close_prices.min()
            )

            indicators["price_from_52w_high"] = (current_price - high_52w) / high_52w
            indicators["price_from_52w_low"] = (current_price - low_52w) / low_52w

            # RSI (Relative Strength Index)
            indicators["rsi_14"] = self._calculate_rsi(close_prices, 14)

            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(close_prices, 20, 2)
            indicators["bb_position"] = (
                (current_price - bb_lower) / (bb_upper - bb_lower)
                if bb_upper != bb_lower
                else 0.5
            )

            return indicators

        except Exception as e:
            logger.error(f"Error calculating price indicators: {e}")
            return self._get_empty_price_indicators()

    def _get_empty_price_indicators(self) -> Dict[str, float]:
        """Return empty/default price indicators"""
        return {
            "current_price": 0,
            "sma_20": 0,
            "sma_50": 0,
            "sma_200": 0,
            "price_vs_sma20": 0,
            "price_vs_sma50": 0,
            "price_vs_sma200": 0,
            "return_1d": 0,
            "return_5d": 0,
            "return_20d": 0,
            "return_60d": 0,
            "volatility_20d": 0,
            "volatility_60d": 0,
            "price_from_52w_high": 0,
            "price_from_52w_low": 0,
            "rsi_14": 50,
            "bb_position": 0.5,
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI

            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

        except Exception:
            return 50.0

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                return prices.iloc[-1], prices.iloc[-1]

            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            return upper_band.iloc[-1], lower_band.iloc[-1]

        except Exception:
            current = prices.iloc[-1] if not prices.empty else 0
            return current, current

    def calculate_volume_indicators(self, price_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based indicators"""
        try:
            if price_df.empty or "Volume" not in price_df.columns:
                return {
                    "avg_volume_20d": 0,
                    "volume_ratio": 1,
                    "volume_trend": 0,
                    "volume_volatility": 0,
                }

            logger.info("Calculating volume indicators")

            volume = price_df["Volume"].dropna()
            if len(volume) < 20:
                return {
                    "avg_volume_20d": volume.mean() if not volume.empty else 0,
                    "volume_ratio": 1,
                    "volume_trend": 0,
                    "volume_volatility": 0,
                }

            indicators = {}

            # Average volume
            indicators["avg_volume_20d"] = volume.rolling(window=20).mean().iloc[-1]

            # Current volume vs average
            current_volume = volume.iloc[-1]
            indicators["volume_ratio"] = (
                current_volume / indicators["avg_volume_20d"]
                if indicators["avg_volume_20d"] > 0
                else 1
            )

            # Volume trend (comparing recent to historical)
            recent_avg = volume.tail(5).mean()
            historical_avg = (
                volume.iloc[-20:-5].mean() if len(volume) >= 20 else volume.mean()
            )
            indicators["volume_trend"] = (
                (recent_avg - historical_avg) / historical_avg
                if historical_avg > 0
                else 0
            )

            # Volume volatility
            volume_changes = volume.pct_change().dropna()
            indicators["volume_volatility"] = (
                volume_changes.std() if len(volume_changes) > 1 else 0
            )

            return indicators

        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {
                "avg_volume_20d": 0,
                "volume_ratio": 1,
                "volume_trend": 0,
                "volume_volatility": 0,
            }

    def calculate_trend_strength(self, price_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend strength indicators"""
        try:
            if price_df.empty or "Close" not in price_df.columns:
                return {
                    "short_term_trend": 0,
                    "medium_term_trend": 0,
                    "long_term_trend": 0,
                    "trend_consistency": 0,
                    "momentum_score": 0,
                }

            logger.info("Calculating trend strength")

            close_prices = price_df["Close"].dropna()
            if len(close_prices) < 20:
                return {
                    "short_term_trend": 0,
                    "medium_term_trend": 0,
                    "long_term_trend": 0,
                    "trend_consistency": 0,
                    "momentum_score": 0,
                }

            indicators = {}

            # Short-term trend (5-day slope)
            if len(close_prices) >= 5:
                short_slope = self._calculate_price_slope(close_prices.tail(5))
                indicators["short_term_trend"] = short_slope
            else:
                indicators["short_term_trend"] = 0

            # Medium-term trend (20-day slope)
            if len(close_prices) >= 20:
                medium_slope = self._calculate_price_slope(close_prices.tail(20))
                indicators["medium_term_trend"] = medium_slope
            else:
                indicators["medium_term_trend"] = 0

            # Long-term trend (60-day slope)
            if len(close_prices) >= 60:
                long_slope = self._calculate_price_slope(close_prices.tail(60))
                indicators["long_term_trend"] = long_slope
            else:
                indicators["long_term_trend"] = indicators["medium_term_trend"]

            # Trend consistency (how often price moves in same direction)
            returns = close_prices.pct_change().dropna()
            if len(returns) >= 20:
                positive_days = (returns.tail(20) > 0).sum()
                indicators["trend_consistency"] = (
                    abs(positive_days - 10) / 10
                )  # Distance from 50%
            else:
                indicators["trend_consistency"] = 0

            # Momentum score (combination of trends and RSI)
            momentum_score = (
                indicators["short_term_trend"] * 0.5
                + indicators["medium_term_trend"] * 0.3
                + indicators["long_term_trend"] * 0.2
            )
            indicators["momentum_score"] = momentum_score

            return indicators

        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return {
                "short_term_trend": 0,
                "medium_term_trend": 0,
                "long_term_trend": 0,
                "trend_consistency": 0,
                "momentum_score": 0,
            }

    def _calculate_price_slope(self, prices: pd.Series) -> float:
        """Calculate slope of price trend"""
        try:
            if len(prices) < 2:
                return 0

            x = np.arange(len(prices))
            slope = np.polyfit(x, prices.values, 1)[0]

            # Normalize slope by average price
            avg_price = prices.mean()
            normalized_slope = slope / avg_price if avg_price > 0 else 0

            return normalized_slope

        except Exception:
            return 0

    def calculate_risk_indicators(
        self, price_df: pd.DataFrame, financial_ratios: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate risk-related indicators"""
        try:
            logger.info("Calculating risk indicators")

            risk_indicators = {}

            # Price-based risk
            if not price_df.empty and "Close" in price_df.columns:
                close_prices = price_df["Close"].dropna()

                if len(close_prices) >= 20:
                    # Value at Risk (VaR) - 5% percentile of daily returns
                    returns = close_prices.pct_change().dropna()
                    var_5 = returns.quantile(0.05) if len(returns) > 20 else 0
                    risk_indicators["var_5_percent"] = var_5

                    # Maximum drawdown
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    risk_indicators["max_drawdown"] = drawdown.min()

                    # Downside volatility
                    negative_returns = returns[returns < 0]
                    risk_indicators["downside_volatility"] = (
                        negative_returns.std() * np.sqrt(252)
                        if len(negative_returns) > 0
                        else 0
                    )
                else:
                    risk_indicators.update(
                        {
                            "var_5_percent": 0,
                            "max_drawdown": 0,
                            "downside_volatility": 0,
                        }
                    )
            else:
                risk_indicators.update(
                    {"var_5_percent": 0, "max_drawdown": 0, "downside_volatility": 0}
                )

            # Financial risk (from ratios)
            debt_to_equity = financial_ratios.get("debt_to_equity", 0)
            current_ratio = financial_ratios.get("current_ratio", 1)
            interest_coverage = financial_ratios.get("interest_coverage", 1)

            # Leverage risk score
            risk_indicators["leverage_risk"] = min(
                debt_to_equity / 2, 1
            )  # Normalize to 0-1

            # Liquidity risk score
            risk_indicators["liquidity_risk"] = max(
                0, 1 - current_ratio / 2
            )  # Higher when current ratio is low

            # Solvency risk score
            risk_indicators["solvency_risk"] = max(
                0, 1 - interest_coverage / 5
            )  # Higher when coverage is low

            # Overall financial risk (composite)
            financial_risk = (
                risk_indicators["leverage_risk"] * 0.4
                + risk_indicators["liquidity_risk"] * 0.3
                + risk_indicators["solvency_risk"] * 0.3
            )
            risk_indicators["financial_risk_score"] = financial_risk

            return risk_indicators

        except Exception as e:
            logger.error(f"Error calculating risk indicators: {e}")
            return {
                "var_5_percent": 0,
                "max_drawdown": 0,
                "downside_volatility": 0,
                "leverage_risk": 0,
                "liquidity_risk": 0,
                "solvency_risk": 0,
                "financial_risk_score": 0,
            }

    def calculate_all_indicators(
        self, price_df: pd.DataFrame, financial_ratios: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate all technical indicators"""
        try:
            logger.info("Calculating all technical indicators")

            all_indicators = {}

            # Price indicators
            all_indicators.update(self.calculate_price_indicators(price_df))

            # Volume indicators
            all_indicators.update(self.calculate_volume_indicators(price_df))

            # Trend strength
            all_indicators.update(self.calculate_trend_strength(price_df))

            # Risk indicators
            all_indicators.update(
                self.calculate_risk_indicators(price_df, financial_ratios)
            )

            return all_indicators

        except Exception as e:
            logger.error(f"Error calculating all indicators: {e}")
            return {}
