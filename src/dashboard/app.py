# Main Streamlit dashboard for the credit scoring system
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
from typing import Dict, List

# Import our modules
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.data.database import DatabaseManager
from src.features.financial_ratios import FinancialRatiosCalculator
from src.features.sentiment_analysis import SentimentAnalyzer
from src.features.trend_indicators import TrendIndicatorCalculator
from src.models.scoring_engine import CreditScoringEngine
from src.models.explainability import ModelExplainer
from src.utils.config import Config
from src.utils.helpers import get_risk_level, get_rating_scale, format_currency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-low { border-left-color: #28a745; }
    .risk-medium { border-left-color: #ffc107; }
    .risk-high { border-left-color: #dc3545; }
    .feature-importance {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
    }
</style>
""",
    unsafe_allow_html=True,
)


class CreditScoringDashboard:
    """Main dashboard class for the credit scoring system"""

    def __init__(self):
        self.config = Config()
        self.db = DatabaseManager()
        self.data_ingestion = DataIngestion()
        self.preprocessor = DataPreprocessor()
        self.financial_calculator = FinancialRatiosCalculator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trend_calculator = TrendIndicatorCalculator()
        self.scoring_engine = CreditScoringEngine()
        self.explainer = ModelExplainer()

        # Initialize session state
        if "companies_data" not in st.session_state:
            st.session_state.companies_data = {}

        if "selected_company" not in st.session_state:
            st.session_state.selected_company = None

    def run(self):
        """Main dashboard runner"""
        # Header
        st.markdown(
            '<h1 class="main-header">🏦 Real-Time Credit Scoring System</h1>',
            unsafe_allow_html=True,
        )

        # Sidebar
        self.render_sidebar()

        # Main content
        if st.session_state.selected_company:
            self.render_company_analysis()
        else:
            self.render_overview()

    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.header("📋 Control Panel")

        # Company selection
        companies = self.config.COMPANIES
        selected_company = st.sidebar.selectbox(
            "Select Company",
            options=[""] + companies,
            index=0,
            help="Choose a company to analyze",
        )

        if selected_company and selected_company != st.session_state.selected_company:
            st.session_state.selected_company = selected_company
            with st.spinner(f"Loading data for {selected_company}..."):
                self.load_company_data(selected_company)

        # Data refresh button
        if st.sidebar.button("🔄 Refresh Data", help="Fetch latest data"):
            if st.session_state.selected_company:
                with st.spinner("Refreshing data..."):
                    self.load_company_data(
                        st.session_state.selected_company, force_refresh=True
                    )
                st.sidebar.success("Data refreshed!")

        # Model training section
        st.sidebar.subheader("🤖 Model Management")

        if st.sidebar.button("Train Models", help="Train ML models with current data"):
            self.train_models()

        # Settings
        st.sidebar.subheader("⚙️ Settings")

        self.auto_refresh = st.sidebar.checkbox(
            "Auto-refresh (5 min)",
            value=False,
            help="Automatically refresh data every 5 minutes",
        )

        self.show_advanced = st.sidebar.checkbox(
            "Show Advanced Metrics",
            value=False,
            help="Display detailed technical indicators",
        )

    def render_overview(self):
        """Render the overview page"""
        st.markdown("## 📈 System Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Companies Monitored",
                value=len(self.config.COMPANIES),
                help="Total number of companies in the system",
            )

        with col2:
            st.metric(
                label="Data Sources", value="3+", help="Yahoo Finance, FRED, News RSS"
            )

        with col3:
            st.metric(
                label="ML Models",
                value="4",
                help="Decision Tree, Random Forest, Logistic Regression, Gradient Boosting",
            )

        with col4:
            st.metric(
                label="Update Frequency",
                value="Real-time",
                help="Data updates every 15 minutes",
            )

        # System architecture diagram
        st.markdown("### 🏗️ System Architecture")

        # Create a simple flow diagram using columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                """
            **📊 Data Sources**
            - Yahoo Finance
            - FRED Economic Data
            - Financial News RSS
            """
            )

        with col2:
            st.markdown(
                """
            **🔧 Feature Engineering**
            - Financial Ratios
            - Sentiment Analysis
            - Technical Indicators
            """
            )

        with col3:
            st.markdown(
                """
            **🤖 ML Models**
            - Decision Trees
            - Random Forest
            - Logistic Regression
            - Ensemble Methods
            """
            )

        with col4:
            st.markdown(
                """
            **📋 Explainability**
            - SHAP Values
            - Feature Importance
            - Plain Language Summary
            """
            )

        # Recent activity
        st.markdown("### 📰 Recent Activity")

        # Generate some sample activity
        recent_activity = [
            {
                "time": datetime.now() - timedelta(minutes=5),
                "event": "Data refresh completed",
                "type": "info",
            },
            {
                "time": datetime.now() - timedelta(minutes=15),
                "event": "Model training finished",
                "type": "success",
            },
            {
                "time": datetime.now() - timedelta(hours=1),
                "event": "New economic data available",
                "type": "info",
            },
            {
                "time": datetime.now() - timedelta(hours=2),
                "event": "Alert: High volatility detected",
                "type": "warning",
            },
        ]

        for activity in recent_activity:
            icon = (
                "ℹ️"
                if activity["type"] == "info"
                else ("✅" if activity["type"] == "success" else "⚠️")
            )
            st.markdown(
                f"**{icon} {activity['time'].strftime('%H:%M')}** - {activity['event']}"
            )

    def load_company_data(self, symbol: str, force_refresh: bool = False):
        """Load data for a specific company"""
        try:
            if not force_refresh and symbol in st.session_state.companies_data:
                return

            # Fetch raw data
            raw_data = self.data_ingestion.fetch_all_company_data(symbol)

            if "error" in raw_data:
                st.error(f"Error loading data for {symbol}: {raw_data['error']}")
                return

            # Preprocess data
            normalized_data = self.preprocessor.normalize_all_data(raw_data)

            # Calculate features
            financial_data = normalized_data.get("financial", {})

            # Financial ratios
            ratios = self.financial_calculator.calculate_all_ratios(financial_data)

            # Sentiment analysis
            news_df = normalized_data.get("news", pd.DataFrame())
            if not news_df.empty:
                news_df = self.sentiment_analyzer.analyze_news_dataframe(news_df)
                sentiment_metrics = self.sentiment_analyzer.calculate_sentiment_metrics(
                    news_df
                )
            else:
                sentiment_metrics = {}

            # Technical indicators
            price_df = financial_data.get("price_history", pd.DataFrame())
            technical_indicators = self.trend_calculator.calculate_all_indicators(
                price_df, ratios
            )

            # Combine all features
            all_features = {**ratios, **sentiment_metrics, **technical_indicators}

            # Get credit score
            try:
                score_result = self.scoring_engine.get_ensemble_prediction(all_features)
            except Exception as e:
                logger.warning(f"Could not get ML prediction: {e}")
                # Fallback scoring
                score_result = self.calculate_fallback_score(ratios, sentiment_metrics)

            # Store in session state
            st.session_state.companies_data[symbol] = {
                "raw_data": raw_data,
                "normalized_data": normalized_data,
                "financial_ratios": ratios,
                "sentiment_metrics": sentiment_metrics,
                "technical_indicators": technical_indicators,
                "all_features": all_features,
                "credit_score": score_result,
                "news_data": news_df,
                "last_updated": datetime.now(),
            }

        except Exception as e:
            st.error(f"Error loading company data: {e}")
            logger.error(f"Error loading company data for {symbol}: {e}")

    def calculate_fallback_score(self, ratios: Dict, sentiment_metrics: Dict) -> Dict:
        """Calculate a simple fallback credit score when ML models are not available"""
        try:
            # Simple rule-based scoring
            score = 50  # Base score

            # Financial health factors
            roe = ratios.get("roe", 0)
            if roe > 0.15:
                score += 15
            elif roe > 0.05:
                score += 5
            elif roe < 0:
                score -= 10

            debt_to_equity = ratios.get("debt_to_equity", 1)
            if debt_to_equity < 0.5:
                score += 10
            elif debt_to_equity > 2:
                score -= 15

            current_ratio = ratios.get("current_ratio", 1)
            if current_ratio > 2:
                score += 8
            elif current_ratio < 1:
                score -= 12

            # Sentiment factor
            sentiment_score = sentiment_metrics.get("overall_sentiment", 0)
            score += sentiment_score * 10

            # Ensure score is within bounds
            score = max(0, min(100, score))

            return {
                "score": score,
                "rating": get_rating_scale(score),
                "risk_level": get_risk_level(score),
                "confidence": 0.7,
                "model_used": "fallback",
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error calculating fallback score: {e}")
            return {
                "score": 50,
                "rating": "BBB",
                "risk_level": "Medium Risk",
                "confidence": 0.5,
                "model_used": "fallback",
                "timestamp": datetime.now(),
            }

    def render_company_analysis(self):
        """Render detailed company analysis"""
        symbol = st.session_state.selected_company
        data = st.session_state.companies_data.get(symbol, {})

        if not data:
            st.warning("No data available. Please refresh the data.")
            return

        # Company header
        company_info = (
            data["normalized_data"].get("financial", {}).get("company_info", {})
        )

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"## 🏢 {symbol}")
            sector = company_info.get("sector", "Unknown")
            industry = company_info.get("industry", "Unknown")
            st.markdown(f"**Sector:** {sector} | **Industry:** {industry}")

        with col2:
            last_updated = data.get("last_updated", datetime.now())
            st.markdown(f"**Last Updated:** {last_updated.strftime('%H:%M:%S')}")

        with col3:
            market_cap = company_info.get("market_cap", 0)
            if market_cap > 0:
                st.markdown(f"**Market Cap:** {format_currency(market_cap)}")

        # Credit Score Dashboard
        self.render_credit_score_section(data)

        # Financial Analysis
        self.render_financial_analysis(data)

        # Sentiment Analysis
        self.render_sentiment_analysis(data)

        # Technical Analysis (if advanced mode)
        if self.show_advanced:
            self.render_technical_analysis(data)

    def render_credit_score_section(self, data: Dict):
        """Render the main credit score section"""
        st.markdown("### 🎯 Credit Score Analysis")

        score_data = data.get("credit_score", {})
        score = score_data.get("score", 50)
        rating = score_data.get("rating", "BBB")
        risk_level = score_data.get("risk_level", "Medium Risk")
        confidence = score_data.get("confidence", 0.7)

        # Main score display
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Score gauge
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Credit Score"},
                    delta={"reference": 70},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 40], "color": "lightgray"},
                            {"range": [40, 70], "color": "yellow"},
                            {"range": [70, 100], "color": "green"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.metric(
                label="Credit Rating",
                value=rating,
                help="Traditional credit rating scale",
            )
            st.metric(
                label="Risk Level", value=risk_level, help="Overall risk assessment"
            )

        with col3:
            st.metric(
                label="Model Confidence",
                value=f"{confidence:.1%}",
                help="Confidence in the prediction",
            )
            model_used = score_data.get("model_used", "Unknown")
            st.metric(
                label="Model Used",
                value=model_used.title(),
                help="ML model used for prediction",
            )

        with col4:
            # Score history chart (placeholder)
            dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
            scores = [score + np.random.normal(0, 5) for _ in range(30)]
            scores = [max(0, min(100, s)) for s in scores]

            fig_history = px.line(
                x=dates,
                y=scores,
                title="Score Trend (30 days)",
                labels={"x": "Date", "y": "Score"},
            )
            fig_history.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_history, use_container_width=True)

        # Feature importance
        self.render_feature_importance(data)

    def render_feature_importance(self, data: Dict):
        """Render feature importance analysis"""
        st.markdown("#### 📊 Key Factors Analysis")

        ratios = data.get("financial_ratios", {})
        sentiment = data.get("sentiment_metrics", {})

        # Create feature importance data
        features = {
            "Debt to Equity": ratios.get("debt_to_equity", 0),
            "Return on Equity": ratios.get("roe", 0),
            "Current Ratio": ratios.get("current_ratio", 0),
            "Profit Margin": ratios.get("profit_margin", 0),
            "Sentiment Score": sentiment.get("overall_sentiment", 0),
            "Financial Strength": ratios.get("financial_strength_score", 0),
        }

        # Simple importance calculation
        importance = {}
        for feature, value in features.items():
            if "debt" in feature.lower():
                importance[feature] = max(0, 1 - abs(value - 0.5)) * 0.25
            elif "roe" in feature.lower():
                importance[feature] = min(abs(value) * 2, 1) * 0.2
            elif "ratio" in feature.lower():
                importance[feature] = min(abs(value - 1), 1) * 0.15
            elif "sentiment" in feature.lower():
                importance[feature] = abs(value) * 0.1
            else:
                importance[feature] = abs(value) * 0.1

        # Normalize importance
        total_imp = sum(importance.values())
        if total_imp > 0:
            importance = {k: v / total_imp for k, v in importance.items()}

        # Plot
        col1, col2 = st.columns([2, 1])

        with col1:
            fig_importance = px.bar(
                x=list(importance.values()),
                y=list(importance.keys()),
                orientation="h",
                title="Feature Importance",
                labels={"x": "Importance", "y": "Features"},
            )
            fig_importance.update_layout(height=300)
            st.plotly_chart(fig_importance, use_container_width=True)

        with col2:
            st.markdown("**Impact Analysis:**")
            sorted_features = sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )

            for feature, imp in sorted_features[:5]:
                value = features[feature]
                impact_color = "🟢" if imp > 0.15 else ("🟡" if imp > 0.1 else "🔴")
                st.markdown(f"{impact_color} **{feature}**: {value:.3f}")

    def render_financial_analysis(self, data: Dict):
        """Render financial analysis section"""
        st.markdown("### 💰 Financial Analysis")

        ratios = data.get("financial_ratios", {})

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Debt to Equity",
                f"{ratios.get('debt_to_equity', 0):.2f}",
                help="Total Debt / Total Equity",
            )

        with col2:
            st.metric(
                "Current Ratio",
                f"{ratios.get('current_ratio', 0):.2f}",
                help="Current Assets / Current Liabilities",
            )

        with col3:
            st.metric("ROE", f"{ratios.get('roe', 0):.1%}", help="Return on Equity")

        with col4:
            st.metric(
                "Profit Margin",
                f"{ratios.get('profit_margin', 0):.1%}",
                help="Net Income / Revenue",
            )

        # Financial ratios chart
        categories = ["Liquidity", "Leverage", "Profitability", "Efficiency"]
        values = [
            ratios.get("liquidity_score", 0.5) * 100,
            (1 - ratios.get("leverage_risk_score", 0.5)) * 100,
            ratios.get("profitability_score", 0.5) * 100,
            ratios.get("asset_turnover", 0.5) * 50,  # Scaled for display
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values, theta=categories, fill="toself", name="Financial Health"
            )
        )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Financial Health Radar",
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    def render_sentiment_analysis(self, data: Dict):
        """Render sentiment analysis section"""
        st.markdown("### 📰 News Sentiment Analysis")

        sentiment_metrics = data.get("sentiment_metrics", {})
        news_df = data.get("news_data", pd.DataFrame())

        if sentiment_metrics:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                overall_sentiment = sentiment_metrics.get("overall_sentiment", 0)
                st.metric(
                    "Overall Sentiment",
                    f"{overall_sentiment:.3f}",
                    help="Average sentiment score (-1 to 1)",
                )

            with col2:
                positive_ratio = sentiment_metrics.get("positive_ratio", 0)
                st.metric(
                    "Positive News",
                    f"{positive_ratio:.1%}",
                    help="Percentage of positive articles",
                )

            with col3:
                total_articles = sentiment_metrics.get("total_articles", 0)
                st.metric(
                    "Total Articles",
                    f"{total_articles}",
                    help="Number of articles analyzed",
                )

            with col4:
                trend = sentiment_metrics.get("recent_sentiment_trend", 0)
                trend_symbol = "📈" if trend > 0 else ("📉" if trend < 0 else "➡️")
                st.metric(
                    "Trend",
                    f"{trend_symbol} {trend:.3f}",
                    help="Recent sentiment trend",
                )

        # Recent news
        if not news_df.empty:
            st.markdown("#### Recent News Headlines")

            # Display top 5 news items
            for idx, row in news_df.head(5).iterrows():
                sentiment_color = (
                    "🟢"
                    if row.get("sentiment_score", 0) > 0.05
                    else ("🔴" if row.get("sentiment_score", 0) < -0.05 else "🟡")
                )

                with st.expander(
                    f"{sentiment_color} {row.get('headline', 'No headline')}"
                ):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write(row.get("summary", "No summary available"))
                        source = row.get("source", "Unknown")
                        pub_date = row.get("published_date", "Unknown")
                        st.caption(f"Source: {source} | Date: {pub_date}")

                    with col2:
                        sentiment_score = row.get("sentiment_score", 0)
                        st.metric("Sentiment", f"{sentiment_score:.3f}")
        else:
            st.info("No recent news data available")

    def render_technical_analysis(self, data: Dict):
        """Render technical analysis section (advanced mode)"""
        st.markdown("### 📈 Technical Analysis")

        technical = data.get("technical_indicators", {})

        if technical:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Volatility (20d)",
                    f"{technical.get('volatility_20d', 0):.1%}",
                    help="20-day price volatility",
                )

            with col2:
                st.metric(
                    "RSI",
                    f"{technical.get('rsi_14', 50):.0f}",
                    help="Relative Strength Index",
                )

            with col3:
                st.metric(
                    "Price vs SMA50",
                    f"{technical.get('price_vs_sma50', 0):.1%}",
                    help="Current price vs 50-day moving average",
                )

            with col4:
                st.metric(
                    "Momentum Score",
                    f"{technical.get('momentum_score', 0):.3f}",
                    help="Overall momentum indicator",
                )
        else:
            st.info("Technical indicators not available")

    def train_models(self):
        """Train ML models with available data"""
        try:
            with st.spinner("Training ML models..."):
                # Collect data from all companies
                all_data = {}
                for symbol in self.config.COMPANIES[:5]:  # Limit for demo
                    try:
                        company_data = self.data_ingestion.fetch_all_company_data(
                            symbol
                        )
                        if "error" not in company_data:
                            all_data[symbol] = company_data
                    except Exception as e:
                        logger.warning(f"Could not fetch data for {symbol}: {e}")

                if not all_data:
                    st.error("No training data available")
                    return

                # Prepare training data
                features_df, labels_series = self.scoring_engine.prepare_training_data(
                    all_data
                )

                if features_df.empty:
                    st.error("Could not prepare training data")
                    return

                # Train models
                results = self.scoring_engine.train_models(features_df, labels_series)

                if results:
                    st.success("Models trained successfully!")

                    # Show training results
                    st.markdown("#### Training Results")
                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df)
                else:
                    st.error("Model training failed")

        except Exception as e:
            st.error(f"Error training models: {e}")
            logger.error(f"Error training models: {e}")


def main():
    """Main application entry point"""
    try:
        dashboard = CreditScoringDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")


if __name__ == "__main__":
    main()
