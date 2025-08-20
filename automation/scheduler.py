# Automated scheduler for data refresh and model retraining
import schedule
import time
import logging
from datetime import datetime
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.data.database import DatabaseManager
from src.models.scoring_engine import CreditScoringEngine
from utils.config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("automation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class AutomationScheduler:
    """Handles automated tasks for the credit scoring system"""

    def __init__(self):
        self.config = Config()
        self.db = DatabaseManager()
        self.ingestion = DataIngestion()
        self.preprocessor = DataPreprocessor()
        self.scoring_engine = CreditScoringEngine()

    def refresh_data(self):
        """Refresh data for all companies"""
        try:
            logger.info("Starting automated data refresh")

            for symbol in self.config.COMPANIES:
                try:
                    logger.info(f"Refreshing data for {symbol}")

                    # Fetch new data
                    raw_data = self.ingestion.fetch_all_company_data(symbol)

                    if "error" in raw_data:
                        logger.warning(
                            f"Could not fetch data for {symbol}: {raw_data['error']}"
                        )
                        continue

                    # Preprocess data
                    normalized_data = self.preprocessor.normalize_all_data(raw_data)

                    # Store in database
                    self.store_company_data(symbol, normalized_data)

                    logger.info(f"Successfully refreshed data for {symbol}")

                except Exception as e:
                    logger.error(f"Error refreshing data for {symbol}: {e}")
                    continue

            logger.info("Data refresh completed")

        except Exception as e:
            logger.error(f"Error in data refresh job: {e}")

    def store_company_data(self, symbol: str, normalized_data: dict):
        """Store normalized data in the database"""
        try:
            # Get or create company
            company = self.db.get_company_by_symbol(symbol)
            if not company:
                financial_data = normalized_data.get("financial", {})
                company_info = financial_data.get("company_info", {})

                company_id = self.db.insert_company(
                    symbol=symbol,
                    name=company_info.get("longName", symbol),
                    sector=company_info.get("sector", "Unknown"),
                    industry=company_info.get("industry", "Unknown"),
                )
            else:
                company_id = company["id"]

            # Store financial data
            financial_data = normalized_data.get("financial", {})
            if "company_info" in financial_data:
                self.db.insert_financial_data(
                    company_id, financial_data["company_info"]
                )

            # Store news data
            news_data = normalized_data.get("news")
            if news_data is not None and not news_data.empty:
                self.db.insert_news_data(company_id, news_data)

            logger.info(f"Stored data for {symbol} in database")

        except Exception as e:
            logger.error(f"Error storing data for {symbol}: {e}")

    def retrain_models(self):
        """Retrain ML models with latest data"""
        try:
            logger.info("Starting model retraining")

            # Collect data for training
            training_data = {}

            for symbol in self.config.COMPANIES:
                try:
                    company_data = self.ingestion.fetch_all_company_data(symbol)
                    if "error" not in company_data:
                        training_data[symbol] = company_data
                except Exception as e:
                    logger.warning(f"Could not get training data for {symbol}: {e}")

            if len(training_data) < 3:
                logger.warning("Insufficient data for model training")
                return

            # Prepare and train models
            features_df, labels_series = self.scoring_engine.prepare_training_data(
                training_data
            )

            if not features_df.empty:
                results = self.scoring_engine.train_models(features_df, labels_series)

                if results:
                    logger.info("Model retraining completed successfully")

                    # Log training results
                    for model_name, metrics in results.items():
                        logger.info(
                            f"{model_name}: CV Score = {metrics.get('cv_mean', 0):.3f}"
                        )
                else:
                    logger.error("Model retraining failed")
            else:
                logger.warning("No training data available")

        except Exception as e:
            logger.error(f"Error in model retraining: {e}")

    def generate_alerts(self):
        """Generate alerts for significant score changes"""
        try:
            logger.info("Checking for alerts")

            alerts = []

            for symbol in self.config.COMPANIES:
                try:
                    company = self.db.get_company_by_symbol(symbol)
                    if not company:
                        continue

                    # Get score history
                    score_history = self.db.get_score_history(company["id"], days=7)

                    if len(score_history) >= 2:
                        latest_score = score_history.iloc[-1]["score"]
                        previous_score = score_history.iloc[-2]["score"]

                        score_change = latest_score - previous_score

                        # Alert if score changed by more than threshold
                        if (
                            abs(score_change)
                            > self.config.DASHBOARD_CONFIG["alert_threshold"]
                        ):
                            direction = "increased" if score_change > 0 else "decreased"
                            alerts.append(
                                {
                                    "symbol": symbol,
                                    "message": f"Credit score {direction} by {abs(score_change):.1f} points",
                                    "current_score": latest_score,
                                    "change": score_change,
                                    "timestamp": datetime.now(),
                                }
                            )

                except Exception as e:
                    logger.warning(f"Error checking alerts for {symbol}: {e}")

            if alerts:
                logger.info(f"Generated {len(alerts)} alerts")
                for alert in alerts:
                    logger.warning(f"ALERT - {alert['symbol']}: {alert['message']}")
            else:
                logger.info("No alerts generated")

        except Exception as e:
            logger.error(f"Error generating alerts: {e}")

    def health_check(self):
        """Perform system health check"""
        try:
            logger.info("Performing health check")

            # Check database connectivity
            companies = self.db.get_all_companies()
            logger.info(f"Database: {len(companies)} companies in system")

            # Check data freshness
            fresh_data_count = 0
            for symbol in self.config.COMPANIES[:5]:  # Check first 5
                try:
                    data = self.ingestion.fetch_yahoo_finance_data(symbol, period="1d")
                    if data and "symbol" in data:
                        fresh_data_count += 1
                except:
                    pass

            logger.info(f"Data sources: {fresh_data_count}/5 responding")

            # Check model availability
            try:
                test_features = {f"feature_{i}": 0.5 for i in range(10)}
                score = self.scoring_engine.predict_credit_score(test_features)
                logger.info(
                    f"Models: Working (test score: {score.get('score', 'N/A')})"
                )
            except Exception as e:
                logger.warning(f"Models: Error - {e}")

            logger.info("Health check completed")

        except Exception as e:
            logger.error(f"Error in health check: {e}")

    def setup_schedule(self):
        """Set up the automation schedule"""
        logger.info("Setting up automation schedule")

        # Data refresh every 15 minutes
        schedule.every(15).minutes.do(self.refresh_data)

        # Model retraining every week
        schedule.every().monday.at("02:00").do(self.retrain_models)

        # Alert checking every hour
        schedule.every().hour.do(self.generate_alerts)

        # Health check every 6 hours
        schedule.every(6).hours.do(self.health_check)

        logger.info("Schedule configured:")
        logger.info("- Data refresh: Every 15 minutes")
        logger.info("- Model retraining: Weekly (Monday 2:00 AM)")
        logger.info("- Alert checking: Every hour")
        logger.info("- Health check: Every 6 hours")

    def run(self):
        """Run the automation scheduler"""
        logger.info("Starting Credit Scoring System Automation")

        # Setup schedule
        self.setup_schedule()

        # Initial health check
        self.health_check()

        # Run scheduler
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("Automation stopped by user")
        except Exception as e:
            logger.error(f"Automation error: {e}")


def main():
    """Main entry point"""
    scheduler = AutomationScheduler()
    scheduler.run()


if __name__ == "__main__":
    main()
