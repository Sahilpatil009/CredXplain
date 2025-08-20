# Credit scoring engine with ML models
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import joblib
from datetime import datetime
import os
import sys

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.utils.config import Config
from src.utils.helpers import normalize_score, get_rating_scale, get_risk_level

logger = logging.getLogger(__name__)


class CreditScoringEngine:
    """Machine Learning engine for credit scoring"""

    def __init__(self):
        self.config = Config()
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.model_version = "v1.0"

        # Initialize models
        self.models = {
            "decision_tree": DecisionTreeClassifier(
                max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42
            ),
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=20, random_state=42
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
        }

    def prepare_training_data(
        self, companies_data: Dict
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from company features"""
        try:
            logger.info("Preparing training data")

            features_list = []
            labels_list = []

            for symbol, data in companies_data.items():
                if "error" in data:
                    continue

                # Extract features
                features = self._extract_features_from_data(data)
                if not features:
                    continue

                # Create synthetic labels for demonstration
                # In a real system, you would have historical credit events
                label = self._generate_synthetic_label(features)

                features_list.append(features)
                labels_list.append(label)

            if not features_list:
                raise ValueError("No valid training data available")

            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            labels_series = pd.Series(labels_list)

            # Store feature names
            self.feature_names = features_df.columns.tolist()

            # Handle missing values
            features_df = features_df.fillna(0)

            logger.info(
                f"Prepared training data: {len(features_df)} samples, {len(features_df.columns)} features"
            )

            return features_df, labels_series

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.Series()

    def _extract_features_from_data(self, company_data: Dict) -> Dict[str, float]:
        """Extract ML features from company data"""
        try:
            features = {}

            # Financial features
            financial_data = company_data.get("financial_data", {})
            if "company_info" in financial_data:
                info = financial_data["company_info"]
                features.update(
                    {
                        "market_cap": info.get("market_cap", 0),
                        "pe_ratio": info.get("pe_ratio", 0),
                        "pb_ratio": info.get("pb_ratio", 0),
                        "debt_to_equity": info.get("debt_to_equity", 0),
                        "current_ratio": info.get("current_ratio", 0),
                        "roe": info.get("roe", 0),
                        "roa": info.get("roa", 0),
                        "profit_margin": info.get("profit_margin", 0),
                        "revenue_growth": info.get("revenue_growth", 0),
                        "beta": info.get("beta", 1.0),
                        "dividend_yield": info.get("dividend_yield", 0),
                    }
                )

            # Price-based features
            if "price_history" in financial_data:
                price_df = pd.DataFrame(financial_data["price_history"])
                if not price_df.empty and "Close" in price_df.columns:
                    close_prices = price_df["Close"].dropna()
                    if len(close_prices) > 1:
                        # Calculate volatility
                        returns = close_prices.pct_change().dropna()
                        features["volatility"] = returns.std() * np.sqrt(252)
                        features["avg_return"] = returns.mean() * 252

                        # Price momentum
                        if len(close_prices) >= 20:
                            features["momentum_20d"] = (
                                close_prices.iloc[-1] - close_prices.iloc[-21]
                            ) / close_prices.iloc[-21]

                        # Volume if available
                        if "Volume" in price_df.columns:
                            volume = price_df["Volume"].dropna()
                            if not volume.empty:
                                features["avg_volume"] = volume.mean()

            # Sentiment features (if available)
            news_data = company_data.get("news_data", [])
            if news_data:
                # This would be calculated by sentiment analyzer
                features["sentiment_score"] = 0.1  # Placeholder
                features["news_volume"] = len(news_data)

            # Economic features (latest values)
            economic_data = company_data.get("economic_data", {})
            if "indicators" in economic_data:
                indicators = economic_data["indicators"]
                for name, data in indicators.items():
                    if data:
                        latest_value = list(data.values())[-1] if data else 0
                        features[f"economic_{name}"] = latest_value

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}

    def _generate_synthetic_label(self, features: Dict[str, float]) -> str:
        """Generate synthetic credit rating for training (demo purposes)"""
        try:
            # Simple scoring based on key ratios
            score = 0.5  # Start with neutral

            # Profitability impact
            roe = features.get("roe", 0)
            if roe > 0.15:
                score += 0.2
            elif roe < 0.05:
                score -= 0.2

            # Leverage impact
            debt_to_equity = features.get("debt_to_equity", 0)
            if debt_to_equity < 0.5:
                score += 0.15
            elif debt_to_equity > 1.5:
                score -= 0.25

            # Liquidity impact
            current_ratio = features.get("current_ratio", 1)
            if current_ratio > 2:
                score += 0.1
            elif current_ratio < 1:
                score -= 0.15

            # Volatility impact
            volatility = features.get("volatility", 0.2)
            if volatility < 0.2:
                score += 0.1
            elif volatility > 0.4:
                score -= 0.15

            # Convert to categorical rating
            if score > 0.7:
                return "AAA"
            elif score > 0.6:
                return "AA"
            elif score > 0.5:
                return "A"
            elif score > 0.4:
                return "BBB"
            elif score > 0.3:
                return "BB"
            elif score > 0.2:
                return "B"
            else:
                return "CCC"

        except Exception as e:
            logger.error(f"Error generating synthetic label: {e}")
            return "BBB"

    def train_models(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> Dict[str, float]:
        """Train all ML models"""
        try:
            logger.info("Training ML models")

            if features.empty or labels.empty:
                raise ValueError("No training data provided")

            # Encode labels
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels)

            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled_df, labels_encoded, test_size=0.2, random_state=42
            )

            results = {}

            # Train each model
            for name, model in self.models.items():
                logger.info(f"Training {name}")

                # Train model
                model.fit(X_train, y_train)

                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)

                # Cross-validation
                cv_scores = cross_val_score(
                    model, features_scaled_df, labels_encoded, cv=5
                )

                results[name] = {
                    "train_accuracy": train_score,
                    "test_accuracy": test_score,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                }

                logger.info(
                    f"{name} - Train: {train_score:.3f}, Test: {test_score:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}"
                )

            # Store label encoder
            self.label_encoder = label_encoder
            self.is_trained = True

            # Save models
            self.save_models()

            return results

        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}

    def predict_credit_score(
        self, features: Dict[str, float], model_name: str = "random_forest"
    ) -> Dict[str, float]:
        """Predict credit score for a company"""
        try:
            if not self.is_trained:
                logger.warning("Models not trained, loading from disk")
                self.load_models()

            if model_name not in self.models:
                model_name = "random_forest"

            # Prepare features
            feature_df = pd.DataFrame([features])

            # Ensure all required features are present
            for feature_name in self.feature_names:
                if feature_name not in feature_df.columns:
                    feature_df[feature_name] = 0

            # Reorder columns to match training
            feature_df = feature_df[self.feature_names]

            # Scale features
            features_scaled = self.scaler.transform(feature_df)

            # Make prediction
            model = self.models[model_name]

            # Get prediction probabilities
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features_scaled)[0]
                prediction_idx = np.argmax(probabilities)
                confidence = probabilities[prediction_idx]
            else:
                prediction_idx = model.predict(features_scaled)[0]
                confidence = 0.8  # Default confidence

            # Convert to rating
            rating = self.label_encoder.inverse_transform([prediction_idx])[0]

            # Convert rating to numerical score (0-100)
            rating_to_score = {
                "AAA": 95,
                "AA": 85,
                "A": 75,
                "BBB": 65,
                "BB": 55,
                "B": 45,
                "CCC": 25,
                "D": 10,
            }
            score = rating_to_score.get(rating, 50)

            # Get risk level
            risk_level = get_risk_level(score)

            return {
                "score": score,
                "rating": rating,
                "risk_level": risk_level,
                "confidence": confidence,
                "model_used": model_name,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error predicting credit score: {e}")
            return {
                "score": 50,
                "rating": "BBB",
                "risk_level": "Medium Risk",
                "confidence": 0.5,
                "model_used": model_name,
                "timestamp": datetime.now(),
            }

    def get_ensemble_prediction(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get ensemble prediction from multiple models"""
        try:
            predictions = []

            for model_name in ["decision_tree", "random_forest", "gradient_boosting"]:
                pred = self.predict_credit_score(features, model_name)
                predictions.append(pred["score"])

            # Calculate ensemble score
            ensemble_score = np.mean(predictions)
            ensemble_std = np.std(predictions)

            rating = get_rating_scale(ensemble_score)
            risk_level = get_risk_level(ensemble_score)

            return {
                "score": ensemble_score,
                "rating": rating,
                "risk_level": risk_level,
                "confidence": max(
                    0.5, 1 - ensemble_std / 20
                ),  # Higher std = lower confidence
                "model_used": "ensemble",
                "individual_scores": predictions,
                "score_std": ensemble_std,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error getting ensemble prediction: {e}")
            return self.predict_credit_score(features)

    def save_models(self, model_dir: str = "models/trained"):
        """Save trained models to disk"""
        try:
            os.makedirs(model_dir, exist_ok=True)

            # Save models
            for name, model in self.models.items():
                joblib.dump(model, f"{model_dir}/{name}.pkl")

            # Save scaler and other components
            joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
            joblib.dump(self.label_encoder, f"{model_dir}/label_encoder.pkl")
            joblib.dump(self.feature_names, f"{model_dir}/feature_names.pkl")

            # Save metadata
            metadata = {
                "model_version": self.model_version,
                "training_date": datetime.now().isoformat(),
                "feature_count": len(self.feature_names),
            }

            import json

            with open(f"{model_dir}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Models saved to {model_dir}")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_models(self, model_dir: str = "models/trained"):
        """Load trained models from disk"""
        try:
            # Load models
            for name in self.models.keys():
                model_path = f"{model_dir}/{name}.pkl"
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)

            # Load other components
            if os.path.exists(f"{model_dir}/scaler.pkl"):
                self.scaler = joblib.load(f"{model_dir}/scaler.pkl")

            if os.path.exists(f"{model_dir}/label_encoder.pkl"):
                self.label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")

            if os.path.exists(f"{model_dir}/feature_names.pkl"):
                self.feature_names = joblib.load(f"{model_dir}/feature_names.pkl")

            self.is_trained = True
            logger.info(f"Models loaded from {model_dir}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
