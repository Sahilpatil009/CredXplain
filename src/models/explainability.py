# Model explainability using SHAP and feature importance
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import sys
import os

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available, using feature importance only")

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.utils.helpers import create_feature_summary

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Provides model explainability using SHAP values and feature importance"""

    def __init__(self):
        self.explainer = None
        self.feature_names = []
        self.baseline_value = 0

    def initialize_explainer(
        self, model, X_train: pd.DataFrame, model_type: str = "tree"
    ):
        """Initialize SHAP explainer for the given model"""
        try:
            if not SHAP_AVAILABLE:
                logger.warning("SHAP not available, skipping explainer initialization")
                return

            self.feature_names = X_train.columns.tolist()

            if model_type == "tree":
                # For tree-based models (RandomForest, GradientBoosting, DecisionTree)
                self.explainer = shap.TreeExplainer(model)
            elif model_type == "linear":
                # For linear models (LogisticRegression)
                self.explainer = shap.LinearExplainer(model, X_train)
            else:
                # General explainer (works with any model but slower)
                self.explainer = shap.Explainer(model, X_train)

            # Calculate baseline
            if hasattr(self.explainer, "expected_value"):
                if isinstance(self.explainer.expected_value, np.ndarray):
                    self.baseline_value = self.explainer.expected_value[0]
                else:
                    self.baseline_value = self.explainer.expected_value

            logger.info(f"SHAP explainer initialized for {model_type} model")

        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")

    def get_feature_importance(
        self, model, feature_names: List[str]
    ) -> Dict[str, float]:
        """Get feature importance from the model"""
        try:
            importance_dict = {}

            if hasattr(model, "feature_importances_"):
                # Tree-based models
                importances = model.feature_importances_
                for i, name in enumerate(feature_names):
                    importance_dict[name] = float(importances[i])

            elif hasattr(model, "coef_"):
                # Linear models
                coefficients = (
                    model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                )
                for i, name in enumerate(feature_names):
                    importance_dict[name] = float(abs(coefficients[i]))

            else:
                # Default: equal importance
                for name in feature_names:
                    importance_dict[name] = 1.0 / len(feature_names)

            # Normalize to sum to 1
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {
                    k: v / total_importance for k, v in importance_dict.items()
                }

            return importance_dict

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

    def explain_prediction(
        self, features: pd.DataFrame, max_features: int = 10
    ) -> Dict:
        """Get SHAP explanation for a single prediction"""
        try:
            if not SHAP_AVAILABLE or self.explainer is None:
                return self._fallback_explanation(features, max_features)

            # Get SHAP values
            shap_values = self.explainer.shap_values(features)

            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first class for simplicity

            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # First sample

            # Create explanation dictionary
            explanation = {
                "shap_values": {},
                "feature_contributions": {},
                "baseline_value": self.baseline_value,
                "prediction_value": self.baseline_value + np.sum(shap_values),
                "top_positive_features": [],
                "top_negative_features": [],
                "explanation_text": "",
            }

            # Map SHAP values to feature names
            for i, feature_name in enumerate(self.feature_names):
                if i < len(shap_values):
                    explanation["shap_values"][feature_name] = float(shap_values[i])
                    explanation["feature_contributions"][feature_name] = {
                        "value": (
                            float(features.iloc[0, i])
                            if i < len(features.columns)
                            else 0
                        ),
                        "contribution": float(shap_values[i]),
                    }

            # Sort features by absolute contribution
            sorted_features = sorted(
                explanation["shap_values"].items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            # Get top features
            positive_features = [
                (k, v) for k, v in sorted_features[:max_features] if v > 0
            ]
            negative_features = [
                (k, v) for k, v in sorted_features[:max_features] if v < 0
            ]

            explanation["top_positive_features"] = positive_features
            explanation["top_negative_features"] = negative_features

            # Create explanation text
            explanation["explanation_text"] = self._create_explanation_text(
                positive_features,
                negative_features,
                explanation["feature_contributions"],
            )

            return explanation

        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return self._fallback_explanation(features, max_features)

    def _fallback_explanation(
        self, features: pd.DataFrame, max_features: int = 10
    ) -> Dict:
        """Fallback explanation when SHAP is not available"""
        try:
            # Simple rule-based explanation
            explanation = {
                "shap_values": {},
                "feature_contributions": {},
                "baseline_value": 0.5,
                "prediction_value": 0.5,
                "top_positive_features": [],
                "top_negative_features": [],
                "explanation_text": "Detailed explanation not available (SHAP not installed)",
            }

            # Basic feature analysis
            feature_values = features.iloc[0].to_dict()

            # Simple scoring rules
            for feature, value in feature_values.items():
                score = self._simple_feature_score(feature, value)
                explanation["feature_contributions"][feature] = {
                    "value": value,
                    "contribution": score,
                }

                if abs(score) > 0.01:  # Only significant contributions
                    if score > 0:
                        explanation["top_positive_features"].append((feature, score))
                    else:
                        explanation["top_negative_features"].append((feature, score))

            # Sort by contribution
            explanation["top_positive_features"].sort(key=lambda x: x[1], reverse=True)
            explanation["top_negative_features"].sort(key=lambda x: x[1])

            # Limit to max_features
            explanation["top_positive_features"] = explanation["top_positive_features"][
                :max_features
            ]
            explanation["top_negative_features"] = explanation["top_negative_features"][
                :max_features
            ]

            return explanation

        except Exception as e:
            logger.error(f"Error in fallback explanation: {e}")
            return {
                "explanation_text": f"Error generating explanation: {e}",
                "top_positive_features": [],
                "top_negative_features": [],
            }

    def _simple_feature_score(self, feature_name: str, value: float) -> float:
        """Simple scoring for features when SHAP is not available"""
        try:
            # Basic rules for common financial ratios
            if "roe" in feature_name.lower():
                return min(value * 2, 0.2) if value > 0 else max(value * 2, -0.2)

            elif "debt_to_equity" in feature_name.lower():
                return max(min(1 - value, 0.2), -0.2)  # Lower debt is better

            elif "current_ratio" in feature_name.lower():
                optimal = 2.0
                return max(min((value - optimal) * 0.1, 0.2), -0.2)

            elif "profit_margin" in feature_name.lower():
                return min(value * 5, 0.2) if value > 0 else max(value * 5, -0.2)

            elif "volatility" in feature_name.lower():
                return max(min(-value * 2, 0.2), -0.2)  # Lower volatility is better

            elif "sentiment" in feature_name.lower():
                return min(value * 0.3, 0.1) if value > 0 else max(value * 0.3, -0.1)

            else:
                # Default: assume positive values are good up to a point
                if value > 0:
                    return min(value * 0.01, 0.05)
                else:
                    return max(value * 0.01, -0.05)

        except Exception:
            return 0.0

    def _create_explanation_text(
        self,
        positive_features: List[Tuple],
        negative_features: List[Tuple],
        feature_contributions: Dict,
    ) -> str:
        """Create human-readable explanation text"""
        try:
            explanation_parts = []

            if positive_features:
                pos_text = "Positive factors: "
                pos_descriptions = []
                for feature, contribution in positive_features[:3]:  # Top 3
                    value = feature_contributions.get(feature, {}).get("value", 0)
                    desc = self._describe_feature_impact(
                        feature, value, contribution, True
                    )
                    if desc:
                        pos_descriptions.append(desc)

                if pos_descriptions:
                    pos_text += "; ".join(pos_descriptions)
                    explanation_parts.append(pos_text)

            if negative_features:
                neg_text = "Negative factors: "
                neg_descriptions = []
                for feature, contribution in negative_features[:3]:  # Top 3
                    value = feature_contributions.get(feature, {}).get("value", 0)
                    desc = self._describe_feature_impact(
                        feature, value, contribution, False
                    )
                    if desc:
                        neg_descriptions.append(desc)

                if neg_descriptions:
                    neg_text += "; ".join(neg_descriptions)
                    explanation_parts.append(neg_text)

            if not explanation_parts:
                return "Score based on balanced financial metrics"

            return ". ".join(explanation_parts) + "."

        except Exception as e:
            logger.error(f"Error creating explanation text: {e}")
            return "Unable to generate detailed explanation"

    def _describe_feature_impact(
        self, feature: str, value: float, contribution: float, is_positive: bool
    ) -> str:
        """Describe the impact of a specific feature"""
        try:
            # Clean feature name
            clean_name = feature.replace("_", " ").replace("economic ", "").title()

            # Format value
            if "ratio" in feature.lower() or "margin" in feature.lower():
                value_str = f"{value:.2f}"
            elif "cap" in feature.lower() or "volume" in feature.lower():
                if value > 1e9:
                    value_str = f"${value/1e9:.1f}B"
                elif value > 1e6:
                    value_str = f"${value/1e6:.1f}M"
                else:
                    value_str = f"{value:.0f}"
            else:
                value_str = f"{value:.2f}"

            # Create description
            impact = "supports" if is_positive else "pressures"

            if "debt_to_equity" in feature.lower():
                level = (
                    "low" if value < 0.5 else ("moderate" if value < 1.0 else "high")
                )
                return f"{level} debt-to-equity ratio ({value_str}) {impact} credit quality"

            elif "roe" in feature.lower():
                level = (
                    "strong"
                    if value > 0.15
                    else ("adequate" if value > 0.05 else "weak")
                )
                return f"{level} return on equity ({value_str}) {impact} score"

            elif "current_ratio" in feature.lower():
                level = (
                    "strong" if value > 2.0 else ("adequate" if value > 1.0 else "weak")
                )
                return f"{level} liquidity position ({value_str}) {impact} creditworthiness"

            elif "sentiment" in feature.lower():
                sentiment_desc = "positive" if value > 0 else "negative"
                return f"{sentiment_desc} news sentiment ({value_str}) {impact} market perception"

            else:
                return f"{clean_name} ({value_str}) {impact} overall assessment"

        except Exception:
            return f"{feature}: {value:.2f}"

    def create_summary_explanation(self, explanation: Dict, score: float) -> str:
        """Create a comprehensive summary explanation"""
        try:
            score_desc = "high" if score > 70 else ("moderate" if score > 50 else "low")
            risk_level = (
                "low risk"
                if score > 70
                else ("medium risk" if score > 50 else "high risk")
            )

            summary = f"Credit score of {score:.0f} indicates {score_desc} creditworthiness ({risk_level}). "

            if explanation.get("explanation_text"):
                summary += explanation["explanation_text"]

            # Add trend information if available
            positive_count = len(explanation.get("top_positive_features", []))
            negative_count = len(explanation.get("top_negative_features", []))

            if positive_count > negative_count:
                summary += " Overall financial profile shows more positive than negative indicators."
            elif negative_count > positive_count:
                summary += " Several risk factors require attention and monitoring."
            else:
                summary += " Financial profile shows balanced risk factors."

            return summary

        except Exception as e:
            logger.error(f"Error creating summary explanation: {e}")
            return f"Credit score: {score:.0f}. Detailed explanation not available."
