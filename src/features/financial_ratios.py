# Financial ratios calculation
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.utils.helpers import safe_divide, calculate_percentage_change

logger = logging.getLogger(__name__)


class FinancialRatiosCalculator:
    """Calculate various financial ratios for credit scoring"""

    def __init__(self):
        pass

    def calculate_liquidity_ratios(
        self, balance_sheet: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate liquidity ratios"""
        try:
            ratios = {}

            if balance_sheet.empty:
                return {
                    "current_ratio": 0,
                    "quick_ratio": 0,
                    "cash_ratio": 0,
                    "working_capital": 0,
                }

            # Get the most recent data (first column)
            latest = (
                balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
            )

            # Current Ratio = Current Assets / Current Liabilities
            current_assets = latest.get("Total Current Assets", 0)
            current_liabilities = latest.get("Total Current Liabilities", 0)
            ratios["current_ratio"] = safe_divide(
                current_assets, current_liabilities, 0
            )

            # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
            inventory = latest.get("Inventory", 0)
            quick_assets = current_assets - inventory
            ratios["quick_ratio"] = safe_divide(quick_assets, current_liabilities, 0)

            # Cash Ratio = Cash / Current Liabilities
            cash = latest.get("Cash And Cash Equivalents", 0)
            ratios["cash_ratio"] = safe_divide(cash, current_liabilities, 0)

            # Working Capital
            ratios["working_capital"] = current_assets - current_liabilities

            return ratios

        except Exception as e:
            logger.error(f"Error calculating liquidity ratios: {e}")
            return {
                "current_ratio": 0,
                "quick_ratio": 0,
                "cash_ratio": 0,
                "working_capital": 0,
            }

    def calculate_leverage_ratios(
        self, balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate leverage/debt ratios"""
        try:
            ratios = {}

            if balance_sheet.empty:
                return {
                    "debt_to_equity": 0,
                    "debt_to_assets": 0,
                    "equity_ratio": 0,
                    "interest_coverage": 0,
                }

            # Get the most recent data
            bs_latest = (
                balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
            )
            income_latest = (
                income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
            )

            # Total debt
            total_debt = bs_latest.get("Total Debt", 0)
            if total_debt == 0:
                # Alternative calculation
                long_term_debt = bs_latest.get("Long Term Debt", 0)
                short_term_debt = bs_latest.get("Current Debt", 0)
                total_debt = long_term_debt + short_term_debt

            # Total equity
            total_equity = bs_latest.get("Total Stockholder Equity", 0)
            total_assets = bs_latest.get("Total Assets", 0)

            # Debt-to-Equity Ratio
            ratios["debt_to_equity"] = safe_divide(total_debt, total_equity, 0)

            # Debt-to-Assets Ratio
            ratios["debt_to_assets"] = safe_divide(total_debt, total_assets, 0)

            # Equity Ratio
            ratios["equity_ratio"] = safe_divide(total_equity, total_assets, 0)

            # Interest Coverage Ratio = EBIT / Interest Expense
            ebit = income_latest.get("EBIT", 0)
            if ebit == 0:
                # Alternative: Operating Income
                ebit = income_latest.get("Operating Income", 0)

            interest_expense = income_latest.get("Interest Expense", 0)
            ratios["interest_coverage"] = safe_divide(ebit, abs(interest_expense), 0)

            return ratios

        except Exception as e:
            logger.error(f"Error calculating leverage ratios: {e}")
            return {
                "debt_to_equity": 0,
                "debt_to_assets": 0,
                "equity_ratio": 0,
                "interest_coverage": 0,
            }

    def calculate_profitability_ratios(
        self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate profitability ratios"""
        try:
            ratios = {}

            if income_stmt.empty:
                return {
                    "profit_margin": 0,
                    "operating_margin": 0,
                    "roe": 0,
                    "roa": 0,
                    "roic": 0,
                }

            income_latest = (
                income_stmt.iloc[:, 0] if not income_stmt.empty else pd.Series()
            )
            bs_latest = (
                balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
            )

            # Revenue and profit metrics
            revenue = income_latest.get("Total Revenue", 0)
            net_income = income_latest.get("Net Income", 0)
            operating_income = income_latest.get("Operating Income", 0)

            # Profit Margin = Net Income / Revenue
            ratios["profit_margin"] = safe_divide(net_income, revenue, 0)

            # Operating Margin = Operating Income / Revenue
            ratios["operating_margin"] = safe_divide(operating_income, revenue, 0)

            # Return on Equity (ROE) = Net Income / Total Equity
            total_equity = bs_latest.get("Total Stockholder Equity", 0)
            ratios["roe"] = safe_divide(net_income, total_equity, 0)

            # Return on Assets (ROA) = Net Income / Total Assets
            total_assets = bs_latest.get("Total Assets", 0)
            ratios["roa"] = safe_divide(net_income, total_assets, 0)

            # Return on Invested Capital (ROIC)
            invested_capital = total_equity + bs_latest.get("Total Debt", 0)
            ratios["roic"] = safe_divide(net_income, invested_capital, 0)

            return ratios

        except Exception as e:
            logger.error(f"Error calculating profitability ratios: {e}")
            return {
                "profit_margin": 0,
                "operating_margin": 0,
                "roe": 0,
                "roa": 0,
                "roic": 0,
            }

    def calculate_efficiency_ratios(
        self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate efficiency/activity ratios"""
        try:
            ratios = {}

            if income_stmt.empty or balance_sheet.empty:
                return {
                    "asset_turnover": 0,
                    "inventory_turnover": 0,
                    "receivables_turnover": 0,
                    "equity_turnover": 0,
                }

            income_latest = income_stmt.iloc[:, 0]
            bs_latest = balance_sheet.iloc[:, 0]

            revenue = income_latest.get("Total Revenue", 0)
            cogs = income_latest.get("Cost Of Revenue", 0)

            # Asset Turnover = Revenue / Total Assets
            total_assets = bs_latest.get("Total Assets", 0)
            ratios["asset_turnover"] = safe_divide(revenue, total_assets, 0)

            # Inventory Turnover = COGS / Inventory
            inventory = bs_latest.get("Inventory", 0)
            ratios["inventory_turnover"] = safe_divide(cogs, inventory, 0)

            # Receivables Turnover = Revenue / Accounts Receivable
            receivables = bs_latest.get("Accounts Receivable", 0)
            ratios["receivables_turnover"] = safe_divide(revenue, receivables, 0)

            # Equity Turnover = Revenue / Total Equity
            total_equity = bs_latest.get("Total Stockholder Equity", 0)
            ratios["equity_turnover"] = safe_divide(revenue, total_equity, 0)

            return ratios

        except Exception as e:
            logger.error(f"Error calculating efficiency ratios: {e}")
            return {
                "asset_turnover": 0,
                "inventory_turnover": 0,
                "receivables_turnover": 0,
                "equity_turnover": 0,
            }

    def calculate_cash_flow_ratios(
        self, cash_flow: pd.DataFrame, balance_sheet: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate cash flow ratios"""
        try:
            ratios = {}

            if cash_flow.empty:
                return {
                    "operating_cash_flow_ratio": 0,
                    "free_cash_flow": 0,
                    "cash_coverage_ratio": 0,
                    "cash_flow_to_debt": 0,
                }

            cf_latest = cash_flow.iloc[:, 0]
            bs_latest = (
                balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series()
            )

            # Operating Cash Flow
            operating_cf = cf_latest.get("Operating Cash Flow", 0)

            # Free Cash Flow = Operating Cash Flow - Capital Expenditures
            capex = cf_latest.get("Capital Expenditures", 0)
            ratios["free_cash_flow"] = operating_cf - abs(capex)

            # Operating Cash Flow Ratio = Operating Cash Flow / Current Liabilities
            current_liabilities = bs_latest.get("Total Current Liabilities", 0)
            ratios["operating_cash_flow_ratio"] = safe_divide(
                operating_cf, current_liabilities, 0
            )

            # Cash Coverage Ratio = Operating Cash Flow / Interest Expense
            # Note: We'll use a default interest expense if not available
            ratios["cash_coverage_ratio"] = safe_divide(
                operating_cf, 1000000, 0
            )  # Placeholder

            # Cash Flow to Debt Ratio = Operating Cash Flow / Total Debt
            total_debt = bs_latest.get("Total Debt", 0)
            ratios["cash_flow_to_debt"] = safe_divide(operating_cf, total_debt, 0)

            return ratios

        except Exception as e:
            logger.error(f"Error calculating cash flow ratios: {e}")
            return {
                "operating_cash_flow_ratio": 0,
                "free_cash_flow": 0,
                "cash_coverage_ratio": 0,
                "cash_flow_to_debt": 0,
            }

    def calculate_all_ratios(self, financial_data: Dict) -> Dict[str, float]:
        """Calculate all financial ratios"""
        try:
            logger.info("Calculating all financial ratios")

            # Extract financial statements
            balance_sheet = financial_data.get("balance_sheet", pd.DataFrame())
            income_stmt = financial_data.get("income_statement", pd.DataFrame())
            cash_flow = financial_data.get("cash_flow", pd.DataFrame())

            all_ratios = {}

            # Calculate different categories of ratios
            all_ratios.update(self.calculate_liquidity_ratios(balance_sheet))
            all_ratios.update(
                self.calculate_leverage_ratios(balance_sheet, income_stmt)
            )
            all_ratios.update(
                self.calculate_profitability_ratios(income_stmt, balance_sheet)
            )
            all_ratios.update(
                self.calculate_efficiency_ratios(income_stmt, balance_sheet)
            )
            all_ratios.update(self.calculate_cash_flow_ratios(cash_flow, balance_sheet))

            # Add company info ratios if available
            company_info = financial_data.get("company_info", {})
            if company_info:
                all_ratios.update(
                    {
                        "market_cap": company_info.get("market_cap", 0),
                        "pe_ratio": company_info.get("pe_ratio", 0),
                        "pb_ratio": company_info.get("pb_ratio", 0),
                        "beta": company_info.get("beta", 1.0),
                        "dividend_yield": company_info.get("dividend_yield", 0),
                    }
                )

            # Calculate derived metrics
            all_ratios.update(self._calculate_derived_metrics(all_ratios))

            return all_ratios

        except Exception as e:
            logger.error(f"Error calculating all ratios: {e}")
            return {}

    def _calculate_derived_metrics(self, ratios: Dict[str, float]) -> Dict[str, float]:
        """Calculate derived financial metrics"""
        derived = {}

        # Financial Strength Score (composite)
        financial_strength = (
            min(ratios.get("current_ratio", 0) / 2, 1) * 0.2  # Liquidity
            + min(1 / max(ratios.get("debt_to_equity", 1), 0.1), 1)
            * 0.3  # Low leverage
            + min(ratios.get("roe", 0) * 10, 1) * 0.25  # Profitability
            + min(ratios.get("interest_coverage", 0) / 5, 1) * 0.25  # Debt service
        )
        derived["financial_strength_score"] = financial_strength

        # Liquidity Score
        liquidity_score = (
            min(ratios.get("current_ratio", 0) / 2, 1) * 0.4
            + min(ratios.get("quick_ratio", 0) / 1.5, 1) * 0.6
        )
        derived["liquidity_score"] = liquidity_score

        # Leverage Risk Score (higher = more risky)
        leverage_risk = min(ratios.get("debt_to_equity", 0) / 2, 1)
        derived["leverage_risk_score"] = leverage_risk

        # Profitability Score
        profitability_score = (
            min(ratios.get("profit_margin", 0) * 20, 1) * 0.4
            + min(ratios.get("roe", 0) * 10, 1) * 0.6
        )
        derived["profitability_score"] = profitability_score

        return derived
