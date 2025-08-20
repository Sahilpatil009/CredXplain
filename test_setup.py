# Test script to verify the credit scoring system setup
import sys
import os
import logging
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    try:
        import pandas as pd

        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False

    try:
        import numpy as np

        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False

    try:
        import streamlit as st

        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
        return False

    try:
        import yfinance as yf

        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
        return False

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        print("✅ vaderSentiment imported successfully")
    except ImportError as e:
        print(f"❌ vaderSentiment import failed: {e}")
        return False

    try:
        import sklearn

        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False

    try:
        import plotly

        print("✅ plotly imported successfully")
    except ImportError as e:
        print(f"❌ plotly import failed: {e}")
        return False

    return True


def test_data_ingestion():
    """Test data ingestion functionality"""
    print("\nTesting data ingestion...")

    try:
        from src.data.ingestion import DataIngestion
        from src.utils.config import Config

        ingestion = DataIngestion()
        config = Config()

        # Test with a single company
        test_symbol = "AAPL"
        print(f"Fetching data for {test_symbol}...")

        data = ingestion.fetch_yahoo_finance_data(test_symbol, period="1mo")

        if data and "symbol" in data:
            print(f"✅ Successfully fetched data for {test_symbol}")
            return True
        else:
            print(f"❌ Failed to fetch data for {test_symbol}")
            return False

    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        return False


def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("\nTesting sentiment analysis...")

    try:
        from src.features.sentiment_analysis import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        # Test with sample texts
        test_texts = [
            "Apple reports strong quarterly earnings beating expectations",
            "Company faces regulatory challenges and declining revenues",
            "Stock price remains stable amid market uncertainty",
        ]

        for text in test_texts:
            result = analyzer.analyze_text_sentiment(text)
            if result and "sentiment_label" in result:
                print(
                    f"✅ Sentiment analysis working: '{text[:30]}...' -> {result['sentiment_label']}"
                )
            else:
                print(f"❌ Sentiment analysis failed for: {text[:30]}...")
                return False

        return True

    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        return False


def test_database():
    """Test database functionality"""
    print("\nTesting database...")

    try:
        from src.data.database import DatabaseManager

        db = DatabaseManager()

        # Test company insertion
        company_id = db.insert_company("TEST", "Test Company", "Technology", "Software")

        if company_id:
            print("✅ Database operations working")

            # Test retrieval
            company = db.get_company_by_symbol("TEST")
            if company:
                print("✅ Database retrieval working")
                return True
            else:
                print("❌ Database retrieval failed")
                return False
        else:
            print("❌ Database insertion failed")
            return False

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


def test_financial_ratios():
    """Test financial ratios calculation"""
    print("\nTesting financial ratios...")

    try:
        from src.features.financial_ratios import FinancialRatiosCalculator
        import pandas as pd

        calculator = FinancialRatiosCalculator()

        # Create sample balance sheet data
        sample_data = {
            "Total Current Assets": [1000000],
            "Total Current Liabilities": [500000],
            "Total Debt": [300000],
            "Total Stockholder Equity": [1200000],
            "Total Assets": [2000000],
        }

        balance_sheet = pd.DataFrame(sample_data)

        ratios = calculator.calculate_liquidity_ratios(balance_sheet)

        if ratios and "current_ratio" in ratios:
            print(
                f"✅ Financial ratios calculation working: Current Ratio = {ratios['current_ratio']:.2f}"
            )
            return True
        else:
            print("❌ Financial ratios calculation failed")
            return False

    except Exception as e:
        print(f"❌ Financial ratios test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("🏦 Credit Scoring System - Setup Verification")
    print("=" * 50)

    tests = [
        ("Package Imports", test_imports),
        ("Data Ingestion", test_data_ingestion),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Database Operations", test_database),
        ("Financial Ratios", test_financial_ratios),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! Your credit scoring system is ready to run.")
        print("\nTo start the dashboard, run:")
        print("streamlit run src/dashboard/app.py")
    else:
        print(
            f"\n⚠️  {len(results) - passed} tests failed. Please check the error messages above."
        )
        print(
            "Make sure all dependencies are properly installed in your virtual environment."
        )

    return passed == len(results)


if __name__ == "__main__":
    run_all_tests()
