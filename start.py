# Startup script for the Credit Scoring System
import os
import sys
import subprocess
import logging
from datetime import datetime


def activate_environment():
    """Instructions for activating the virtual environment"""
    print("🏦 Credit Scoring System Startup")
    print("=" * 50)
    print()

    print("📋 Prerequisites:")
    print("1. Virtual environment should be created and activated")
    print("2. All dependencies should be installed")
    print()

    print("🔧 To activate the virtual environment:")
    print("Windows PowerShell:")
    print("   .\\venv\\Scripts\\Activate.ps1")
    print()
    print("Windows Command Prompt:")
    print("   venv\\Scripts\\activate.bat")
    print()
    print("Linux/Mac:")
    print("   source venv/bin/activate")
    print()


def check_environment():
    """Check if we're in a virtual environment"""
    venv_active = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

    if venv_active:
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️  Virtual environment not detected")
        print("Please activate your virtual environment first")
        return False


def start_dashboard():
    """Start the Streamlit dashboard"""
    try:
        print("🚀 Starting Credit Scoring Dashboard...")
        print("=" * 50)
        print()
        print("Dashboard will open in your browser at: http://localhost:8501")
        print("Press Ctrl+C to stop the dashboard")
        print()

        # Change to the correct directory
        dashboard_path = os.path.join("src", "dashboard", "app.py")

        if not os.path.exists(dashboard_path):
            print(f"❌ Dashboard file not found: {dashboard_path}")
            return False

        # Run streamlit
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            dashboard_path,
            "--server.port=8501",
        ]
        subprocess.run(cmd)

        return True

    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return False


def run_tests():
    """Run the test suite"""
    try:
        print("🧪 Running System Tests...")
        print("=" * 50)

        # Run the test script
        test_script = "test_setup.py"
        if os.path.exists(test_script):
            subprocess.run([sys.executable, test_script])
        else:
            print(f"❌ Test script not found: {test_script}")

    except Exception as e:
        print(f"❌ Error running tests: {e}")


def show_menu():
    """Show the main menu"""
    print("\n📋 What would you like to do?")
    print("1. Run system tests")
    print("2. Start dashboard")
    print("3. Show environment info")
    print("4. Exit")
    print()

    choice = input("Enter your choice (1-4): ").strip()
    return choice


def show_environment_info():
    """Show information about the current environment"""
    print("\n🔍 Environment Information:")
    print("=" * 30)
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Virtual Environment: {'Yes' if check_environment() else 'No'}")
    print()

    # Show key packages
    try:
        import pandas

        print(f"✅ pandas: {pandas.__version__}")
    except ImportError:
        print("❌ pandas: Not installed")

    try:
        import streamlit

        print(f"✅ streamlit: {streamlit.__version__}")
    except ImportError:
        print("❌ streamlit: Not installed")

    try:
        import sklearn

        print(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn: Not installed")


def main():
    """Main application entry point"""
    activate_environment()

    if not check_environment():
        print("\nPlease activate your virtual environment and try again.")
        return

    while True:
        choice = show_menu()

        if choice == "1":
            run_tests()
        elif choice == "2":
            start_dashboard()
        elif choice == "3":
            show_environment_info()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
