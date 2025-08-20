#!/usr/bin/env python3
"""
Simple script to start the Streamlit dashboard
"""
import subprocess
import sys
import os

# Set the working directory
os.chdir(r"c:\Users\Sahil\OneDrive\My Learnings\CredTech Hakathon")

# Set environment variable to skip Streamlit welcome
os.environ["STREAMLIT_GLOBAL_DEV_MODE"] = "false"

# Start streamlit
try:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/dashboard/app.py",
            "--server.headless",
            "true",
            "--server.port",
            "8502",
            "--browser.gatherUsageStats",
            "false",
        ]
    )
except KeyboardInterrupt:
    print("\n🛑 Dashboard stopped by user")
