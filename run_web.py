#!/usr/bin/env python3
"""
Web Application Launcher
-----------------------
This script directly launches the Streamlit web interface for 
the Data Analysis Pipeline without requiring command-line arguments.

Usage:
    python run_web.py

Author: Senior Data Engineer
Date: May 1, 2025
Version: 1.0
"""

import os
import sys
import importlib.util
import subprocess
import platform

def main():
    """Launch the web application."""
    try:
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Check if Streamlit is installed
        streamlit_spec = importlib.util.find_spec("streamlit")
        if streamlit_spec is None:
            print("Error: Streamlit is not installed.")
            print("Please install it with: pip install streamlit")
            sys.exit(1)
        
        # On Windows or with newer Python versions, use subprocess to avoid asyncio errors
        is_windows = platform.system() == "Windows"
        python_version = sys.version_info
        
        if is_windows or (python_version.major == 3 and python_version.minor >= 11):
            print("Starting Streamlit web interface using direct command...")
            # Use subprocess to run streamlit directly, which avoids asyncio issues
            cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd)
        else:
            # Traditional method for older Python versions on non-Windows platforms
            print("Starting Streamlit web interface using bootstrap method...")
            import streamlit.web.bootstrap
            streamlit.web.bootstrap.run("app.py", "", [], flag_options={})
        
    except ImportError:
        print("Error: Required modules not found.")
        print("Please make sure Streamlit is installed:")
        print("  pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching web application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()