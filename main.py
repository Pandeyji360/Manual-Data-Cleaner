#!/usr/bin/env python3
"""
Automated Data Analysis Pipeline
--------------------------------
This script serves as the entry point for running the data analysis pipeline.
It provides advanced data cleaning, visualization, and real-time monitoring
capabilities for large datasets.

Usage:
    python main.py [options]
    
Options:
    --desktop      Run the desktop GUI application
    --web          Run the web-based Streamlit application (default)
    --generate-data Generate demo datasets for testing

Author: Senior Data Engineer
Date: May 1, 2025
Version: 1.0
"""

import os
import sys
import argparse
import importlib.util
import subprocess
import platform
from demo_data_generator import (
    generate_user_data,
    generate_transaction_data,
    generate_product_data,
    generate_customer_support_data,
    generate_website_analytics_data,
    save_csv
)

def ensure_logs_directory():
    """Ensure the logs directory exists."""
    os.makedirs("logs", exist_ok=True)


def generate_demo_datasets(output_dir="demo_datasets"):
    """
    Generate demo datasets for analysis.
    
    Args:
        output_dir (str): Directory to save the generated datasets
    """
    print(f"Generating demo datasets in {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate user data (100,000 records)
    print("Generating user data...")
    user_df = generate_user_data(num_rows=100000, error_rate=0.1, include_missing=True)
    save_csv(user_df, os.path.join(output_dir, "user_data.csv"))
    print(f"✅ User data saved: {len(user_df)} records")
    
    # Generate transaction data (200,000 records)
    print("Generating transaction data...")
    transaction_df = generate_transaction_data(num_rows=200000, error_rate=0.1, include_missing=True)
    save_csv(transaction_df, os.path.join(output_dir, "transaction_data.csv"))
    print(f"✅ Transaction data saved: {len(transaction_df)} records")
    
    # Generate product data (10,000 records)
    print("Generating product data...")
    product_df = generate_product_data(num_rows=10000, error_rate=0.1, include_missing=True)
    save_csv(product_df, os.path.join(output_dir, "product_data.csv"))
    print(f"✅ Product data saved: {len(product_df)} records")
    
    # Generate customer support data (50,000 records)
    print("Generating customer support data...")
    cs_df = generate_customer_support_data(num_rows=50000, error_rate=0.1, include_missing=True)
    save_csv(cs_df, os.path.join(output_dir, "customer_support_data.csv"))
    print(f"✅ Customer support data saved: {len(cs_df)} records")
    
    # Generate website analytics data (200,000 records)
    print("Generating website analytics data...")
    analytics_df = generate_website_analytics_data(num_rows=200000, error_rate=0.1, include_missing=True)
    save_csv(analytics_df, os.path.join(output_dir, "website_analytics_data.csv"))
    print(f"✅ Website analytics data saved: {len(analytics_df)} records")
    
    print("\n🎉 Demo dataset generation complete!")
    print(f"Generated 5 datasets with a total of {len(user_df) + len(transaction_df) + len(product_df) + len(cs_df) + len(analytics_df):,} records")
    print(f"Files are saved in the '{output_dir}' directory")


def run_streamlit_app():
    """Launch the Streamlit web application."""
    print("Starting Streamlit web interface for Data Analysis Pipeline...")
    
    # Import streamlit module dynamically to avoid errors if not installed
    streamlit_spec = importlib.util.find_spec("streamlit")
    if streamlit_spec is None:
        print("Error: Streamlit is not installed or not found.")
        print("Install it with: pip install streamlit")
        sys.exit(1)
    
    # Check platform and Python version
    is_windows = platform.system() == "Windows"
    python_version = sys.version_info
    
    if is_windows or (python_version.major == 3 and python_version.minor >= 11):
        # For Windows or newer Python versions, use subprocess to avoid asyncio issues
        print("Using direct command to launch Streamlit (recommended for Windows/Python 3.11+)...")
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        # Traditional method for older Python versions on non-Windows platforms
        print("Using bootstrap method to launch Streamlit...")
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run("app.py", "", [], flag_options={})


def run_desktop_app():
    """Launch the desktop GUI application."""
    print("Starting desktop application for Data Analysis Pipeline...")
    
    # Check if PyQt5 is installed
    pyqt_spec = importlib.util.find_spec("PyQt5")
    if pyqt_spec is None:
        print("Error: PyQt5 is not installed or not found.")
        print("Install it with: pip install pyqt5")
        sys.exit(1)
    
    # Import and run the desktop app
    try:
        from desktop_app import main as run_qt_app
        run_qt_app()
    except ImportError as e:
        print(f"Error loading desktop application: {e}")
        print("Make sure desktop_app.py is in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running desktop application: {e}")
        sys.exit(1)


def main():
    """Main entry point of the application."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Automated Data Analysis Pipeline")
    parser.add_argument("--generate-data", action="store_true", help="Generate demo datasets")
    parser.add_argument("--output-dir", default="demo_datasets", help="Directory to save generated datasets")
    parser.add_argument("--desktop", action="store_true", help="Run the desktop GUI application")
    parser.add_argument("--web", action="store_true", help="Run the web-based Streamlit application")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure logs directory exists
    ensure_logs_directory()
    
    # Generate demo datasets if requested
    if args.generate_data:
        generate_demo_datasets(args.output_dir)
        print("\nYou can now run the application with:")
        print("  - Web Interface: python main.py --web")
        print("  - Desktop Application: python main.py --desktop")
        print("  - Default (Web): python main.py")
        return
    
    # Determine which application to run (desktop or web)
    if args.desktop:
        run_desktop_app()
    elif args.web or (not args.desktop and not args.web):  # Default to web if nothing specified
        run_streamlit_app()


if __name__ == "__main__":
    main()