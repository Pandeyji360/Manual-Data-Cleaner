#!/usr/bin/env python3
"""
Desktop Application Launcher
----------------------------
This script directly launches the desktop GUI application for 
the Data Analysis Pipeline without requiring command-line arguments.

Usage:
    python run_desktop.py

Author: Senior Data Engineer
Date: May 1, 2025
Version: 1.0
"""

import os
import sys

def main():
    """Launch the desktop application."""
    try:
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Import and run the desktop app
        from desktop_app import main as run_desktop
        run_desktop()
        
    except ImportError:
        print("Error: Required modules not found.")
        print("Please make sure PyQt5 and matplotlib are installed:")
        print("  pip install pyqt5 matplotlib plotly")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching desktop application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()