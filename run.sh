#!/bin/bash

# Automated Data Analysis Pipeline Runner
# ---------------------------------------
# This script makes it easier to run the data analysis pipeline

# Set default values
DATA_DIR="demo_datasets"
GENERATE_DATA=false
HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --generate-data)
            GENERATE_DATA=true
            shift
            ;;
        --output-dir)
            DATA_DIR="$2"
            shift
            shift
            ;;
        --help)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            HELP=true
            shift
            ;;
    esac
done

# Display help message
if $HELP; then
    echo "Automated Data Analysis Pipeline"
    echo "--------------------------------"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --generate-data      Generate demo datasets"
    echo "  --output-dir DIR     Directory to save generated datasets (default: demo_datasets)"
    echo "  --help               Display this help message"
    echo ""
    exit 0
fi

# Generate demo datasets if requested
if $GENERATE_DATA; then
    echo "Generating demo datasets..."
    python main.py --generate-data --output-dir "$DATA_DIR"
    exit 0
fi

# Run the application
echo "Starting Automated Data Analysis Pipeline..."
python main.py