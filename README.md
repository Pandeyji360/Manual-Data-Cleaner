# Automated Data Analysis Pipeline

An advanced data analysis system with powerful data cleaning, visualization, and real-time monitoring capabilities for large datasets.

## Overview

This application provides a comprehensive solution for data professionals to:

- **Analyze datasets** with up to millions of records
- **Identify data quality issues** automatically
- **Clean and transform data** with customizable methods
- **Visualize patterns and insights** with interactive charts
- **Monitor data processing** in real-time
- **Generate detailed reports** on data quality

## Features

- **Intuitive UI**: User-friendly Streamlit interface for easy interaction
- **Large Data Support**: Uses Dask for distributed data processing when files exceed 100MB
- **Advanced Data Cleaning**: Handles missing values, outliers, duplicates, and data type inconsistencies
- **Intelligent Recommendations**: Suggests optimal cleaning strategies based on data characteristics
- **Interactive Visualizations**: Comprehensive charts for data distributions, correlations, and outliers
- **Real-time Monitoring**: Tracks processing time, memory usage, and operation logs
- **Exportable Results**: Download cleaned data and reports

## Demo Datasets

The application comes with a data generator that can create realistic sample datasets with:

- **User data**: 100,000 records
- **Transaction data**: 200,000 records
- **Product data**: 10,000 records
- **Customer support data**: 50,000 records
- **Website analytics data**: 200,000 records

Each dataset includes intentional data quality issues (missing values, outliers, format errors) to demonstrate the application's capabilities.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Required Python packages (see requirements in pyproject.toml)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/data-analysis-pipeline.git
cd data-analysis-pipeline

# Install dependencies
pip install -e .
```

### Running the Application

You can use the included shell script to run the application:

```bash
# Make the script executable
chmod +x run.sh

# Run the application
./run.sh
```

Or run directly with Python:

```bash
# Run the main application
python main.py

# Generate demo datasets
python main.py --generate-data
```

### Command-line Options

```
Usage: ./run.sh [options]

Options:
  --generate-data      Generate demo datasets
  --output-dir DIR     Directory to save generated datasets (default: demo_datasets)
  --help               Display help message
```

## Usage Guide

1. **Start the application** using the commands above
2. **Upload data** via the sidebar file uploader
3. **Analyze data quality** in the "Data Quality Analysis" tab
4. **Clean data** using recommended strategies in the "Data Cleaning" tab
5. **Visualize insights** in the "Visualization" tab
6. **Monitor processing** through the "Processing Logs" tab
7. **Export results** using the download buttons

## Main Components

- **app.py**: Main Streamlit application interface
- **data_cleaning.py**: Data quality assessment and cleaning functions
- **data_visualization.py**: Visualization components and chart generation
- **demo_data_generator.py**: Sample dataset generation
- **logger.py**: Logging and activity tracking system
- **utils.py**: Helper functions and utilities
- **main.py**: Application entry point

## Customization

You can customize the application by:

- Modifying data cleaning strategies in `data_cleaning.py`
- Adding new visualization types in `data_visualization.py`
- Configuring application settings in the `.streamlit/config.toml` file

## Performance Considerations

- For very large files (>500MB), consider increasing system memory
- The application uses Dask for files >100MB to optimize performance
- Memory usage is monitored in real-time to prevent crashes

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the interactive web framework
- Pandas and Dask for powerful data processing
- Plotly and Matplotlib for data visualization