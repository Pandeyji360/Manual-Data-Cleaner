import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, Union, List, Tuple
import sys
import os

def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive summary of a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Summary DataFrame with stats for each column
    """
    # Create a dictionary to store the summary
    summary_dict = {}
    
    # For each column, compute statistics
    for column in df.columns:
        col_stats = {}
        # Data type
        col_stats['Data Type'] = str(df[column].dtype)
        
        # Count of non-null values
        col_stats['Non-Null Count'] = df[column].count()
        
        # Count of null values
        col_stats['Null Count'] = df[column].isnull().sum()
        
        # Null percentage
        col_stats['Null %'] = round(df[column].isnull().sum() / len(df) * 100, 2)
        
        # Unique values count
        col_stats['Unique Values'] = df[column].nunique()
        
        # Unique percentage
        col_stats['Unique %'] = round(df[column].nunique() / len(df) * 100, 2)
        
        # Memory usage in KB
        col_stats['Memory Usage (KB)'] = round(df[column].memory_usage(deep=True) / 1024, 2)
        
        # For numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            col_stats['Min'] = df[column].min()
            col_stats['Max'] = df[column].max()
            col_stats['Mean'] = round(df[column].mean(), 2)
            col_stats['Median'] = df[column].median()
            col_stats['Std Dev'] = round(df[column].std(), 2)
            col_stats['Skewness'] = round(df[column].skew(), 2)
            col_stats['Kurtosis'] = round(df[column].kurtosis(), 2)
            col_stats['Zero Count'] = (df[column] == 0).sum()
            col_stats['Zero %'] = round((df[column] == 0).sum() / len(df) * 100, 2)
            
            # Check for potential outliers using IQR
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))).sum()
            col_stats['Potential Outliers'] = outliers
            col_stats['Outliers %'] = round(outliers / len(df) * 100, 2)
        
        # For string columns
        elif pd.api.types.is_string_dtype(df[column]) or df[column].dtype == 'object':
            # Most common value and its frequency
            if df[column].nunique() > 0:  # Only if there are non-null values
                most_common = df[column].value_counts().index[0]
                most_common_count = df[column].value_counts().iloc[0]
                most_common_pct = round(most_common_count / len(df) * 100, 2)
                
                col_stats['Most Common'] = str(most_common)[:50]  # Truncate long values
                col_stats['Most Common Count'] = most_common_count
                col_stats['Most Common %'] = most_common_pct
                
                # String length stats if column contains strings
                if df[column].dtype == 'object':
                    try:
                        # Use apply with lambda to prevent errors with non-string types
                        string_lens = df[column].dropna().apply(lambda x: len(str(x)))
                        col_stats['Min Length'] = string_lens.min()
                        col_stats['Max Length'] = string_lens.max()
                        col_stats['Mean Length'] = round(string_lens.mean(), 2)
                    except:
                        # If error occurs, skip these stats
                        pass
        
        # For datetime columns
        elif pd.api.types.is_datetime64_dtype(df[column]):
            col_stats['Min Date'] = df[column].min()
            col_stats['Max Date'] = df[column].max()
            col_stats['Range (days)'] = (df[column].max() - df[column].min()).days
        
        # Add column stats to the summary dictionary
        summary_dict[column] = col_stats
    
    # Convert the dictionary to a DataFrame
    summary_df = pd.DataFrame.from_dict(summary_dict, orient='index')
    
    # Fill NaN values with 'N/A' for better display
    summary_df = summary_df.fillna('N/A')
    
    return summary_df

def convert_df_to_csv(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to CSV format.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        str: CSV string representation of the DataFrame
    """
    # Convert dataframe to csv
    csv = df.to_csv(index=False)
    return csv

def get_file_size(file) -> str:
    """
    Get human-readable file size.
    
    Args:
        file: File object
        
    Returns:
        str: Human-readable file size
    """
    # Get file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    # Convert to human-readable format
    for unit in ['B', 'KB', 'MB', 'GB']:
        if file_size < 1024.0:
            return f"{file_size:.2f} {unit}"
        file_size /= 1024.0
    
    return f"{file_size:.2f} TB"

def is_large_file(file, threshold_mb=100) -> bool:
    """
    Check if a file is larger than the threshold size.
    
    Args:
        file: File object
        threshold_mb (int): Size threshold in megabytes
        
    Returns:
        bool: True if file is larger than threshold, False otherwise
    """
    # Get file size in bytes
    file.seek(0, os.SEEK_END)
    file_size_bytes = file.tell()
    file.seek(0)
    
    # Convert threshold to bytes
    threshold_bytes = threshold_mb * 1024 * 1024
    
    return file_size_bytes > threshold_bytes

def get_memory_usage() -> Dict[str, Union[float, str]]:
    """
    Get current memory usage of the application.
    
    Returns:
        dict: Dictionary with memory usage information
    """
    # Get memory usage in bytes
    mem_info = {}
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info['memory_usage_bytes'] = process.memory_info().rss
        mem_info['memory_usage'] = f"{process.memory_info().rss / (1024 * 1024):.2f} MB"
        mem_info['memory_percent'] = f"{process.memory_percent():.2f}%"
    except ImportError:
        # If psutil is not available, use a simple estimate
        mem_info['memory_usage'] = f"psutil module not available"
        mem_info['memory_percent'] = "N/A"
    
    return mem_info

def infer_encoding(file, n_chars=20000):
    """
    Try to infer the encoding of a text file.
    
    Args:
        file: File object
        n_chars (int): Number of characters to read for inference
        
    Returns:
        str: Detected encoding
    """
    try:
        import chardet
        
        # Read a sample of the file
        file.seek(0)
        raw_data = file.read(n_chars)
        file.seek(0)
        
        if isinstance(raw_data, str):
            # If already decoded (string), encode to bytes
            raw_data = raw_data.encode()
        
        # Detect encoding
        result = chardet.detect(raw_data)
        return result['encoding']
    
    except ImportError:
        # If chardet is not available, default to utf-8
        return 'utf-8'
    except Exception:
        # In case of any other error, default to utf-8
        return 'utf-8'

def determine_delimiter(file):
    """
    Try to determine the delimiter used in a CSV file.
    
    Args:
        file: File object
        
    Returns:
        str: Detected delimiter
    """
    # Read the first line
    file.seek(0)
    first_line = file.readline().decode('utf-8')
    file.seek(0)
    
    # Common delimiters
    delimiters = [',', ';', '\t', '|']
    
    # Count occurrences of each delimiter
    counts = {delimiter: first_line.count(delimiter) for delimiter in delimiters}
    
    # Choose the delimiter with the highest count
    max_count = 0
    detected_delimiter = ','  # Default to comma
    
    for delimiter, count in counts.items():
        if count > max_count:
            max_count = count
            detected_delimiter = delimiter
    
    return detected_delimiter
