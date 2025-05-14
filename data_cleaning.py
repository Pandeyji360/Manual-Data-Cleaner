import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def identify_data_issues(df):
    """
    Identify various data quality issues in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        
    Returns:
        dict: Dictionary with identified issues by type and column
    """
    issues = {
        'missing_values': {},
        'outliers': {},
        'invalid_formats': {},
        'duplicates': {},
        'inconsistent_values': {}
    }
    
    # Check for missing values
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        issues['missing_values'][col] = {'count': int(missing[col]), 'percentage': round(missing[col] / len(df) * 100, 2)}
    
    # Check for duplicated rows
    if df.duplicated().any():
        issues['duplicates']['rows'] = {'count': int(df.duplicated().sum()), 'percentage': round(df.duplicated().sum() / len(df) * 100, 2)}
    
    # Check for outliers in numerical columns using IQR method
    for col in df.select_dtypes(include=['number']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].values
        if len(outliers) > 0:
            issues['outliers'][col] = outliers.tolist()[:10]  # Limit to first 10 for brevity
    
    # Check for invalid formats in common data types
    # Email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Date validation (simple check for yyyy-mm-dd format)
    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
    
    # URL validation (simple)
    url_pattern = r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(/[-\w%!$&\'()*+,;=:]+)*$'
    
    # Phone number validation (simple international format)
    phone_pattern = r'^\+?[0-9]{8,15}$'
    
    # Check string columns
    for col in df.select_dtypes(include=['object']).columns:
        # Check for possible emails
        if col.lower().find('email') != -1:
            invalid_emails = df[~df[col].str.contains(email_pattern, na=False, regex=True)]
            if not invalid_emails.empty:
                issues['invalid_formats'][col] = {'type': 'email', 'examples': invalid_emails[col].dropna().tolist()[:5]}
        
        # Check for possible dates
        elif col.lower().find('date') != -1:
            invalid_dates = df[~df[col].str.contains(date_pattern, na=False, regex=True)]
            if not invalid_dates.empty:
                issues['invalid_formats'][col] = {'type': 'date', 'examples': invalid_dates[col].dropna().tolist()[:5]}
        
        # Check for possible URLs
        elif col.lower().find('url') != -1 or col.lower().find('website') != -1:
            invalid_urls = df[~df[col].str.contains(url_pattern, na=False, regex=True)]
            if not invalid_urls.empty:
                issues['invalid_formats'][col] = {'type': 'url', 'examples': invalid_urls[col].dropna().tolist()[:5]}
        
        # Check for possible phone numbers
        elif col.lower().find('phone') != -1:
            invalid_phones = df[~df[col].str.contains(phone_pattern, na=False, regex=True)]
            if not invalid_phones.empty:
                issues['invalid_formats'][col] = {'type': 'phone', 'examples': invalid_phones[col].dropna().tolist()[:5]}
        
        # Check for inconsistent capitalization
        value_counts = df[col].value_counts()
        for val in value_counts.index:
            if isinstance(val, str) and val.lower() in [v.lower() for v in value_counts.index if v != val]:
                if col not in issues['inconsistent_values']:
                    issues['inconsistent_values'][col] = []
                issues['inconsistent_values'][col].append(val)
    
    return issues

def get_data_quality_report(df):
    """
    Generate a comprehensive data quality report for a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        dict: Dictionary with different data quality reports
    """
    quality_report = {}
    
    # Basic stats
    basic_stats = df.describe(include='all').T
    basic_stats['missing'] = df.isnull().sum()
    basic_stats['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
    basic_stats['unique_values'] = df.nunique()
    basic_stats['unique_pct'] = (df.nunique() / len(df) * 100).round(2)
    quality_report['basic_stats'] = basic_stats
    
    # Data types
    dtypes_df = pd.DataFrame(df.dtypes, columns=['data_type'])
    dtypes_df['memory_usage'] = df.memory_usage(deep=True)[1:] / 1024
    dtypes_df['memory_usage'] = dtypes_df['memory_usage'].round(2).astype(str) + ' KB'
    quality_report['data_types'] = dtypes_df
    
    # Value distributions for categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cat_distributions = {}
    for col in cat_cols:
        value_counts = df[col].value_counts().head(10)
        cat_distributions[col] = pd.DataFrame({
            'value': value_counts.index,
            'count': value_counts.values,
            'percentage': (value_counts.values / len(df) * 100).round(2)
        })
    quality_report['categorical_distributions'] = cat_distributions
    
    # Statistical tests for numerical columns
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        numeric_tests = pd.DataFrame(index=num_cols)
        numeric_tests['mean'] = df[num_cols].mean()
        numeric_tests['median'] = df[num_cols].median()
        numeric_tests['std'] = df[num_cols].std()
        numeric_tests['min'] = df[num_cols].min()
        numeric_tests['max'] = df[num_cols].max()
        numeric_tests['skewness'] = df[num_cols].skew()
        numeric_tests['kurtosis'] = df[num_cols].kurtosis()
        quality_report['numeric_tests'] = numeric_tests
    
    # Duplicates
    duplicate_count = df.duplicated().sum()
    quality_report['duplicates'] = {
        'count': duplicate_count,
        'percentage': round(duplicate_count / len(df) * 100, 2)
    }
    
    return quality_report

def suggest_cleaning_strategy(df):
    """
    Suggest data cleaning strategies for a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        dict: Dictionary with suggested cleaning strategies by column
    """
    strategies = {}
    
    # Check each column
    for col in df.columns:
        col_strategies = {}
        
        # Get data type
        dtype = df[col].dtype
        missing_count = df[col].isnull().sum()
        
        # Check for missing values
        if missing_count > 0:
            missing_pct = missing_count / len(df) * 100
            
            if pd.api.types.is_numeric_dtype(dtype):
                if missing_pct < 5:
                    col_strategies['missing_values'] = f"Replace with mean or median (recommended for {missing_pct:.2f}% missing)"
                elif missing_pct < 15:
                    col_strategies['missing_values'] = f"Consider KNN or regression imputation ({missing_pct:.2f}% missing)"
                elif missing_pct < 30:
                    col_strategies['missing_values'] = f"Consider creating a 'missing' indicator and then impute ({missing_pct:.2f}% missing)"
                else:
                    col_strategies['missing_values'] = f"Consider dropping the column ({missing_pct:.2f}% missing is significant)"
            
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                if missing_pct < 5:
                    col_strategies['missing_values'] = f"Replace with mode or a placeholder like 'Unknown' ({missing_pct:.2f}% missing)"
                elif missing_pct < 15:
                    col_strategies['missing_values'] = f"Consider using a placeholder or creating a 'missing' category ({missing_pct:.2f}% missing)"
                elif missing_pct < 30:
                    col_strategies['missing_values'] = f"Consider creating a 'missing' indicator ({missing_pct:.2f}% missing)"
                else:
                    col_strategies['missing_values'] = f"Consider dropping the column ({missing_pct:.2f}% missing is significant)"
            
            elif pd.api.types.is_datetime64_dtype(dtype):
                if missing_pct < 10:
                    col_strategies['missing_values'] = f"Forward or backward fill from adjacent values ({missing_pct:.2f}% missing)"
                else:
                    col_strategies['missing_values'] = f"Consider creating a 'missing' indicator or dropping based on analysis importance ({missing_pct:.2f}% missing)"
        
        # Check for outliers in numerical columns
        if pd.api.types.is_numeric_dtype(dtype):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            if len(outliers) > 0:
                outlier_pct = len(outliers) / len(df) * 100
                if outlier_pct < 1:
                    col_strategies['outliers'] = f"Minor outliers detected ({outlier_pct:.2f}%). Consider capping/flooring at 3 std devs."
                elif outlier_pct < 5:
                    col_strategies['outliers'] = f"Moderate outliers ({outlier_pct:.2f}%). Consider capping/flooring or winsorization."
                else:
                    col_strategies['outliers'] = f"Significant outliers ({outlier_pct:.2f}%). Examine distribution - possible transformation needed."
        
        # Check for potential data type issues
        if pd.api.types.is_object_dtype(dtype):
            # Check if should be datetime
            if col.lower().find('date') != -1 or col.lower().find('time') != -1:
                col_strategies['data_type'] = "Consider converting to datetime format"
            
            # Check if should be numeric
            numeric_pct = df[col].str.isnumeric().mean() if hasattr(df[col].str, 'isnumeric') else 0
            if numeric_pct > 0.5:
                col_strategies['data_type'] = "Column may contain numeric values. Consider type conversion."
            
            # Check if should be categorical
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05:
                col_strategies['data_type'] = f"Low cardinality ({df[col].nunique()} unique values). Consider converting to categorical."
        
        # Check if numeric column has too many zeros
        if pd.api.types.is_numeric_dtype(dtype):
            zero_pct = (df[col] == 0).mean() * 100
            if zero_pct > 30:
                col_strategies['zeros'] = f"High percentage of zeros ({zero_pct:.2f}%). Consider log transformation or zero-inflated models."
        
        # Add strategies to the dictionary if any were found
        if col_strategies:
            strategies[col] = col_strategies
    
    return strategies

def clean_data(df, cleaning_settings):
    """
    Clean the data based on the provided settings.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        cleaning_settings (dict): Dictionary with cleaning settings
        
    Returns:
        tuple: (cleaned_df, cleaning_log)
    """
    # Create a copy of the DataFrame to avoid modifying the original
    cleaned_df = df.copy()
    
    # Initialize cleaning log
    cleaning_log = {}
    
    # Handle duplicates
    if cleaning_settings.get('handle_duplicates', False):
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_df)
        
        cleaning_log['duplicates'] = f"Removed {removed_rows} duplicate rows ({removed_rows/initial_rows*100:.2f}% of data)"
    
    # Handle missing values
    missing_strategy = cleaning_settings.get('handle_missing', 'Drop rows')
    
    # Get columns with missing values
    cols_with_missing = cleaned_df.columns[cleaned_df.isnull().any()].tolist()
    
    if cols_with_missing:
        missing_log = {}
        
        for col in cols_with_missing:
            # Check if column has custom cleaning settings
            custom_settings = cleaning_settings.get('custom_columns', {}).get(col, {})
            col_strategy = custom_settings.get('missing_strategy', missing_strategy)
            
            # Get the original missing count
            missing_count = cleaned_df[col].isnull().sum()
            
            # Apply the appropriate strategy
            if col_strategy == "Drop rows":
                initial_rows = len(cleaned_df)
                cleaned_df = cleaned_df.dropna(subset=[col])
                removed_rows = initial_rows - len(cleaned_df)
                missing_log[col] = f"Dropped {removed_rows} rows with missing values"
            
            elif col_strategy == "Fill with mean/mode":
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                    missing_log[col] = f"Filled {missing_count} missing values with mean ({cleaned_df[col].mean():.2f})"
                else:
                    # For categorical, use mode
                    mode_value = cleaned_df[col].mode()[0]
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                    missing_log[col] = f"Filled {missing_count} missing values with mode ('{mode_value}')"
            
            elif col_strategy == "Fill with median":
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    missing_log[col] = f"Filled {missing_count} missing values with median ({cleaned_df[col].median():.2f})"
                else:
                    # For non-numeric, revert to mode
                    mode_value = cleaned_df[col].mode()[0]
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                    missing_log[col] = f"Column not numeric. Filled {missing_count} missing values with mode ('{mode_value}')"
            
            elif col_strategy == "Forward fill":
                cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                remaining_missing = cleaned_df[col].isnull().sum()
                missing_log[col] = f"Forward filled {missing_count - remaining_missing} missing values. {remaining_missing} still missing."
                
                # If values still missing, fill with backward fill
                if remaining_missing > 0:
                    cleaned_df[col] = cleaned_df[col].fillna(method='bfill')
                    final_missing = cleaned_df[col].isnull().sum()
                    missing_log[col] += f" Backward filled remaining, {final_missing} still missing."
            
            elif col_strategy == "Backward fill":
                cleaned_df[col] = cleaned_df[col].fillna(method='bfill')
                remaining_missing = cleaned_df[col].isnull().sum()
                missing_log[col] = f"Backward filled {missing_count - remaining_missing} missing values. {remaining_missing} still missing."
                
                # If values still missing, fill with forward fill
                if remaining_missing > 0:
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                    final_missing = cleaned_df[col].isnull().sum()
                    missing_log[col] += f" Forward filled remaining, {final_missing} still missing."
            
            elif col_strategy == "Custom value":
                if 'fill_value' in custom_settings:
                    fill_value = custom_settings['fill_value']
                else:
                    fill_value = cleaning_settings.get('fill_value', '0')
                
                # Convert fill value to appropriate type
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    try:
                        fill_value = float(fill_value)
                    except ValueError:
                        fill_value = 0
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                missing_log[col] = f"Filled {missing_count} missing values with custom value: {fill_value}"
            
            elif col_strategy == "Mean":
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                    missing_log[col] = f"Filled {missing_count} missing values with mean ({cleaned_df[col].mean():.2f})"
                else:
                    # For non-numeric, revert to mode
                    mode_value = cleaned_df[col].mode()[0]
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                    missing_log[col] = f"Column not numeric. Filled {missing_count} missing values with mode ('{mode_value}')"
            
            elif col_strategy == "Median":
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    missing_log[col] = f"Filled {missing_count} missing values with median ({cleaned_df[col].median():.2f})"
                else:
                    # For non-numeric, revert to mode
                    mode_value = cleaned_df[col].mode()[0]
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                    missing_log[col] = f"Column not numeric. Filled {missing_count} missing values with mode ('{mode_value}')"
            
            elif col_strategy == "Mode":
                mode_value = cleaned_df[col].mode()[0]
                cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                missing_log[col] = f"Filled {missing_count} missing values with mode ('{mode_value}')"
            
            elif col_strategy == "Empty string":
                cleaned_df[col] = cleaned_df[col].fillna('')
                missing_log[col] = f"Filled {missing_count} missing values with empty string"
            
            elif col_strategy == "Current date":
                current_date = datetime.now().strftime('%Y-%m-%d')
                cleaned_df[col] = cleaned_df[col].fillna(current_date)
                missing_log[col] = f"Filled {missing_count} missing values with current date ({current_date})"
        
        cleaning_log['missing_values'] = missing_log
    
    # Handle outliers
    if cleaning_settings.get('handle_outliers', 'Keep all') != 'Keep all':
        outlier_strategy = cleaning_settings.get('handle_outliers')
        outlier_log = {}
        
        # Process numerical columns for outliers
        for col in cleaned_df.select_dtypes(include=['number']).columns:
            # Calculate bounds
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers_mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                if outlier_strategy == "Remove outliers":
                    initial_rows = len(cleaned_df)
                    cleaned_df = cleaned_df[~outliers_mask]
                    removed_rows = initial_rows - len(cleaned_df)
                    outlier_log[col] = f"Removed {removed_rows} rows with outliers (outside {lower_bound:.2f} to {upper_bound:.2f})"
                
                elif outlier_strategy == "Cap outliers":
                    # Cap the values at the bounds
                    cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
                    cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
                    outlier_log[col] = f"Capped {outlier_count} outliers to range [{lower_bound:.2f}, {upper_bound:.2f}]"
        
        if outlier_log:
            cleaning_log['outliers'] = outlier_log
    
    # Data type validation and conversion
    if cleaning_settings.get('data_validation', False):
        dtype_log = {}
        
        # Check if date columns are properly formatted
        for col in cleaned_df.columns:
            col_lower = col.lower()
            
            # Check for date columns
            if 'date' in col_lower or 'time' in col_lower:
                if not pd.api.types.is_datetime64_dtype(cleaned_df[col]):
                    try:
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
                        dtype_log[col] = f"Converted to datetime format"
                    except:
                        dtype_log[col] = f"Failed to convert to datetime"
            
            # Check for potential numeric columns stored as strings
            elif pd.api.types.is_object_dtype(cleaned_df[col]):
                # Try to convert to numeric if it looks numeric
                try:
                    numeric_col = pd.to_numeric(cleaned_df[col], errors='coerce')
                    # If more than 80% of values converted successfully, keep the conversion
                    if numeric_col.notnull().mean() > 0.8:
                        cleaned_df[col] = numeric_col
                        dtype_log[col] = f"Converted from string to numeric"
                except:
                    pass
        
        if dtype_log:
            cleaning_log['data_type_conversion'] = dtype_log
    
    # Normalize numerical columns if requested
    if cleaning_settings.get('normalize_data', False):
        norm_log = {}
        
        # Get numerical columns
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            # Using MinMaxScaler to scale values between 0 and 1
            scaler = MinMaxScaler()
            cleaned_df[numeric_cols] = scaler.fit_transform(cleaned_df[numeric_cols])
            
            norm_log['columns'] = f"Normalized {len(numeric_cols)} columns using MinMaxScaler (0-1 range)"
            norm_log['details'] = f"Affected columns: {', '.join(numeric_cols)}"
            
            cleaning_log['normalization'] = norm_log
    
    # Final check for any remaining missing values
    remaining_missing = cleaned_df.isnull().sum().sum()
    if remaining_missing > 0:
        cleaning_log['remaining_issues'] = f"There are still {remaining_missing} missing values in the dataset"
    
    return cleaned_df, cleaning_log
