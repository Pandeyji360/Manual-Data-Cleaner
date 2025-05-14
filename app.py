import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import time
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from pathlib import Path
import dask.dataframe as dd

# Import custom modules
from data_cleaning import (
    identify_data_issues, 
    clean_data, 
    get_data_quality_report,
    suggest_cleaning_strategy
)
from data_visualization import (
    plot_missing_data,
    plot_data_distribution,
    plot_categorical_data,
    plot_correlation_matrix,
    plot_outliers,
    plot_cleaned_vs_original
)
from utils import (
    get_data_summary,
    convert_df_to_csv,
    get_file_size,
    is_large_file
)
from logger import setup_logger, log_action

# Set up logger
logger = setup_logger()

# Set page config
st.set_page_config(
    page_title="Automated Data Analysis Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'original_data' not in st.session_state:
    st.session_state.original_data = {}
if 'cleaning_logs' not in st.session_state:
    st.session_state.cleaning_logs = {}
if 'file_stats' not in st.session_state:
    st.session_state.file_stats = {}
if 'data_quality_reports' not in st.session_state:
    st.session_state.data_quality_reports = {}
if 'show_logs' not in st.session_state:
    st.session_state.show_logs = False

def main():
    # Display header with background image
    st.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h1>📊 Automated Data Analysis Pipeline</h1>
            <p>Advanced data cleaning, visualization, and real-time monitoring for large datasets</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Display dashboard image
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1542744173-05336fcc7ad4", use_column_width=True)
    
    # Sidebar for file upload and selection
    with st.sidebar:
        st.header("Data Upload & Selection")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload one or more CSV files", 
            type=["csv"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Process each uploaded file
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                
                # Only process files that haven't been processed yet
                if file_name not in st.session_state.uploaded_files:
                    file_size = get_file_size(uploaded_file)
                    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Log the file upload
                    log_message = f"File uploaded: {file_name} ({file_size})"
                    logger.info(log_message)
                    log_action(log_message)
                    
                    # Store file information
                    st.session_state.uploaded_files[file_name] = {
                        'file': uploaded_file,
                        'size': file_size,
                        'upload_time': upload_time
                    }
                    
                    # Check if file is large (>100MB)
                    if is_large_file(uploaded_file):
                        st.session_state.uploaded_files[file_name]['large_file'] = True
                        st.info(f"⚠️ {file_name} is a large file. Using Dask for processing.")
                    else:
                        st.session_state.uploaded_files[file_name]['large_file'] = False
            
            # File selector
            if st.session_state.uploaded_files:
                st.subheader("Select a file to analyze")
                selected_file = st.selectbox(
                    "Choose a file", 
                    list(st.session_state.uploaded_files.keys()),
                    index=0
                )
                
                if st.button("Load Selected File"):
                    with st.spinner(f"Loading {selected_file}..."):
                        # Set the current file
                        st.session_state.current_file = selected_file
                        file_info = st.session_state.uploaded_files[selected_file]
                        
                        # Reset file data
                        if selected_file in st.session_state.processed_data:
                            del st.session_state.processed_data[selected_file]
                        if selected_file in st.session_state.original_data:
                            del st.session_state.original_data[selected_file]
                        
                        # Load the file
                        try:
                            file_info['file'].seek(0)
                            
                            # Use Dask for large files
                            if file_info.get('large_file', False):
                                # Save to temp file first for Dask
                                temp_file = f"temp_{selected_file}"
                                with open(temp_file, 'wb') as f:
                                    f.write(file_info['file'].getvalue())
                                
                                df = dd.read_csv(temp_file).compute()
                                os.remove(temp_file)  # Clean up temp file
                            else:
                                df = pd.read_csv(file_info['file'])
                            
                            # Store original data
                            st.session_state.original_data[selected_file] = df.copy()
                            
                            # Generate data quality report
                            data_quality_report = get_data_quality_report(df)
                            st.session_state.data_quality_reports[selected_file] = data_quality_report
                            
                            # Log the successful load
                            log_message = f"File loaded for analysis: {selected_file}"
                            logger.info(log_message)
                            log_action(log_message)
                            
                            # Record file stats
                            st.session_state.file_stats[selected_file] = {
                                'rows': len(df),
                                'columns': len(df.columns),
                                'size': file_info['size'],
                                'load_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            st.success(f"✅ File {selected_file} loaded successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            error_msg = f"Error loading file {selected_file}: {str(e)}"
                            logger.error(error_msg)
                            log_action(error_msg)
                            st.error(error_msg)
        
        # Show logs toggle
        st.subheader("Logs")
        if st.checkbox("Show processing logs", value=st.session_state.show_logs):
            st.session_state.show_logs = True
        else:
            st.session_state.show_logs = False
            
    # Main content area
    if st.session_state.current_file:
        selected_file = st.session_state.current_file
        
        # Get the data
        if selected_file in st.session_state.original_data:
            df = st.session_state.original_data[selected_file]
            
            # Tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Data Overview", 
                "🔍 Data Quality Analysis", 
                "🧹 Data Cleaning", 
                "📈 Visualization", 
                "📝 Processing Logs"
            ])
            
            # Tab 1: Data Overview
            with tab1:
                st.header(f"Data Overview: {selected_file}")
                
                # File information
                file_stats = st.session_state.file_stats[selected_file]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rows", f"{file_stats['rows']:,}")
                
                with col2:
                    st.metric("Columns", file_stats['columns'])
                
                with col3:
                    st.metric("File Size", file_stats['size'])
                
                with col4:
                    st.metric("Loaded At", file_stats['load_time'])
                
                # Data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data summary
                st.subheader("Data Summary")
                summary = get_data_summary(df)
                st.dataframe(summary, use_container_width=True)
                
                # Data types
                st.subheader("Data Types")
                dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
                dtypes_df.reset_index(inplace=True)
                dtypes_df.columns = ['Column', 'Data Type']
                st.dataframe(dtypes_df, use_container_width=True)
            
            # Tab 2: Data Quality Analysis
            with tab2:
                st.header("Data Quality Analysis")
                
                # Data quality overview
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Check for data issues
                    issues = identify_data_issues(df)
                    
                    st.subheader("Data Quality Issues")
                    
                    # Missing values
                    missing_count = df.isna().sum().sum()
                    missing_percent = (missing_count / (df.shape[0] * df.shape[1])) * 100
                    
                    # Duplicate rows
                    duplicate_count = df.duplicated().sum()
                    duplicate_percent = (duplicate_count / df.shape[0]) * 100
                    
                    # Outliers count (simplified)
                    outlier_count = sum([len(issues.get('outliers', {}).get(col, [])) for col in df.select_dtypes(include=['number']).columns])
                    
                    # Display metrics
                    st.metric("Missing Values", f"{missing_count:,} ({missing_percent:.2f}%)")
                    st.metric("Duplicate Rows", f"{duplicate_count:,} ({duplicate_percent:.2f}%)")
                    st.metric("Potential Outliers", f"{outlier_count:,}")
                    
                    # Data quality score (simple implementation)
                    columns_with_issues = len([col for col in issues if issues[col]])
                    quality_score = 100 - (columns_with_issues / len(df.columns) * 100)
                    
                    # Display quality score with color
                    if quality_score >= 80:
                        st.success(f"Data Quality Score: {quality_score:.2f}%")
                    elif quality_score >= 50:
                        st.warning(f"Data Quality Score: {quality_score:.2f}%")
                    else:
                        st.error(f"Data Quality Score: {quality_score:.2f}%")
                
                with col2:
                    # Plot missing data
                    st.subheader("Missing Data Visualization")
                    missing_fig = plot_missing_data(df)
                    st.plotly_chart(missing_fig, use_container_width=True)
                
                # Display detailed issues
                st.subheader("Detailed Data Issues")
                if issues:
                    for issue_type, columns in issues.items():
                        if columns:
                            with st.expander(f"{issue_type.title()} Issues"):
                                for col, details in columns.items():
                                    if isinstance(details, list) and len(details) > 0:
                                        st.write(f"**{col}**: {len(details)} issues detected")
                                        st.write(f"Example values: {', '.join(str(x) for x in details[:5])}")
                                    elif isinstance(details, dict):
                                        st.write(f"**{col}**: Issue details: {details}")
                                    else:
                                        st.write(f"**{col}**: Issue detected")
                else:
                    st.success("No significant data issues detected!")
                
                # Display data quality report
                if selected_file in st.session_state.data_quality_reports:
                    with st.expander("Full Data Quality Report"):
                        quality_report = st.session_state.data_quality_reports[selected_file]
                        for section, content in quality_report.items():
                            st.subheader(section)
                            
                            # Handle different types of content correctly
                            if section == 'categorical_distributions':
                                # This is a nested dictionary structure
                                for col_name, df_data in content.items():
                                    st.write(f"**{col_name}**")
                                    st.dataframe(df_data, use_container_width=True)
                            else:
                                # Regular dataframe
                                st.dataframe(content, use_container_width=True)
            
            # Tab 3: Data Cleaning
            with tab3:
                st.header("Data Cleaning")
                
                # Cleaning strategy recommendation
                st.subheader("Recommended Cleaning Strategy")
                cleaning_strategies = suggest_cleaning_strategy(df)
                
                for col, strategies in cleaning_strategies.items():
                    if strategies:
                        with st.expander(f"Recommendations for '{col}'"):
                            for strategy_type, details in strategies.items():
                                st.write(f"**{strategy_type}**: {details}")
                
                # Cleaning options
                st.subheader("Cleaning Options")
                
                # Cleaning settings
                col1, col2 = st.columns(2)
                
                with col1:
                    handle_missing = st.selectbox(
                        "Handle Missing Values",
                        ["Drop rows", "Fill with mean/mode", "Fill with median", "Forward fill", "Backward fill", "Custom value"]
                    )
                    
                    if handle_missing == "Custom value":
                        fill_value = st.text_input("Fill Value", "0")
                    
                    handle_duplicates = st.checkbox("Remove duplicate rows", value=True)
                
                with col2:
                    handle_outliers = st.selectbox(
                        "Handle Outliers",
                        ["Keep all", "Remove outliers", "Cap outliers"]
                    )
                    
                    data_validation = st.checkbox("Validate data types", value=True)
                    normalize_data = st.checkbox("Normalize numerical columns", value=False)
                
                # Custom cleaning for specific columns
                with st.expander("Custom Column Cleaning"):
                    # Let user select columns to apply custom cleaning
                    selected_columns = st.multiselect(
                        "Select columns for custom cleaning",
                        df.columns.tolist()
                    )
                    
                    if selected_columns:
                        for col in selected_columns:
                            st.write(f"**Column: {col}**")
                            
                            # Get column data type
                            col_type = df[col].dtype
                            
                            if pd.api.types.is_numeric_dtype(col_type):
                                st.write("Numeric column")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    custom_missing = st.selectbox(
                                        f"Handle missing in '{col}'",
                                        ["Default", "Mean", "Median", "Custom value"]
                                    )
                                
                                with col2:
                                    if custom_missing == "Custom value":
                                        custom_value = st.number_input(f"Fill value for '{col}'", value=0.0)
                            
                            elif pd.api.types.is_string_dtype(col_type):
                                st.write("Text column")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    custom_missing = st.selectbox(
                                        f"Handle missing in '{col}'",
                                        ["Default", "Mode", "Empty string", "Custom value"]
                                    )
                                
                                with col2:
                                    if custom_missing == "Custom value":
                                        custom_value = st.text_input(f"Fill value for '{col}'", "unknown")
                            
                            elif pd.api.types.is_datetime64_dtype(col_type):
                                st.write("Date column")
                                custom_missing = st.selectbox(
                                    f"Handle missing in '{col}'",
                                    ["Default", "Forward fill", "Backward fill", "Current date"]
                                )
                
                # Run cleaning process
                if st.button("Run Data Cleaning"):
                    with st.spinner("Cleaning data..."):
                        # Create cleaning settings dictionary
                        cleaning_settings = {
                            'handle_missing': handle_missing,
                            'handle_duplicates': handle_duplicates,
                            'handle_outliers': handle_outliers,
                            'data_validation': data_validation,
                            'normalize_data': normalize_data,
                        }
                        
                        if handle_missing == "Custom value":
                            cleaning_settings['fill_value'] = fill_value
                        
                        # Add custom column cleaning settings
                        if selected_columns:
                            cleaning_settings['custom_columns'] = {}
                            for col in selected_columns:
                                col_type = df[col].dtype
                                
                                if 'custom_missing' in locals():
                                    if custom_missing != "Default":
                                        if 'custom_value' in locals() and custom_missing == "Custom value":
                                            cleaning_settings['custom_columns'][col] = {
                                                'missing_strategy': custom_missing,
                                                'fill_value': custom_value
                                            }
                                        else:
                                            cleaning_settings['custom_columns'][col] = {
                                                'missing_strategy': custom_missing
                                            }
                        
                        # Run cleaning process
                        try:
                            start_time = time.time()
                            cleaned_df, cleaning_log = clean_data(df, cleaning_settings)
                            end_time = time.time()
                            
                            # Store cleaned data
                            st.session_state.processed_data[selected_file] = cleaned_df
                            
                            # Store cleaning log
                            st.session_state.cleaning_logs[selected_file] = cleaning_log
                            
                            # Log the cleaning action
                            log_message = f"Data cleaned for {selected_file}. Processing time: {end_time - start_time:.2f} seconds"
                            logger.info(log_message)
                            log_action(log_message)
                            
                            st.success("✅ Data cleaning completed successfully!")
                            
                            # Display comparison
                            st.subheader("Before & After Cleaning")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Before Cleaning**")
                                st.write(f"Total rows: {len(df)}")
                                st.write(f"Missing values: {df.isna().sum().sum()}")
                                st.write(f"Duplicate rows: {df.duplicated().sum()}")
                            
                            with col2:
                                st.write("**After Cleaning**")
                                st.write(f"Total rows: {len(cleaned_df)}")
                                st.write(f"Missing values: {cleaned_df.isna().sum().sum()}")
                                st.write(f"Duplicate rows: {cleaned_df.duplicated().sum()}")
                            
                            # Show comparison chart
                            comparison_fig = plot_cleaned_vs_original(df, cleaned_df)
                            st.plotly_chart(comparison_fig, use_container_width=True)
                            
                            # Option to download cleaned data
                            csv = convert_df_to_csv(cleaned_df)
                            file_name = f"cleaned_{selected_file}"
                            
                            st.download_button(
                                label=f"Download Cleaned Data as CSV",
                                data=csv,
                                file_name=file_name,
                                mime='text/csv',
                            )
                            
                        except Exception as e:
                            error_msg = f"Error during data cleaning: {str(e)}"
                            logger.error(error_msg)
                            log_action(error_msg)
                            st.error(error_msg)
                
                # Show cleaning log if available
                if selected_file in st.session_state.cleaning_logs:
                    with st.expander("View Cleaning Log"):
                        cleaning_log = st.session_state.cleaning_logs[selected_file]
                        for step, details in cleaning_log.items():
                            st.write(f"**{step}**")
                            st.write(details)
            
            # Tab 4: Visualization
            with tab4:
                st.header("Data Visualization")
                
                # Choose dataset for visualization
                data_version = st.radio(
                    "Select data version for visualization:",
                    ["Original Data", "Cleaned Data"]
                )
                
                # Get the appropriate dataframe
                if data_version == "Cleaned Data" and selected_file in st.session_state.processed_data:
                    viz_df = st.session_state.processed_data[selected_file]
                else:
                    viz_df = df
                    if data_version == "Cleaned Data":
                        st.warning("Cleaned data not available. Using original data for visualization.")
                
                # Visualization options
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution plots
                    st.subheader("Distribution Analysis")
                    
                    numeric_cols = viz_df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        dist_col = st.selectbox("Select column for distribution analysis", numeric_cols)
                        
                        dist_plot_type = st.selectbox(
                            "Select plot type",
                            ["Histogram", "Box Plot", "Violin Plot", "KDE Plot"]
                        )
                        
                        if dist_plot_type == "Histogram":
                            fig = px.histogram(viz_df, x=dist_col, title=f"Histogram of {dist_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif dist_plot_type == "Box Plot":
                            fig = px.box(viz_df, y=dist_col, title=f"Box Plot of {dist_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif dist_plot_type == "Violin Plot":
                            fig = px.violin(viz_df, y=dist_col, title=f"Violin Plot of {dist_col}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif dist_plot_type == "KDE Plot":
                            # Using matplotlib for KDE plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.kdeplot(data=viz_df[dist_col].dropna(), ax=ax)
                            ax.set_title(f"KDE Plot of {dist_col}")
                            st.pyplot(fig)
                    else:
                        st.info("No numeric columns available for distribution analysis.")
                
                with col2:
                    # Categorical plots
                    st.subheader("Categorical Analysis")
                    
                    # Get categorical columns (including object and category dtype)
                    cat_cols = viz_df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    # Add low-cardinality numeric columns as potential categorical
                    for col in viz_df.select_dtypes(include=['number']).columns:
                        if viz_df[col].nunique() < 20:  # Arbitrary threshold
                            cat_cols.append(col)
                    
                    if cat_cols:
                        cat_col = st.selectbox("Select column for categorical analysis", cat_cols)
                        
                        # Get value counts and limit to top categories
                        val_counts = viz_df[cat_col].value_counts().head(20)  # Limit to top 20
                        
                        cat_plot_type = st.selectbox(
                            "Select plot type",
                            ["Bar Chart", "Pie Chart", "Treemap"]
                        )
                        
                        if cat_plot_type == "Bar Chart":
                            fig = px.bar(
                                val_counts, 
                                x=val_counts.index, 
                                y=val_counts.values,
                                title=f"Bar Chart of {cat_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif cat_plot_type == "Pie Chart":
                            fig = px.pie(
                                names=val_counts.index, 
                                values=val_counts.values,
                                title=f"Pie Chart of {cat_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif cat_plot_type == "Treemap":
                            fig = px.treemap(
                                names=val_counts.index, 
                                values=val_counts.values,
                                title=f"Treemap of {cat_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No categorical columns available for analysis.")
                
                # Correlation analysis
                st.subheader("Correlation Analysis")
                
                numeric_cols = viz_df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) >= 2:
                    corr_fig = plot_correlation_matrix(viz_df[numeric_cols])
                    st.plotly_chart(corr_fig, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for correlation analysis.")
                
                # Advanced visualization options
                with st.expander("Advanced Visualization"):
                    viz_type = st.selectbox(
                        "Select visualization type",
                        [
                            "Scatter Plot", 
                            "Line Chart", 
                            "Bubble Chart", 
                            "Heatmap", 
                            "Parallel Coordinates",
                            "3D Scatter Plot"
                        ]
                    )
                    
                    if viz_type == "Scatter Plot":
                        if len(numeric_cols) >= 2:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                x_col = st.selectbox("X-axis column", numeric_cols, index=0)
                            
                            with col2:
                                y_col = st.selectbox("Y-axis column", numeric_cols, index=min(1, len(numeric_cols)-1))
                            
                            with col3:
                                color_col = st.selectbox("Color by (optional)", ["None"] + viz_df.columns.tolist())
                            
                            if color_col == "None":
                                fig = px.scatter(viz_df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                            else:
                                fig = px.scatter(viz_df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col} (colored by {color_col})")
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Need at least 2 numeric columns for scatter plot.")
                    
                    elif viz_type == "Line Chart":
                        if len(numeric_cols) >= 1:
                            y_cols = st.multiselect("Select columns for Y-axis", numeric_cols, default=[numeric_cols[0]])
                            
                            if y_cols:
                                # Create index for X-axis
                                x_values = np.arange(len(viz_df))
                                
                                # Create line chart
                                fig = px.line(title="Line Chart")
                                
                                for col in y_cols:
                                    fig.add_scatter(x=x_values, y=viz_df[col], name=col)
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Please select at least one column for the line chart.")
                        else:
                            st.info("Need at least 1 numeric column for line chart.")
                    
                    elif viz_type == "Bubble Chart":
                        if len(numeric_cols) >= 3:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                x_col = st.selectbox("X-axis", numeric_cols, index=0)
                            
                            with col2:
                                y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
                            
                            with col3:
                                size_col = st.selectbox("Size", numeric_cols, index=min(2, len(numeric_cols)-1))
                            
                            with col4:
                                color_col = st.selectbox("Color", ["None"] + viz_df.columns.tolist())
                            
                            if color_col == "None":
                                fig = px.scatter(viz_df, x=x_col, y=y_col, size=size_col, title=f"Bubble Chart: {x_col} vs {y_col} (size: {size_col})")
                            else:
                                fig = px.scatter(viz_df, x=x_col, y=y_col, size=size_col, color=color_col, title=f"Bubble Chart: {x_col} vs {y_col} (size: {size_col}, color: {color_col})")
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Need at least 3 numeric columns for bubble chart.")
                    
                    elif viz_type == "Heatmap":
                        if len(numeric_cols) >= 2:
                            corr_data = viz_df[numeric_cols].corr()
                            
                            fig = px.imshow(
                                corr_data,
                                text_auto=True,
                                color_continuous_scale='RdBu_r',
                                title="Correlation Heatmap"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Need at least 2 numeric columns for heatmap.")
                    
                    elif viz_type == "Parallel Coordinates":
                        if len(numeric_cols) >= 3:
                            selected_cols = st.multiselect(
                                "Select columns for parallel coordinates",
                                numeric_cols,
                                default=numeric_cols[:min(5, len(numeric_cols))]
                            )
                            
                            color_col = st.selectbox(
                                "Color by",
                                ["None"] + viz_df.columns.tolist()
                            )
                            
                            if selected_cols:
                                if color_col == "None":
                                    fig = px.parallel_coordinates(
                                        viz_df, 
                                        dimensions=selected_cols,
                                        title="Parallel Coordinates Plot"
                                    )
                                else:
                                    fig = px.parallel_coordinates(
                                        viz_df, 
                                        dimensions=selected_cols,
                                        color=color_col,
                                        title=f"Parallel Coordinates Plot (colored by {color_col})"
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Please select columns for the parallel coordinates plot.")
                        else:
                            st.info("Need at least 3 numeric columns for parallel coordinates.")
                    
                    elif viz_type == "3D Scatter Plot":
                        if len(numeric_cols) >= 3:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                x_col = st.selectbox("X-axis (3D)", numeric_cols, index=0)
                            
                            with col2:
                                y_col = st.selectbox("Y-axis (3D)", numeric_cols, index=min(1, len(numeric_cols)-1))
                            
                            with col3:
                                z_col = st.selectbox("Z-axis", numeric_cols, index=min(2, len(numeric_cols)-1))
                            
                            with col4:
                                color_col = st.selectbox("Color (3D)", ["None"] + viz_df.columns.tolist())
                            
                            if color_col == "None":
                                fig = px.scatter_3d(
                                    viz_df, 
                                    x=x_col, 
                                    y=y_col, 
                                    z=z_col,
                                    title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}"
                                )
                            else:
                                fig = px.scatter_3d(
                                    viz_df, 
                                    x=x_col, 
                                    y=y_col, 
                                    z=z_col,
                                    color=color_col,
                                    title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col} (colored by {color_col})"
                                )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Need at least 3 numeric columns for 3D scatter plot.")
            
            # Tab 5: Processing Logs
            with tab5:
                st.header("Processing Logs")
                
                # Display logs
                if 'logs' in st.session_state and st.session_state.logs:
                    logs = st.session_state.logs
                    
                    # Filter logs for current file
                    file_logs = [log for log in logs if selected_file in log['message']]
                    
                    if file_logs:
                        log_df = pd.DataFrame(file_logs)
                        st.dataframe(log_df, use_container_width=True)
                    else:
                        st.info(f"No logs available for {selected_file}")
                else:
                    st.info("No processing logs available")
                
                # Display cleaning logs if available
                if selected_file in st.session_state.cleaning_logs:
                    st.subheader("Data Cleaning Log")
                    cleaning_log = st.session_state.cleaning_logs[selected_file]
                    
                    # Convert cleaning log to DataFrame for display
                    log_items = []
                    for step, details in cleaning_log.items():
                        log_items.append({
                            'Step': step,
                            'Details': details
                        })
                    
                    if log_items:
                        log_df = pd.DataFrame(log_items)
                        st.dataframe(log_df, use_container_width=True)
        
        else:
            st.warning(f"Data for {selected_file} not loaded. Please load the file first.")
    
    # Display logs at the bottom if enabled
    if st.session_state.show_logs:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Processing Logs")
        
        if 'logs' in st.session_state and st.session_state.logs:
            for log in st.session_state.logs[-10:]:  # Show last 10 logs
                st.sidebar.write(f"**{log['timestamp']}**: {log['message']}")
        else:
            st.sidebar.write("No logs available")

# Initialize the logs list if it doesn't exist
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Custom function to log actions and store in session state
def log_action(message):
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'message': message
    }
    st.session_state.logs.append(log_entry)

if __name__ == "__main__":
    main()
