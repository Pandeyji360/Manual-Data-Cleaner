#!/usr/bin/env python3
"""
Desktop Application for Data Analysis
------------------------------------
This script provides a desktop GUI for the data analysis pipeline.
It offers the same functionality as the web version but in a native desktop application.

Usage:
    python desktop_app.py

Author: Senior Data Engineer
Date: May 1, 2025
Version: 1.0
"""

import os
import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox, 
                            QTableWidget, QTableWidgetItem, QProgressBar, QCheckBox,
                            QSpinBox, QDoubleSpinBox, QTextEdit, QSplitter, QMessageBox,
                            QScrollArea, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon

# Import functionality from our modules
from data_cleaning import identify_data_issues, get_data_quality_report, suggest_cleaning_strategy, clean_data
from data_visualization import (plot_missing_data, plot_data_distribution, plot_categorical_data, 
                              plot_correlation_matrix, plot_outliers, plot_cleaned_vs_original)
from utils import get_data_summary, get_memory_usage, get_file_size, is_large_file
from logger import setup_logger, log_action

# Set up logging
logger = setup_logger()

# Import matplotlib for embedding plots in PyQt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# For converting Plotly charts to Matplotlib
import plotly.io as pio

class MatplotlibCanvas(FigureCanvas):
    """Class to represent the FigureCanvas widget"""
    def __init__(self, figure=None):
        if figure is None:
            figure = Figure(figsize=(5, 4), dpi=100)
        self.figure = figure
        super(MatplotlibCanvas, self).__init__(self.figure)
        self.setSizePolicy(QApplication.style().objectName() != "windows" and "Expanding" or "Preferred", "Expanding")
        self.updateGeometry()

class DataProcessingWorker(QThread):
    """Worker thread for data processing tasks"""
    update_progress = pyqtSignal(int)
    update_log = pyqtSignal(str)
    finished_task = pyqtSignal(dict)
    
    def __init__(self, task_type, data, parameters=None):
        super().__init__()
        self.task_type = task_type
        self.data = data
        self.parameters = parameters or {}
        
    def run(self):
        try:
            if self.task_type == 'identify_issues':
                # Update progress
                self.update_progress.emit(10)
                self.update_log.emit("Starting data quality analysis...")
                
                # Identify data issues
                self.update_log.emit("Identifying data issues...")
                issues = identify_data_issues(self.data)
                self.update_progress.emit(40)
                
                # Get data quality report
                self.update_log.emit("Generating data quality report...")
                quality_report = get_data_quality_report(self.data)
                self.update_progress.emit(70)
                
                # Suggest cleaning strategy
                self.update_log.emit("Suggesting cleaning strategies...")
                cleaning_strategy = suggest_cleaning_strategy(self.data)
                self.update_progress.emit(90)
                
                # Compile results
                results = {
                    'issues': issues,
                    'quality_report': quality_report,
                    'cleaning_strategy': cleaning_strategy
                }
                
                self.update_log.emit("Data quality analysis complete!")
                self.update_progress.emit(100)
                self.finished_task.emit(results)
                
            elif self.task_type == 'clean_data':
                cleaning_settings = self.parameters.get('cleaning_settings', {})
                
                self.update_progress.emit(10)
                self.update_log.emit("Starting data cleaning process...")
                
                # Clean the data
                self.update_log.emit("Applying cleaning operations...")
                cleaned_df, cleaning_log = clean_data(self.data, cleaning_settings)
                self.update_progress.emit(70)
                
                self.update_log.emit("Finalizing results...")
                results = {
                    'cleaned_df': cleaned_df,
                    'cleaning_log': cleaning_log
                }
                
                self.update_log.emit("Data cleaning complete!")
                self.update_progress.emit(100)
                self.finished_task.emit(results)
                
            # Add other task types as needed
                
        except Exception as e:
            self.update_log.emit(f"Error in processing: {str(e)}")
            log_action(f"Error in processing: {str(e)}", level="ERROR")
            self.finished_task.emit({'error': str(e)})

class DataAnalysisApp(QMainWindow):
    """Main window class for the data analysis application"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analysis Pipeline")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data storage
        self.dataframes = {}
        self.current_df = None
        self.cleaned_df = None
        self.current_file = None
        
        # Set up the UI
        self.setup_ui()
        
        # Log application start
        log_action("Desktop application started")
        
    def setup_ui(self):
        """Set up the user interface"""
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        main_layout = QVBoxLayout(self.central_widget)
        
        # Add header
        header_layout = QHBoxLayout()
        title_label = QLabel("Data Analysis Pipeline")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Add memory usage indicator
        self.memory_label = QLabel("Memory Usage: N/A")
        header_layout.addWidget(self.memory_label)
        
        # Add refresh button
        refresh_btn = QPushButton("Refresh Memory")
        refresh_btn.clicked.connect(self.update_memory_usage)
        header_layout.addWidget(refresh_btn)
        
        main_layout.addLayout(header_layout)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Add tabs
        self.setup_data_loading_tab()
        self.setup_data_cleaning_tab()
        self.setup_data_visualization_tab()
        self.setup_analysis_tab()
        
        main_layout.addWidget(self.tabs)
        
        # Add status bar with log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(100)
        main_layout.addWidget(QLabel("Log:"))
        main_layout.addWidget(self.log_display)
        
        # Update memory usage on startup
        self.update_memory_usage()
    
    def setup_data_loading_tab(self):
        """Set up the data loading tab"""
        data_loading_tab = QWidget()
        layout = QVBoxLayout(data_loading_tab)
        
        # File selection section
        file_section = QFrame()
        file_section.setFrameShape(QFrame.StyledPanel)
        file_layout = QVBoxLayout(file_section)
        
        file_label = QLabel("Select CSV Files for Analysis:")
        file_label.setFont(QFont("Arial", 12))
        file_layout.addWidget(file_label)
        
        file_buttons_layout = QHBoxLayout()
        
        # Single file selection
        select_file_btn = QPushButton("Select Single File")
        select_file_btn.clicked.connect(self.load_single_file)
        file_buttons_layout.addWidget(select_file_btn)
        
        # Multiple file selection
        select_multiple_btn = QPushButton("Select Multiple Files")
        select_multiple_btn.clicked.connect(self.load_multiple_files)
        file_buttons_layout.addWidget(select_multiple_btn)
        
        # Select directory
        select_dir_btn = QPushButton("Select Directory")
        select_dir_btn.clicked.connect(self.load_directory)
        file_buttons_layout.addWidget(select_dir_btn)
        
        file_buttons_layout.addStretch()
        file_layout.addLayout(file_buttons_layout)
        
        # List of loaded files
        file_layout.addWidget(QLabel("Loaded Files:"))
        self.files_table = QTableWidget(0, 3)
        self.files_table.setHorizontalHeaderLabels(["Filename", "Size", "Actions"])
        self.files_table.horizontalHeader().setStretchLastSection(True)
        file_layout.addWidget(self.files_table)
        
        layout.addWidget(file_section)
        
        # Data preview section
        preview_section = QFrame()
        preview_section.setFrameShape(QFrame.StyledPanel)
        preview_layout = QVBoxLayout(preview_section)
        
        preview_header = QHBoxLayout()
        preview_label = QLabel("Data Preview:")
        preview_label.setFont(QFont("Arial", 12))
        preview_header.addWidget(preview_label)
        
        self.file_selector = QComboBox()
        self.file_selector.currentIndexChanged.connect(self.change_preview_file)
        preview_header.addWidget(self.file_selector)
        
        preview_header.addStretch()
        preview_layout.addLayout(preview_header)
        
        self.data_preview_table = QTableWidget()
        preview_layout.addWidget(self.data_preview_table)
        
        layout.addWidget(preview_section)
        
        self.tabs.addTab(data_loading_tab, "Data Loading")
    
    def setup_data_cleaning_tab(self):
        """Set up the data cleaning tab"""
        data_cleaning_tab = QWidget()
        layout = QVBoxLayout(data_cleaning_tab)
        
        # File selection for cleaning
        file_selection = QHBoxLayout()
        file_selection.addWidget(QLabel("Select File to Clean:"))
        self.cleaning_file_selector = QComboBox()
        file_selection.addWidget(self.cleaning_file_selector)
        analyze_btn = QPushButton("Analyze Data Quality")
        analyze_btn.clicked.connect(self.analyze_data_quality)
        file_selection.addWidget(analyze_btn)
        file_selection.addStretch()
        layout.addLayout(file_selection)
        
        # Data quality results
        results_splitter = QSplitter(Qt.Horizontal)
        
        # Issues and statistics panel
        issues_widget = QWidget()
        issues_layout = QVBoxLayout(issues_widget)
        issues_layout.addWidget(QLabel("Data Issues:"))
        self.issues_text = QTextEdit()
        self.issues_text.setReadOnly(True)
        issues_layout.addWidget(self.issues_text)
        results_splitter.addWidget(issues_widget)
        
        # Cleaning options panel
        cleaning_widget = QWidget()
        cleaning_layout = QVBoxLayout(cleaning_widget)
        cleaning_layout.addWidget(QLabel("Cleaning Options:"))
        self.cleaning_options_widget = QWidget()
        cleaning_options_layout = QVBoxLayout(self.cleaning_options_widget)
        self.cleaning_options_widget.setLayout(cleaning_options_layout)
        
        # Wrap in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.cleaning_options_widget)
        cleaning_layout.addWidget(scroll_area)
        
        # Clean data button
        clean_data_btn = QPushButton("Clean Data")
        clean_data_btn.clicked.connect(self.perform_data_cleaning)
        cleaning_layout.addWidget(clean_data_btn)
        
        results_splitter.addWidget(cleaning_widget)
        
        # Progress bar and log for cleaning process
        self.cleaning_progress = QProgressBar()
        layout.addWidget(self.cleaning_progress)
        
        layout.addWidget(results_splitter)
        
        # Cleaning results comparison
        comparison_label = QLabel("Original vs Cleaned Data:")
        layout.addWidget(comparison_label)
        
        comparison_splitter = QSplitter(Qt.Horizontal)
        
        # Original data table
        original_widget = QWidget()
        original_layout = QVBoxLayout(original_widget)
        original_layout.addWidget(QLabel("Original Data:"))
        self.original_data_table = QTableWidget()
        original_layout.addWidget(self.original_data_table)
        comparison_splitter.addWidget(original_widget)
        
        # Cleaned data table
        cleaned_widget = QWidget()
        cleaned_layout = QVBoxLayout(cleaned_widget)
        cleaned_layout.addWidget(QLabel("Cleaned Data:"))
        self.cleaned_data_table = QTableWidget()
        cleaned_layout.addWidget(self.cleaned_data_table)
        comparison_splitter.addWidget(cleaned_widget)
        
        layout.addWidget(comparison_splitter)
        
        # Save cleaned data button
        save_btn = QPushButton("Save Cleaned Data")
        save_btn.clicked.connect(self.save_cleaned_data)
        layout.addWidget(save_btn)
        
        self.tabs.addTab(data_cleaning_tab, "Data Cleaning")
    
    def setup_data_visualization_tab(self):
        """Set up the data visualization tab"""
        data_viz_tab = QWidget()
        layout = QVBoxLayout(data_viz_tab)
        
        # File selection for visualization
        file_selection = QHBoxLayout()
        file_selection.addWidget(QLabel("Select Data for Visualization:"))
        self.viz_file_selector = QComboBox()
        file_selection.addWidget(self.viz_file_selector)
        
        # Add option to use cleaned data if available
        self.use_cleaned_data = QCheckBox("Use Cleaned Data (if available)")
        file_selection.addWidget(self.use_cleaned_data)
        
        file_selection.addStretch()
        layout.addLayout(file_selection)
        
        # Visualization controls
        controls_layout = QHBoxLayout()
        
        # Visualization type selection
        controls_layout.addWidget(QLabel("Visualization Type:"))
        self.viz_type_selector = QComboBox()
        self.viz_type_selector.addItems([
            "Missing Data Heatmap", 
            "Data Distribution", 
            "Categorical Data Analysis", 
            "Correlation Matrix", 
            "Outlier Detection",
            "Before vs After Cleaning"
        ])
        controls_layout.addWidget(self.viz_type_selector)
        
        # Column selection for applicable visualizations
        controls_layout.addWidget(QLabel("Column:"))
        self.column_selector = QComboBox()
        controls_layout.addWidget(self.column_selector)
        
        # Generate visualization button
        generate_btn = QPushButton("Generate Visualization")
        generate_btn.clicked.connect(self.generate_visualization)
        controls_layout.addWidget(generate_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Visualization display area
        self.viz_display = QVBoxLayout()
        viz_container = QWidget()
        viz_container.setLayout(self.viz_display)
        
        # Wrap in scroll area for large visualizations
        viz_scroll = QScrollArea()
        viz_scroll.setWidgetResizable(True)
        viz_scroll.setWidget(viz_container)
        
        layout.addWidget(viz_scroll)
        
        # Connect signals
        self.viz_type_selector.currentIndexChanged.connect(self.update_viz_options)
        self.viz_file_selector.currentIndexChanged.connect(self.update_column_selector)
        
        self.tabs.addTab(data_viz_tab, "Data Visualization")
    
    def setup_analysis_tab(self):
        """Set up the advanced analysis tab"""
        analysis_tab = QWidget()
        layout = QVBoxLayout(analysis_tab)
        
        # File selection for analysis
        file_selection = QHBoxLayout()
        file_selection.addWidget(QLabel("Select Data for Analysis:"))
        self.analysis_file_selector = QComboBox()
        file_selection.addWidget(self.analysis_file_selector)
        
        # Add option to use cleaned data if available
        self.use_cleaned_for_analysis = QCheckBox("Use Cleaned Data (if available)")
        file_selection.addWidget(self.use_cleaned_for_analysis)
        
        file_selection.addStretch()
        layout.addLayout(file_selection)
        
        # Analysis options in a tabbed interface
        analysis_tabs = QTabWidget()
        
        # Summary statistics tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        summary_btn = QPushButton("Generate Summary Statistics")
        summary_btn.clicked.connect(self.generate_summary_statistics)
        summary_layout.addWidget(summary_btn)
        
        self.summary_table = QTableWidget()
        summary_layout.addWidget(self.summary_table)
        
        analysis_tabs.addTab(summary_tab, "Summary Statistics")
        
        # Data profile tab
        profile_tab = QWidget()
        profile_layout = QVBoxLayout(profile_tab)
        
        profile_controls = QHBoxLayout()
        profile_controls.addWidget(QLabel("Column:"))
        self.profile_column_selector = QComboBox()
        profile_controls.addWidget(self.profile_column_selector)
        
        profile_btn = QPushButton("Generate Column Profile")
        profile_btn.clicked.connect(self.generate_column_profile)
        profile_controls.addWidget(profile_btn)
        
        profile_controls.addStretch()
        profile_layout.addLayout(profile_controls)
        
        self.profile_display = QVBoxLayout()
        profile_container = QWidget()
        profile_container.setLayout(self.profile_display)
        
        profile_scroll = QScrollArea()
        profile_scroll.setWidgetResizable(True)
        profile_scroll.setWidget(profile_container)
        
        profile_layout.addWidget(profile_scroll)
        
        analysis_tabs.addTab(profile_tab, "Column Profiling")
        
        # Duplicate detection tab
        duplicates_tab = QWidget()
        duplicates_layout = QVBoxLayout(duplicates_tab)
        
        duplicate_controls = QHBoxLayout()
        duplicate_controls.addWidget(QLabel("Check columns:"))
        self.duplicate_all_columns = QCheckBox("All Columns")
        self.duplicate_all_columns.setChecked(True)
        duplicate_controls.addWidget(self.duplicate_all_columns)
        
        find_duplicates_btn = QPushButton("Find Duplicates")
        find_duplicates_btn.clicked.connect(self.find_duplicates)
        duplicate_controls.addWidget(find_duplicates_btn)
        
        duplicate_controls.addStretch()
        duplicates_layout.addLayout(duplicate_controls)
        
        self.duplicate_column_selector = QTableWidget()
        self.duplicate_column_selector.setColumnCount(2)
        self.duplicate_column_selector.setHorizontalHeaderLabels(["Column", "Include"])
        duplicates_layout.addWidget(self.duplicate_column_selector)
        self.duplicate_column_selector.setVisible(False)
        
        self.duplicate_all_columns.toggled.connect(
            lambda checked: self.duplicate_column_selector.setVisible(not checked)
        )
        
        self.duplicates_result = QTableWidget()
        duplicates_layout.addWidget(self.duplicates_result)
        
        analysis_tabs.addTab(duplicates_tab, "Duplicate Detection")
        
        layout.addWidget(analysis_tabs)
        
        # Connect signals
        self.analysis_file_selector.currentIndexChanged.connect(self.update_analysis_options)
        
        self.tabs.addTab(analysis_tab, "Data Analysis")
    
    def load_single_file(self):
        """Load a single CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.load_file(file_path)
    
    def load_multiple_files(self):
        """Load multiple CSV files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV Files", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        for file_path in file_paths:
            self.load_file(file_path)
    
    def load_directory(self):
        """Load all CSV files from a directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        
        if directory:
            csv_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                       if f.lower().endswith('.csv')]
            
            for file_path in csv_files:
                self.load_file(file_path)
    
    def load_file(self, file_path):
        """Load a CSV file into the application"""
        try:
            # Check if file is too large
            file_size = os.path.getsize(file_path)
            file_size_str = self.format_file_size(file_size)
            
            # Log file loading
            log_message = f"Loading file: {os.path.basename(file_path)} ({file_size_str})"
            log_action(log_message)
            self.add_log_message(log_message)
            
            # For large files, use chunk processing
            if file_size > 100 * 1024 * 1024:  # 100MB
                self.add_log_message(f"Large file detected. Using optimized loading.")
                # Use dask for larger files
                import dask.dataframe as dd
                dask_df = dd.read_csv(file_path)
                df = dask_df.head(1000)  # Preview with first 1000 rows
                # Store reference to full dask dataframe
                file_key = os.path.basename(file_path)
                self.dataframes[file_key] = {'data': dask_df, 'type': 'dask', 'path': file_path}
            else:
                # Standard pandas loading for smaller files
                df = pd.read_csv(file_path)
                file_key = os.path.basename(file_path)
                self.dataframes[file_key] = {'data': df, 'type': 'pandas', 'path': file_path}
            
            # Update UI with loaded file
            self.update_file_list()
            self.update_file_selectors()
            
            # Update current file and show preview
            self.current_file = file_key
            self.show_data_preview(df)
            
            # Update memory usage
            self.update_memory_usage()
            
            # Success message
            success_message = f"Successfully loaded {file_key} with {len(df)} rows and {len(df.columns)} columns"
            self.add_log_message(success_message)
            
        except Exception as e:
            error_message = f"Error loading file {file_path}: {str(e)}"
            log_action(error_message, level="ERROR")
            self.add_log_message(error_message)
            QMessageBox.critical(self, "Error Loading File", error_message)
    
    def update_file_list(self):
        """Update the table of loaded files"""
        self.files_table.setRowCount(0)  # Clear table
        
        for row, (filename, data_dict) in enumerate(self.dataframes.items()):
            self.files_table.insertRow(row)
            
            # Filename
            self.files_table.setItem(row, 0, QTableWidgetItem(filename))
            
            # File size
            if data_dict['type'] == 'pandas':
                size_text = f"{self.format_file_size(data_dict['data'].memory_usage(deep=True).sum())}"
            else:
                file_size = os.path.getsize(data_dict['path'])
                size_text = f"{self.format_file_size(file_size)} (Dask)"
            
            self.files_table.setItem(row, 1, QTableWidgetItem(size_text))
            
            # Actions
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda checked, f=filename: self.remove_file(f))
            self.files_table.setCellWidget(row, 2, remove_btn)
    
    def update_file_selectors(self):
        """Update all file selector comboboxes"""
        # Block signals to prevent triggering events during update
        self.file_selector.blockSignals(True)
        self.cleaning_file_selector.blockSignals(True)
        self.viz_file_selector.blockSignals(True)
        self.analysis_file_selector.blockSignals(True)
        
        # Store current selections
        current_preview = self.file_selector.currentText()
        current_cleaning = self.cleaning_file_selector.currentText()
        current_viz = self.viz_file_selector.currentText()
        current_analysis = self.analysis_file_selector.currentText()
        
        # Clear and refill selectors
        for selector in [self.file_selector, self.cleaning_file_selector, 
                       self.viz_file_selector, self.analysis_file_selector]:
            selector.clear()
            selector.addItems(sorted(self.dataframes.keys()))
        
        # Restore previous selections if still valid
        if current_preview in self.dataframes:
            self.file_selector.setCurrentText(current_preview)
        if current_cleaning in self.dataframes:
            self.cleaning_file_selector.setCurrentText(current_cleaning)
        if current_viz in self.dataframes:
            self.viz_file_selector.setCurrentText(current_viz)
        if current_analysis in self.dataframes:
            self.analysis_file_selector.setCurrentText(current_analysis)
        
        # Unblock signals
        self.file_selector.blockSignals(False)
        self.cleaning_file_selector.blockSignals(False)
        self.viz_file_selector.blockSignals(False)
        self.analysis_file_selector.blockSignals(False)
        
        # Update dependent selectors
        if self.file_selector.count() > 0:
            self.change_preview_file()
        
        if self.cleaning_file_selector.count() > 0:
            self.update_cleaning_options()
        
        if self.viz_file_selector.count() > 0:
            self.update_column_selector()
        
        if self.analysis_file_selector.count() > 0:
            self.update_analysis_options()
    
    def remove_file(self, filename):
        """Remove a file from the application"""
        if filename in self.dataframes:
            del self.dataframes[filename]
            log_action(f"Removed file: {filename}")
            self.add_log_message(f"Removed file: {filename}")
            
            # Update UI
            self.update_file_list()
            self.update_file_selectors()
            
            # Update memory usage
            self.update_memory_usage()
    
    def change_preview_file(self):
        """Change the file being previewed"""
        selected_file = self.file_selector.currentText()
        if selected_file and selected_file in self.dataframes:
            self.current_file = selected_file
            data_dict = self.dataframes[selected_file]
            
            if data_dict['type'] == 'pandas':
                self.current_df = data_dict['data']
                self.show_data_preview(self.current_df)
            else:
                # For dask dataframes, show the first n rows
                dask_df = data_dict['data']
                preview_df = dask_df.head(1000)
                self.current_df = preview_df
                self.show_data_preview(preview_df, is_preview=True)
    
    def show_data_preview(self, df, is_preview=False):
        """Display a preview of the dataframe in the table"""
        # Clear table
        self.data_preview_table.clear()
        
        # Set row and column count
        rows, cols = min(1000, len(df)), len(df.columns)
        self.data_preview_table.setRowCount(rows)
        self.data_preview_table.setColumnCount(cols)
        
        # Set headers
        self.data_preview_table.setHorizontalHeaderLabels(df.columns)
        
        # Fill data
        for i in range(rows):
            for j in range(cols):
                value = str(df.iloc[i, j])
                self.data_preview_table.setItem(i, j, QTableWidgetItem(value))
        
        # If preview of large file, add note
        if is_preview:
            self.add_log_message("Note: Showing preview of large file (first 1000 rows)")
    
    def analyze_data_quality(self):
        """Analyze the quality of the selected data"""
        selected_file = self.cleaning_file_selector.currentText()
        if not selected_file:
            QMessageBox.warning(self, "No File Selected", "Please select a file for analysis")
            return
        
        data_dict = self.dataframes[selected_file]
        
        # Check if we're working with a dask dataframe
        if data_dict['type'] == 'dask':
            QMessageBox.information(
                self, 
                "Large File Detected", 
                "You're working with a large file. Analysis will be performed on a sample."
            )
            # Convert dask dataframe sample to pandas for analysis
            df = data_dict['data'].head(10000)
        else:
            df = data_dict['data']
        
        # Reset UI elements
        self.cleaning_progress.setValue(0)
        self.issues_text.clear()
        
        # Clear any previous cleaning options
        for i in reversed(range(self.cleaning_options_widget.layout().count())): 
            widget = self.cleaning_options_widget.layout().itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Start worker thread for analysis
        self.worker = DataProcessingWorker('identify_issues', df)
        self.worker.update_progress.connect(self.cleaning_progress.setValue)
        self.worker.update_log.connect(self.add_log_message)
        self.worker.finished_task.connect(self.handle_quality_results)
        self.worker.start()
    
    def handle_quality_results(self, results):
        """Handle the results of data quality analysis"""
        if 'error' in results:
            QMessageBox.critical(self, "Error in Analysis", results['error'])
            return
        
        issues = results['issues']
        quality_report = results['quality_report']
        cleaning_strategy = results['cleaning_strategy']
        
        # Display issues
        issues_text = "Data Quality Issues:\n\n"
        
        # Missing values
        issues_text += "Missing Values:\n"
        for col, count in issues['missing_values'].items():
            percentage = (count / len(self.current_df)) * 100
            issues_text += f"  {col}: {count} values ({percentage:.2f}%)\n"
        
        # Duplicates
        if issues['duplicated_rows'] > 0:
            dup_percentage = (issues['duplicated_rows'] / len(self.current_df)) * 100
            issues_text += f"\nDuplicated Rows: {issues['duplicated_rows']} ({dup_percentage:.2f}%)\n"
        
        # Outliers
        issues_text += "\nOutliers:\n"
        for col, count in issues['outliers'].items():
            percentage = (count / len(self.current_df)) * 100
            issues_text += f"  {col}: {count} values ({percentage:.2f}%)\n"
        
        # Data type issues
        issues_text += "\nData Type Issues:\n"
        for col, type_issues in issues['type_issues'].items():
            issues_text += f"  {col}: {type_issues}\n"
        
        self.issues_text.setText(issues_text)
        
        # Create cleaning options based on suggestions
        for col, strategies in cleaning_strategy.items():
            # Create group box for column
            group_box = QFrame()
            group_box.setFrameShape(QFrame.StyledPanel)
            group_layout = QVBoxLayout(group_box)
            
            group_layout.addWidget(QLabel(f"Column: {col}"))
            
            for strategy in strategies:
                strategy_check = QCheckBox(strategy['description'])
                strategy_check.setChecked(strategy['recommended'])
                strategy_check.setProperty('column', col)
                strategy_check.setProperty('strategy', strategy['type'])
                strategy_check.setProperty('params', strategy.get('params', {}))
                group_layout.addWidget(strategy_check)
            
            self.cleaning_options_widget.layout().addWidget(group_box)
        
        # Add a spacer at the end
        self.cleaning_options_widget.layout().addStretch()
        
        # Update the original data table
        self.show_table_data(self.original_data_table, self.current_df)
        
        # Log completion
        self.add_log_message("Data quality analysis complete. Review results and select cleaning options.")
    
    def perform_data_cleaning(self):
        """Perform data cleaning based on selected options"""
        if not self.current_df is not None:
            QMessageBox.warning(self, "No Data", "Please load and analyze data first")
            return
        
        # Collect selected cleaning options
        cleaning_settings = {}
        
        for i in range(self.cleaning_options_widget.layout().count()):
            item = self.cleaning_options_widget.layout().itemAt(i)
            if item.widget() and isinstance(item.widget(), QFrame):
                frame = item.widget()
                for j in range(frame.layout().count()):
                    widget = frame.layout().itemAt(j).widget()
                    if isinstance(widget, QCheckBox) and widget.isChecked():
                        column = widget.property('column')
                        strategy = widget.property('strategy')
                        params = widget.property('params')
                        
                        if column not in cleaning_settings:
                            cleaning_settings[column] = []
                        
                        cleaning_settings[column].append({
                            'strategy': strategy,
                            'params': params
                        })
        
        if not cleaning_settings:
            QMessageBox.information(self, "No Options Selected", "Please select at least one cleaning option")
            return
        
        # Reset progress
        self.cleaning_progress.setValue(0)
        
        # Start worker thread for cleaning
        self.worker = DataProcessingWorker('clean_data', self.current_df, {'cleaning_settings': cleaning_settings})
        self.worker.update_progress.connect(self.cleaning_progress.setValue)
        self.worker.update_log.connect(self.add_log_message)
        self.worker.finished_task.connect(self.handle_cleaning_results)
        self.worker.start()
    
    def handle_cleaning_results(self, results):
        """Handle the results of data cleaning"""
        if 'error' in results:
            QMessageBox.critical(self, "Error in Cleaning", results['error'])
            return
        
        self.cleaned_df = results['cleaned_df']
        cleaning_log = results['cleaning_log']
        
        # Display cleaning log
        log_text = "Cleaning Operations:\n\n"
        for operation in cleaning_log:
            log_text += f"• {operation}\n"
        
        self.add_log_message(log_text)
        
        # Show cleaned data in table
        self.show_table_data(self.cleaned_data_table, self.cleaned_df)
        
        # Add cleaned data to dropdown for visualization and analysis
        selected_file = self.cleaning_file_selector.currentText()
        cleaned_key = f"{selected_file} (Cleaned)"
        self.dataframes[cleaned_key] = {'data': self.cleaned_df, 'type': 'pandas', 'path': None}
        
        # Update file selectors
        self.update_file_selectors()
        
        # Log completion
        self.add_log_message("Data cleaning complete. Review results in the comparison tables.")
        
        QMessageBox.information(
            self, 
            "Cleaning Complete", 
            f"Data cleaning completed successfully.\n\n"
            f"Original rows: {len(self.current_df)}\n"
            f"Cleaned rows: {len(self.cleaned_df)}\n\n"
            f"The cleaned dataset is now available for visualization and analysis."
        )
    
    def save_cleaned_data(self):
        """Save the cleaned data to a CSV file"""
        if self.cleaned_df is None:
            QMessageBox.warning(self, "No Cleaned Data", "Please clean data before saving")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Cleaned Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.cleaned_df.to_csv(file_path, index=False)
                log_action(f"Saved cleaned data to {file_path}")
                self.add_log_message(f"Successfully saved cleaned data to {file_path}")
                QMessageBox.information(self, "Save Successful", f"Data saved to {file_path}")
            except Exception as e:
                error_message = f"Error saving file: {str(e)}"
                log_action(error_message, level="ERROR")
                self.add_log_message(error_message)
                QMessageBox.critical(self, "Save Error", error_message)
    
    def update_column_selector(self):
        """Update the column selector in visualization tab"""
        selected_file = self.viz_file_selector.currentText()
        if not selected_file or selected_file not in self.dataframes:
            return
        
        # Determine which dataframe to use
        use_cleaned = self.use_cleaned_data.isChecked() and self.cleaned_df is not None
        
        if use_cleaned:
            df = self.cleaned_df
        else:
            data_dict = self.dataframes[selected_file]
            if data_dict['type'] == 'dask':
                # For dask dataframes, we'll just use column names
                df = data_dict['data'].head(1)  # Just to get column names
            else:
                df = data_dict['data']
        
        # Update column selector
        self.column_selector.clear()
        self.column_selector.addItems(df.columns)
        
        # Also update profile column selector
        self.profile_column_selector.clear()
        self.profile_column_selector.addItems(df.columns)
    
    def update_viz_options(self):
        """Update visualization options based on selected type"""
        viz_type = self.viz_type_selector.currentText()
        
        # Show/hide column selector based on visualization type
        if viz_type in ["Data Distribution", "Categorical Data Analysis", "Outlier Detection"]:
            self.column_selector.setEnabled(True)
        else:
            self.column_selector.setEnabled(False)
    
    def generate_visualization(self):
        """Generate the selected visualization"""
        selected_file = self.viz_file_selector.currentText()
        if not selected_file:
            QMessageBox.warning(self, "No File Selected", "Please select a file for visualization")
            return
        
        # Determine which dataframe to use
        use_cleaned = self.use_cleaned_data.isChecked() and self.cleaned_df is not None
        
        if use_cleaned:
            df = self.cleaned_df
        else:
            data_dict = self.dataframes[selected_file]
            if data_dict['type'] == 'dask':
                # For dask dataframes, use a sample
                df = data_dict['data'].head(10000).compute()
                QMessageBox.information(
                    self, 
                    "Large File Sampling", 
                    "Using a sample of 10,000 rows for visualization with large file."
                )
            else:
                df = data_dict['data']
        
        # Clear current visualization
        self.clear_layout(self.viz_display)
        
        # Get visualization type
        viz_type = self.viz_type_selector.currentText()
        
        try:
            if viz_type == "Missing Data Heatmap":
                self.display_missing_data_viz(df)
            
            elif viz_type == "Data Distribution":
                column = self.column_selector.currentText()
                self.display_distribution_viz(df, column)
            
            elif viz_type == "Categorical Data Analysis":
                column = self.column_selector.currentText()
                self.display_categorical_viz(df, column)
            
            elif viz_type == "Correlation Matrix":
                self.display_correlation_viz(df)
            
            elif viz_type == "Outlier Detection":
                column = self.column_selector.currentText()
                self.display_outlier_viz(df, column)
            
            elif viz_type == "Before vs After Cleaning":
                if self.cleaned_df is None:
                    QMessageBox.warning(self, "No Cleaned Data", "Please clean data first")
                    return
                
                self.display_cleaning_comparison_viz(self.current_df, self.cleaned_df)
        
        except Exception as e:
            error_message = f"Error generating visualization: {str(e)}"
            log_action(error_message, level="ERROR")
            self.add_log_message(error_message)
            QMessageBox.critical(self, "Visualization Error", error_message)
    
    def display_missing_data_viz(self, df):
        """Display missing data visualization"""
        self.add_log_message("Generating missing data heatmap...")
        
        # Generate the plotly figure
        fig = plot_missing_data(df)
        
        # Convert plotly to matplotlib
        matplotlib_fig = self.plotly_to_matplotlib(fig)
        
        # Display the figure
        self.display_matplotlib_figure(matplotlib_fig)
    
    def display_distribution_viz(self, df, column):
        """Display distribution visualization for a column"""
        self.add_log_message(f"Generating distribution visualization for column: {column}")
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            fig = plot_data_distribution(df, column)
            matplotlib_fig = self.plotly_to_matplotlib(fig)
            self.display_matplotlib_figure(matplotlib_fig)
        else:
            QMessageBox.warning(
                self, 
                "Non-numeric Column", 
                f"The column '{column}' is not numeric. Please select a numeric column for distribution visualization."
            )
    
    def display_categorical_viz(self, df, column):
        """Display categorical data visualization for a column"""
        self.add_log_message(f"Generating categorical visualization for column: {column}")
        
        # Generate the figures
        bar_chart, pie_chart = plot_categorical_data(df, column)
        
        # Convert and display both charts
        bar_fig = self.plotly_to_matplotlib(bar_chart)
        pie_fig = self.plotly_to_matplotlib(pie_chart)
        
        self.display_matplotlib_figure(bar_fig)
        self.display_matplotlib_figure(pie_fig)
    
    def display_correlation_viz(self, df):
        """Display correlation matrix visualization"""
        self.add_log_message("Generating correlation matrix...")
        
        # Filter numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            QMessageBox.warning(
                self, 
                "Insufficient Numeric Columns", 
                "Need at least 2 numeric columns to generate correlation matrix."
            )
            return
        
        # Generate correlation matrix
        fig = plot_correlation_matrix(df[numeric_cols])
        matplotlib_fig = self.plotly_to_matplotlib(fig)
        self.display_matplotlib_figure(matplotlib_fig)
    
    def display_outlier_viz(self, df, column):
        """Display outlier visualization for a column"""
        self.add_log_message(f"Generating outlier visualization for column: {column}")
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            fig = plot_outliers(df, column)
            matplotlib_fig = self.plotly_to_matplotlib(fig)
            self.display_matplotlib_figure(matplotlib_fig)
        else:
            QMessageBox.warning(
                self, 
                "Non-numeric Column", 
                f"The column '{column}' is not numeric. Please select a numeric column for outlier visualization."
            )
    
    def display_cleaning_comparison_viz(self, original_df, cleaned_df):
        """Display comparison between original and cleaned data"""
        self.add_log_message("Generating comparison visualization between original and cleaned data...")
        
        fig = plot_cleaned_vs_original(original_df, cleaned_df)
        matplotlib_fig = self.plotly_to_matplotlib(fig)
        self.display_matplotlib_figure(matplotlib_fig)
    
    def update_analysis_options(self):
        """Update analysis options when file selection changes"""
        selected_file = self.analysis_file_selector.currentText()
        if not selected_file or selected_file not in self.dataframes:
            return
        
        # Determine which dataframe to use
        use_cleaned = self.use_cleaned_for_analysis.isChecked() and self.cleaned_df is not None
        
        if use_cleaned:
            df = self.cleaned_df
        else:
            data_dict = self.dataframes[selected_file]
            if data_dict['type'] == 'dask':
                # For dask dataframes, we'll just use a sample for UI updates
                df = data_dict['data'].head(1)
            else:
                df = data_dict['data']
        
        # Update profile column selector
        self.profile_column_selector.clear()
        self.profile_column_selector.addItems(df.columns)
        
        # Update duplicate detection column selector
        self.duplicate_column_selector.setRowCount(len(df.columns))
        for i, col in enumerate(df.columns):
            # Column name
            self.duplicate_column_selector.setItem(i, 0, QTableWidgetItem(col))
            
            # Checkbox for inclusion
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            self.duplicate_column_selector.setCellWidget(i, 1, checkbox)
    
    def generate_summary_statistics(self):
        """Generate summary statistics for the selected data"""
        selected_file = self.analysis_file_selector.currentText()
        if not selected_file:
            QMessageBox.warning(self, "No File Selected", "Please select a file for analysis")
            return
        
        # Determine which dataframe to use
        use_cleaned = self.use_cleaned_for_analysis.isChecked() and self.cleaned_df is not None
        
        if use_cleaned:
            df = self.cleaned_df
        else:
            data_dict = self.dataframes[selected_file]
            if data_dict['type'] == 'dask':
                # For dask dataframes, use a sample
                df = data_dict['data'].head(10000).compute()
                QMessageBox.information(
                    self, 
                    "Large File Sampling", 
                    "Using a sample of 10,000 rows for statistics with large file."
                )
            else:
                df = data_dict['data']
        
        try:
            # Get summary statistics
            summary_df = get_data_summary(df)
            
            # Display in table
            self.summary_table.clear()
            
            # Set row and column count
            rows, cols = len(summary_df), len(summary_df.columns)
            self.summary_table.setRowCount(rows)
            self.summary_table.setColumnCount(cols)
            
            # Set headers
            self.summary_table.setHorizontalHeaderLabels(summary_df.columns)
            self.summary_table.setVerticalHeaderLabels(summary_df.index)
            
            # Fill data
            for i in range(rows):
                for j in range(cols):
                    value = str(summary_df.iloc[i, j])
                    self.summary_table.setItem(i, j, QTableWidgetItem(value))
            
            self.add_log_message("Generated summary statistics successfully")
            
        except Exception as e:
            error_message = f"Error generating summary statistics: {str(e)}"
            log_action(error_message, level="ERROR")
            self.add_log_message(error_message)
            QMessageBox.critical(self, "Statistics Error", error_message)
    
    def generate_column_profile(self):
        """Generate a detailed profile for a selected column"""
        selected_file = self.analysis_file_selector.currentText()
        selected_column = self.profile_column_selector.currentText()
        
        if not selected_file or not selected_column:
            QMessageBox.warning(self, "Missing Selection", "Please select a file and column for profiling")
            return
        
        # Determine which dataframe to use
        use_cleaned = self.use_cleaned_for_analysis.isChecked() and self.cleaned_df is not None
        
        if use_cleaned:
            df = self.cleaned_df
        else:
            data_dict = self.dataframes[selected_file]
            if data_dict['type'] == 'dask':
                # For dask dataframes, use a sample
                df = data_dict['data'].head(10000).compute()
                QMessageBox.information(
                    self, 
                    "Large File Sampling", 
                    "Using a sample of 10,000 rows for column profiling with large file."
                )
            else:
                df = data_dict['data']
        
        try:
            # Clear current profile display
            self.clear_layout(self.profile_display)
            
            # Add column statistics
            stats_label = QLabel(f"Column Profile: {selected_column}")
            stats_label.setFont(QFont("Arial", 14, QFont.Bold))
            self.profile_display.addWidget(stats_label)
            
            # Basic statistics
            stats_frame = QFrame()
            stats_frame.setFrameShape(QFrame.StyledPanel)
            stats_layout = QVBoxLayout(stats_frame)
            
            # Get column data
            column_data = df[selected_column]
            
            # Column type
            dtype_label = QLabel(f"Data Type: {column_data.dtype}")
            stats_layout.addWidget(dtype_label)
            
            # Count of values
            count_label = QLabel(f"Count: {column_data.count()} / {len(column_data)}")
            stats_layout.addWidget(count_label)
            
            # Missing values
            missing_count = column_data.isna().sum()
            missing_pct = (missing_count / len(column_data)) * 100
            missing_label = QLabel(f"Missing Values: {missing_count} ({missing_pct:.2f}%)")
            stats_layout.addWidget(missing_label)
            
            # Unique values
            try:
                unique_count = column_data.nunique()
                unique_label = QLabel(f"Unique Values: {unique_count}")
                stats_layout.addWidget(unique_label)
            except:
                # Some types may not support nunique()
                pass
            
            # Add more statistics for numeric data
            if pd.api.types.is_numeric_dtype(column_data):
                numeric_stats = [
                    f"Minimum: {column_data.min()}",
                    f"Maximum: {column_data.max()}",
                    f"Mean: {column_data.mean():.4f}",
                    f"Median: {column_data.median():.4f}",
                    f"Std Dev: {column_data.std():.4f}",
                ]
                
                for stat in numeric_stats:
                    stats_layout.addWidget(QLabel(stat))
            
            self.profile_display.addWidget(stats_frame)
            
            # Add visualizations
            viz_label = QLabel("Visualizations:")
            viz_label.setFont(QFont("Arial", 12, QFont.Bold))
            self.profile_display.addWidget(viz_label)
            
            try:
                # For numerical columns
                if pd.api.types.is_numeric_dtype(column_data):
                    # Distribution plot
                    fig = plot_data_distribution(df, selected_column)
                    matplotlib_fig = self.plotly_to_matplotlib(fig)
                    self.profile_display.addWidget(QLabel("Distribution:"))
                    self.profile_display.addWidget(MatplotlibCanvas(matplotlib_fig))
                    
                    # Outlier plot
                    fig = plot_outliers(df, selected_column)
                    matplotlib_fig = self.plotly_to_matplotlib(fig)
                    self.profile_display.addWidget(QLabel("Outlier Detection:"))
                    self.profile_display.addWidget(MatplotlibCanvas(matplotlib_fig))
                
                # For categorical/text columns
                else:
                    try:
                        # Categorical plots
                        bar_chart, pie_chart = plot_categorical_data(df, selected_column)
                        
                        bar_fig = self.plotly_to_matplotlib(bar_chart)
                        pie_fig = self.plotly_to_matplotlib(pie_chart)
                        
                        self.profile_display.addWidget(QLabel("Value Counts:"))
                        self.profile_display.addWidget(MatplotlibCanvas(bar_fig))
                        
                        self.profile_display.addWidget(QLabel("Value Distribution:"))
                        self.profile_display.addWidget(MatplotlibCanvas(pie_fig))
                    except Exception as e:
                        self.add_log_message(f"Could not generate categorical plots: {str(e)}")
                        
                        # For columns with many unique values, show a table of top values
                        if unique_count > 20:
                            self.profile_display.addWidget(QLabel("Top 20 Most Common Values:"))
                            top_values = column_data.value_counts().head(20)
                            
                            value_table = QTableWidget(len(top_values), 2)
                            value_table.setHorizontalHeaderLabels(["Value", "Count"])
                            
                            for i, (value, count) in enumerate(top_values.items()):
                                value_table.setItem(i, 0, QTableWidgetItem(str(value)))
                                value_table.setItem(i, 1, QTableWidgetItem(str(count)))
                            
                            self.profile_display.addWidget(value_table)
            except Exception as e:
                self.add_log_message(f"Error generating profile visualizations: {str(e)}")
            
            self.add_log_message(f"Generated profile for column: {selected_column}")
            
        except Exception as e:
            error_message = f"Error generating column profile: {str(e)}"
            log_action(error_message, level="ERROR")
            self.add_log_message(error_message)
            QMessageBox.critical(self, "Profiling Error", error_message)
    
    def find_duplicates(self):
        """Find duplicate rows in the data"""
        selected_file = self.analysis_file_selector.currentText()
        if not selected_file:
            QMessageBox.warning(self, "No File Selected", "Please select a file for duplicate detection")
            return
        
        # Determine which dataframe to use
        use_cleaned = self.use_cleaned_for_analysis.isChecked() and self.cleaned_df is not None
        
        if use_cleaned:
            df = self.cleaned_df
        else:
            data_dict = self.dataframes[selected_file]
            if data_dict['type'] == 'dask':
                # For dask dataframes, use a sample
                df = data_dict['data'].head(10000).compute()
                QMessageBox.information(
                    self, 
                    "Large File Sampling", 
                    "Using a sample of 10,000 rows for duplicate detection with large file."
                )
            else:
                df = data_dict['data']
        
        try:
            # Determine which columns to check
            if self.duplicate_all_columns.isChecked():
                subset = None  # Check all columns
            else:
                # Get selected columns from table
                subset = []
                for i in range(self.duplicate_column_selector.rowCount()):
                    checkbox = self.duplicate_column_selector.cellWidget(i, 1)
                    if checkbox and checkbox.isChecked():
                        col_name = self.duplicate_column_selector.item(i, 0).text()
                        subset.append(col_name)
                
                if not subset:
                    QMessageBox.warning(
                        self, 
                        "No Columns Selected", 
                        "Please select at least one column for duplicate detection"
                    )
                    return
            
            # Find duplicates
            duplicates = df.duplicated(subset=subset, keep='first')
            duplicate_df = df[duplicates]
            
            # Display results
            self.duplicates_result.clear()
            
            if len(duplicate_df) == 0:
                QMessageBox.information(
                    self, 
                    "No Duplicates Found", 
                    "No duplicate rows were found with the selected criteria."
                )
                return
            
            # Set row and column count
            rows, cols = len(duplicate_df), len(duplicate_df.columns)
            self.duplicates_result.setRowCount(rows)
            self.duplicates_result.setColumnCount(cols)
            
            # Set headers
            self.duplicates_result.setHorizontalHeaderLabels(duplicate_df.columns)
            
            # Fill data
            for i in range(rows):
                for j in range(cols):
                    value = str(duplicate_df.iloc[i, j])
                    self.duplicates_result.setItem(i, j, QTableWidgetItem(value))
            
            # Log results
            duplicate_count = len(duplicate_df)
            total_count = len(df)
            duplicate_pct = (duplicate_count / total_count) * 100
            
            msg = f"Found {duplicate_count} duplicate rows ({duplicate_pct:.2f}% of data)"
            if subset:
                msg += f" when checking columns: {', '.join(subset)}"
            
            self.add_log_message(msg)
            
        except Exception as e:
            error_message = f"Error finding duplicates: {str(e)}"
            log_action(error_message, level="ERROR")
            self.add_log_message(error_message)
            QMessageBox.critical(self, "Duplicate Detection Error", error_message)
    
    def update_memory_usage(self):
        """Update the memory usage display"""
        memory_info = get_memory_usage()
        
        # Format for display
        memory_text = (f"Memory Usage: {memory_info['usage_percent']}% "
                      f"({memory_info['used_formatted']} of {memory_info['total_formatted']})")
        
        self.memory_label.setText(memory_text)
        
        # Log memory usage periodically
        log_action(f"Memory usage: {memory_info['usage_percent']}% ({memory_info['used_formatted']})")
    
    def add_log_message(self, message):
        """Add a message to the log display"""
        # Get current time
        current_time = pd.Timestamp.now().strftime("%H:%M:%S")
        
        # Add message with timestamp
        self.log_display.append(f"[{current_time}] {message}")
        
        # Auto-scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def show_table_data(self, table_widget, df):
        """Show dataframe in a table widget"""
        # Clear table
        table_widget.clear()
        
        # Set row and column count
        rows, cols = min(1000, len(df)), len(df.columns)
        table_widget.setRowCount(rows)
        table_widget.setColumnCount(cols)
        
        # Set headers
        table_widget.setHorizontalHeaderLabels(df.columns)
        
        # Fill data
        for i in range(rows):
            for j in range(cols):
                value = str(df.iloc[i, j])
                table_widget.setItem(i, j, QTableWidgetItem(value))
    
    def plotly_to_matplotlib(self, fig):
        """Convert a plotly figure to matplotlib figure"""
        # Create a new matplotlib figure
        matplotlib_fig = plt.figure(figsize=(10, 6))
        
        # Use plotly's conversion functionality
        img_bytes = pio.to_image(fig, format="png")
        
        # Use matplotlib to display the image
        ax = matplotlib_fig.add_subplot(111)
        ax.imshow(plt.imread(io.BytesIO(img_bytes)))
        ax.axis('off')
        
        return matplotlib_fig
    
    def display_matplotlib_figure(self, fig):
        """Display a matplotlib figure in the visualization area"""
        canvas = MatplotlibCanvas(fig)
        self.viz_display.addWidget(canvas)
    
    def clear_layout(self, layout):
        """Clear all widgets from a layout"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def format_file_size(self, size_bytes):
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

def main():
    # Create the application
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = DataAnalysisApp()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()