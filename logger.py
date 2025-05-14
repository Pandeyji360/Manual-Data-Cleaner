import logging
import os
import sys
from datetime import datetime

def setup_logger():
    """
    Set up a logger with appropriate formatting and output.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("data_pipeline")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"data_pipeline_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info("Logger initialized successfully")
    
    return logger

def log_action(message, level="INFO"):
    """
    Log an action with the specified level.
    
    Args:
        message (str): Message to log
        level (str): Logging level (INFO, WARNING, ERROR, DEBUG)
    """
    logger = logging.getLogger("data_pipeline")
    
    if level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "DEBUG":
        logger.debug(message)

def get_logs(n=None):
    """
    Retrieve the latest n log entries.
    
    Args:
        n (int, optional): Number of log entries to retrieve. If None, retrieves all.
        
    Returns:
        list: List of log entries as strings
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        return []
    
    # Find the most recent log file
    log_files = [f for f in os.listdir(log_dir) if f.startswith("data_pipeline_")]
    if not log_files:
        return []
    
    most_recent = max(log_files)
    log_file = os.path.join(log_dir, most_recent)
    
    # Read log entries
    with open(log_file, 'r') as f:
        log_entries = f.readlines()
    
    # Return all entries or last n entries
    if n is None:
        return log_entries
    else:
        return log_entries[-n:]
