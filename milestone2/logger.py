"""
Logging configuration
"""

import logging
import sys
from pathlib import Path
from config import LOG_LEVEL, LOG_FORMAT, LOGS_DIR


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Setup logger with both console and file handlers
    
    Args:
        name: Logger name
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global loggers
logger_ml = setup_logger("ml", "ml.log")
logger_rag = setup_logger("rag", "rag.log")
logger_agent = setup_logger("agent", "agent.log")
logger_api = setup_logger("api", "api.log")
logger_ui = setup_logger("ui", "ui.log")
logger_tools = setup_logger("tools", "tools.log")
logger_llm = setup_logger("llm", "llm.log")
