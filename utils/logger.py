"""
Logger utility for the Hybrid AI-Quantum Satellite Image Encryption System.
Provides consistent logging across all modules with file and console output.
"""

import logging
import os
import json
from datetime import datetime


def setup_logger(name: str, config_path: str = None) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name (typically module name like 'AI_ENGINE', 'QUANTUM_ENGINE').
        config_path: Path to config.json. If None, uses default settings.

    Returns:
        Configured logging.Logger instance.
    """
    # Default config
    log_level = "INFO"
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    log_to_file = True
    log_filename = "encryption_system.log"
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

    # Load config if available
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            log_cfg = config.get("logging", {})
            log_level = log_cfg.get("level", log_level)
            log_format = log_cfg.get("format", log_format)
            date_format = log_cfg.get("date_format", date_format)
            log_to_file = log_cfg.get("log_to_file", log_to_file)
            log_filename = log_cfg.get("log_filename", log_filename)
        except Exception:
            pass

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        os.makedirs(logs_dir, exist_ok=True)
        log_file_path = os.path.join(logs_dir, log_filename)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_config_path() -> str:
    """Get the path to config.json."""
    return os.path.join(get_project_root(), "config", "config.json")


def load_config() -> dict:
    """Load and return the project configuration."""
    config_path = get_config_path()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)
