"""
Configuration settings for the case study application.
All important configuration parameters and variables are centralized here.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR

# Database configuration
DATABASE_CONFIG_PATH = os.getenv("DATABASE_CONFIG_PATH", "config.yaml")
METADATA_PATH = os.getenv("METADATA_PATH", "data/metadata.yaml")


# Function to load database configuration
def load_database_config() -> Dict[str, Any]:
    """
    Load database configuration from the config file.
    Returns a dictionary with database connection information.
    """
    config_path = Path(DATABASE_CONFIG_PATH)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


CONFIG = load_database_config()

# Logging configuration
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "project_id": CONFIG["logger_project_id"],
    "credentials_path": CONFIG["logger_credentials_path"],
    "default_level": "INFO",
    "local_logging": True,  # Enable console logging by default
}

# DuckDB configuration
DUCKDB_CONFIG = {
    "in_memory": False,  # Use file-based database by default
    "config_path": DATABASE_CONFIG_PATH,
    "database_path": str("data/duckdb_logs.db"),  # Path for permanent storage
}

# Default database connections
DEFAULT_POSTGRES_CONNECTION = "postgres_datamancers"
DEFAULT_BIGQUERY_CONNECTION = "datamancers"

# Cache settings
CACHE_CONFIG = {
    "enabled": True,
    "ttl": 3600,  # Cache time-to-live in seconds
}


def get_env_settings() -> Dict[str, Any]:
    """
    Get all environment-specific settings.
    Override default settings with environment variables where applicable.
    """
    settings = {
        "database_config_path": DATABASE_CONFIG_PATH,
        "metadata_path": METADATA_PATH,
        "logging": LOGGING.copy(),
        "duckdb": DUCKDB_CONFIG.copy(),
        "cache": CACHE_CONFIG.copy(),
    }

    # Override with environment variables if they exist
    if os.getenv("DATABASE_CONFIG_PATH"):
        settings["database_config_path"] = os.getenv("DATABASE_CONFIG_PATH")

    if os.getenv("METADATA_PATH"):
        settings["metadata_path"] = os.getenv("METADATA_PATH")

    if os.getenv("LOGGER_PROJECT_ID"):
        settings["logging"]["project_id"] = os.getenv("LOGGER_PROJECT_ID")

    if os.getenv("LOGGER_CREDENTIALS_PATH"):
        settings["logging"]["credentials_path"] = os.getenv("LOGGER_CREDENTIALS_PATH")

    return settings
