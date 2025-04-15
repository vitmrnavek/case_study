from google.cloud import logging as google_logging
from google.oauth2 import service_account
import logging
import sys
from typing import Dict, Any
from pathlib import Path
from functools import lru_cache
from case_study.utils.settings import LOGGING


class CloudLogger:
    """A wrapper class for Google Cloud Logging that provides both cloud and local logging capabilities."""

    def __init__(
        self,
        project_id: str,
        credentials_path: str,
        logger_name: str = "default",
        log_level: int = logging.INFO,
        local_logging: bool = LOGGING["local_logging"],
    ):
        """
        Initialize the cloud logger.

        Args:
            project_id: Google Cloud project ID
            credentials_path: Path to the service account credentials file
            logger_name: Name of the logger (default: 'default')
            log_level: Logging level (default: logging.INFO)
            local_logging: Whether to also log to console (default from settings)
        """
        self.project_id = project_id
        self.credentials_path = Path(credentials_path)
        self.logger_name = logger_name

        # Validate credentials file exists
        if not self.credentials_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {credentials_path}")

        # Setup Google Cloud Logging
        credentials = service_account.Credentials.from_service_account_file(
            str(self.credentials_path)
        )
        self.client = google_logging.Client(project=project_id, credentials=credentials)
        self.cloud_logger = self.client.logger(logger_name)

        # Setup local logging if enabled
        if local_logging:
            self.local_logger = logging.getLogger(logger_name)
            self.local_logger.setLevel(log_level)

            # Add console handler if none exists
            if not self.local_logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.local_logger.addHandler(handler)
        else:
            self.local_logger = None

    def _log_locally(self, message: str, level: int, **kwargs):
        """Helper method to log messages locally if local logging is enabled."""
        if self.local_logger:
            self.local_logger.log(level, message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log an info message."""
        self.cloud_logger.log_text(message, severity="INFO")
        self._log_locally(message, logging.INFO, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self.cloud_logger.log_text(message, severity="WARNING")
        self._log_locally(message, logging.WARNING, **kwargs)

    def error(self, message: str, **kwargs):
        """Log an error message."""
        self.cloud_logger.log_text(message, severity="ERROR")
        self._log_locally(message, logging.ERROR, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log a critical message."""
        self.cloud_logger.log_text(message, severity="CRITICAL")
        self._log_locally(message, logging.CRITICAL, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self.cloud_logger.log_text(message, severity="DEBUG")
        self._log_locally(message, logging.DEBUG, **kwargs)

    def log_struct(self, struct: Dict[str, Any], severity: str = "INFO", **kwargs):
        """Log a structured message to Cloud Logging."""
        self.cloud_logger.log_struct(struct, severity=severity)
        if self.local_logger:
            self._log_locally(
                f"Structured log: {struct}", getattr(logging, severity), **kwargs
            )


@lru_cache()
def get_logger(
    logger_name: str = "default",
) -> CloudLogger:
    """
    Get or create a CloudLogger instance.
    Uses connection configuration from the specified YAML file.

    Args:
        logger_name: Name of the logger (default: 'default')
        config_path: Path to the connections configuration file (default: 'connections.yaml')
        connection_name: Name of the BigQuery connection to use (default: 'bigquery_datamancers')

    Returns:
        CloudLogger instance
    """

    return CloudLogger(
        project_id=LOGGING["project_id"],
        credentials_path=LOGGING["credentials_path"],
        logger_name=logger_name,
    )


# Default logger instance
default_logger = get_logger()
