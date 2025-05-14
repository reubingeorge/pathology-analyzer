import datetime
import logging
from pathlib import Path


def setup_logging(log_dir: Path = Path("logs")):
    """Configure logging for the Pathology Analyzer application.

    This function sets up logging with two handlers:
    1. A console handler that logs to stdout
    2. A file handler that logs to a timestamped file in the log_dir directory

    Args:
        log_dir (Path): Directory where log files will be stored

    Returns:
        logging.Logger: The configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(exist_ok=True)

    # Get current timestamp for log filename
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"pathology_analyzer_{current_time}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

    # Configure 3rd party loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Create logger for this module
    logger = logging.getLogger("pathology_analyzer")
    logger.info(f"Starting Pathology Report Analyzer. Logs will be saved to {log_file}")

    return logger


def get_logger(name: str = None):
    """Get a logger with the given name.

    If no name is provided, returns the root pathology_analyzer logger.

    Args:
        name (str, optional): Logger name, which will be prefixed with "pathology_analyzer."

    Returns:
        logging.Logger: The requested logger instance
    """
    if name:
        return logging.getLogger(f"pathology_analyzer.{name}")
    return logging.getLogger("pathology_analyzer")