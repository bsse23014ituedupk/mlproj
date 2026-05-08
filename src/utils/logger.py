"""
Structured logging utility for the Water Potability Classifier.

Provides a pre-configured logger with:
- Console output with colour-coded severity levels
- Consistent timestamp formatting
- Module-level logger factory so every file gets a named logger

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

import logging
import sys
from src.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Create and return a named logger with structured formatting.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

        # Console handler — human-readable format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

        # Format: [TIMESTAMP] LEVEL — module — message
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Prevent log propagation to the root logger (avoids duplicate output)
        logger.propagate = False

    return logger
