"""
Structured logging utility for the Water Potability Classifier.

Provides a pre-configured logger with:
- Console output with colour-coded severity levels (via StreamHandler)
- File output that persists logs to training.log (via FileHandler)
- Consistent timestamp formatting
- Module-level logger factory so every file gets a named logger

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Create and return a named logger with structured formatting.

    Handlers are attached lazily — only on the first call for a given name.
    Both console (stdout) and file (training.log) outputs are configured.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    # Lazy import to avoid circular import at module level
    from src.config import LOG_LEVEL, LOG_FILE

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if not logger.handlers:
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)

        fmt = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # ── Console handler (stdout) ──────────────────────────────────────────
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

        # ── File handler (training.log) ───────────────────────────────────────
        try:
            log_path = Path(LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except Exception:
            # Never crash because of logging setup
            pass

        # Prevent log propagation to the root logger (avoids duplicate output)
        logger.propagate = False

    return logger
