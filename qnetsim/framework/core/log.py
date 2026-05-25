from __future__ import annotations

import logging


def log_init(name: str = "qnetsim.framework", level: int = logging.INFO) -> logging.Logger:
    """Create a simple logger for the standalone framework.

    The standalone bundle does not rely on upstream QNET logging utilities, but
    several DES-oriented helpers still expect a `log_init` function.
    """

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger

