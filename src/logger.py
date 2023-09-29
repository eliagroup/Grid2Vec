import logging

from rich.logging import RichHandler

from . import enums
from .config import settings

logger = logging.getLogger(name=settings.PROJECT_NAME)
logger.setLevel(settings.LOGGING_LEVEL.value)


def _get_logger(name: str, level: enums.LoggingLevel) -> logging.Logger:
    # Configure logging:
    # - set level to ERROR
    # - add Rich handler
    logging.basicConfig(
        level="ERROR",
        format="%(name)s: %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=True,
            ),
        ],
    )

    # Initialize logger and configure logging level
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level.value)

    return logger


logger = _get_logger(
    name=settings.PROJECT_NAME,
    level=settings.LOGGING_LEVEL,
)
