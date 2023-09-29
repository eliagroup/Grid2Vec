import logging

from src.enums import LoggingLevel
from src.logger import _get_logger


def test_name() -> None:
    name = "test"

    logger = _get_logger(
        name=name,
        level=LoggingLevel.DEBUG,
    )

    assert logger.name == name


def test_level(caplog) -> None:
    logger = _get_logger(
        name="Test",
        level=LoggingLevel.ERROR,
    )

    msg = "hello world"

    with caplog.at_level(logging.INFO):
        logger.info(msg)
        assert msg not in caplog.text

        logger.error(msg)
        assert msg in caplog.text
