import enum


@enum.unique
class LoggingLevel(str, enum.Enum):
    """Enumerates possible logging levels."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    NOTSET = "NOTSET"
