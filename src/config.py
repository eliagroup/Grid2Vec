import pydantic

from . import enums


class Settings(pydantic.BaseSettings):
    """Setting class."""

    LOGGING_LEVEL: enums.LoggingLevel = pydantic.Field(
        default=enums.LoggingLevel.DEBUG,
    )
    PROJECT_NAME: str = pydantic.Field(
        default="template",
    )

    class Config:
        """Settings class configuration."""

        frozen = True
        case_sensitive = True
        env_file = "./.env"
        env_file_encoding = "utf-8"


settings = Settings()
