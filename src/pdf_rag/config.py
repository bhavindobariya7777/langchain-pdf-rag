"""Validated environment configuration."""

from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PDF_RAG_", env_file=".env", extra="ignore")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    documents_dir: Path = Path("documents")
    data_dir: Path = Path("data")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=4, ge=1, le=20)
    scan_hour: int = Field(default=2, ge=0, le=23)
    scan_minute: int = Field(default=0, ge=0, le=59)
    timezone: str = "Asia/Kolkata"

    @model_validator(mode="after")
    def validate_settings(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        ZoneInfo(self.timezone)
        return self
