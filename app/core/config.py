import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Options: "openai", "anthropic", "ollama"
    MODEL_PROVIDER: str = "openai"
    
    OPENAI_MODEL: str = "gpt-5-nano"
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"
    OLLAMA_MODEL: str = "llama3.2"

    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_VECTOR_INDEX: str = "cato_hei_index"
    PROPERTIES_DB_NAME: str = "properties.db"
    
    APP_NAME: str = "Cato Agentic AI"
    DEBUG: bool = False
    DEFAULT_SESSION_TTL: int = 3600  # 1 hour

    # Configuration for .env loading
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY