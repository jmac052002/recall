# recall/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict 


class Settings(BaseSettings):
    """All config sourced from enviroment variables / .env file.
    Pydantic validates types at - you'll know immediately if something's missing.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # Anthropic
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-6"
    
    # Voyage AI
    voyage_api_key: str
    voyage_model: str = "voyage-3-lite"
    embedding_dimensions: int = 512 
    
    # PostgreSQL
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    
    # SQLite
    recall_db_path: Path = Path.home() / ".local/share/recall/recall.db"
    
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings() 