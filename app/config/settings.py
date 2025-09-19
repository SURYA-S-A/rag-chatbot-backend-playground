from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LLM_API_KEY: str
    LLM_PROVIDER: str
    LLM_MODEL: str
    QDRANT_API_KEY: str
    QDRANT_URL: str

    model_config = SettingsConfigDict(env_file="../../.env", env_prefix="RAG_CHATBOT_")


settings = Settings()