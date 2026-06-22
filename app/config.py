# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "" # Placeholder for OpenAI API Key
    GOOGLE_API_KEY: str = "" # Placeholder for Google API Key
    GROQ_API_KEY: str = "" # Placeholder for Groq API Key

    class Config:
        env_file = ".env"
    
settings = Settings()