import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_PORT: int = int(os.environ.get("APP_PORT", 8000))
    APP_HOST: str = os.environ.get("APP_HOST", "0.0.0.0")

settings = Settings()