import uvicorn
from config.app_config import settings # type: ignore

if __name__ == "__main__":
    uvicorn.run("app:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True) 