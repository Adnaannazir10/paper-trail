from contextlib import asynccontextmanager
from fastapi import FastAPI
from api import router as api_router
from utils.qdrant_client import qdrant_manager
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup and shutdown events."""
    # Startup
    logger.info("Starting Research Assistant Service...")
    
    # Initialize Qdrant
    try:
        logger.info("Initializing Qdrant vector database...")
        if await qdrant_manager.test_connection():
            await qdrant_manager.create_collection_if_not_exists()
            logger.info("Qdrant initialization completed successfully.")
        else:
            logger.error("Failed to connect to Qdrant. Please check your configuration.")
    except Exception as e:
        logger.error(f"Error during Qdrant initialization: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Research Assistant Service...")


app = FastAPI(
    title="Research Assistant Service", 
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api")