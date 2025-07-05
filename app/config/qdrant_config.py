import os
from dotenv import load_dotenv

load_dotenv()

class QdrantSettings:
    """Qdrant Vector Database Configuration"""
    
    # Connection settings
    HOST: str = os.environ.get("QDRANT_HOST", "localhost")
    PORT: int = int(os.environ.get("QDRANT_PORT", 6333))
    API_KEY: str = os.environ.get("QDRANT_API_KEY", "")
    
    # Collection settings
    COLLECTION_NAME: str = os.environ.get("QDRANT_COLLECTION_NAME", "research_documents")
    VECTOR_SIZE: int = int(os.environ.get("QDRANT_VECTOR_SIZE", 384))  # NoInstruct-small-Embedding-v0 dimension
    
    # Connection timeout settings
    TIMEOUT: int = int(os.environ.get("QDRANT_TIMEOUT", 10))
    
    # SSL/HTTPS settings
    USE_HTTPS: bool = os.environ.get("QDRANT_USE_HTTPS", "false").lower() == "true"

qdrant_settings = QdrantSettings() 