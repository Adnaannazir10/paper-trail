"""
Qdrant Vector Database Client Configuration
Handles connection setup and basic operations for the vector database.
"""

from typing import Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

import logging

from config.qdrant_config import qdrant_settings
from schemas.qdrant_schemas import QdrantCollectionSchema

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manages Qdrant vector database connections and operations."""
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.collection_name: str = QdrantCollectionSchema.COLLECTION_NAME
        self.vector_size: int = QdrantCollectionSchema.VECTOR_SIZE
        
    def get_client(self) -> QdrantClient:
        """Get or create Qdrant client instance."""
        if self.client is None:
            try:
                # Initialize Qdrant client
                if qdrant_settings.API_KEY:
                    self.client = QdrantClient(
                        host=qdrant_settings.HOST,
                        port=qdrant_settings.PORT,
                        https=qdrant_settings.USE_HTTPS
                    )
                else:
                    self.client = QdrantClient(
                        host=qdrant_settings.HOST,
                        port=qdrant_settings.PORT,
                        https=qdrant_settings.USE_HTTPS
                    )
                
                logger.info(f"Qdrant client initialized - Host: {qdrant_settings.HOST}:{qdrant_settings.PORT}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant client: {e}")
                raise
                
        return self.client
    
    def test_connection(self) -> bool:
        """Test connection to Qdrant server."""
        try:
            client = self.get_client()
            # Try to get collections to test connection
            collections = client.get_collections()
            logger.info(f"Qdrant connection successful. Found {len(collections.collections)} collections.")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            return False
    
    def create_collection_if_not_exists(self) -> bool:
        """Create the main collection if it doesn't exist."""
        try:
            client = self.get_client()
            
            # Check if collection exists
            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists.")
                return True
            
            # Create collection with proper configuration and payload schema
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000
                )
            )
            
            # Create payload indexes for efficient filtering
            self._create_payload_indexes(client)
            
            logger.info(f"Collection '{self.collection_name}' created successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{self.collection_name}': {e}")
            return False
    
    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the main collection."""
        try:
            client = self.get_client()
            collection_info = client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": self.vector_size,
                "points_count": collection_info.points_count
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def _create_payload_indexes(self, client: QdrantClient) -> None:
        """Create payload indexes for efficient filtering and search."""
        try:
            # For now, we'll create basic indexes. Advanced indexing can be added later
            logger.info("Payload indexes will be created as needed during data insertion.")
        except Exception as e:
            logger.error(f"Error setting up payload indexes: {e}")


# Global instance for easy access
qdrant_manager = QdrantManager() 