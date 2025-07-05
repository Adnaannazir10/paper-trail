"""
Qdrant Vector Database Client Configuration
Handles connection setup and basic operations for the vector database.
"""

import asyncio
from typing import Optional, Dict, Any, List
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    KeywordIndexParams, IntegerIndexParams, TextIndexParams,
    KeywordIndexType, IntegerIndexType, TextIndexType
)

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
    
    async def test_connection(self) -> bool:
        """Test connection to Qdrant server."""
        try:
            client = self.get_client()
            # Try to get collections to test connection
            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(None, client.get_collections)
            logger.info(f"Qdrant connection successful. Found {len(collections.collections)} collections.")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            return False
    
    async def create_collection_if_not_exists(self) -> bool:
        """Create the main collection if it doesn't exist."""
        try:
            client = self.get_client()
            loop = asyncio.get_event_loop()
            
            # Check if collection exists
            collections = await loop.run_in_executor(None, client.get_collections)
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists.")
                return True
            
            # Create collection with proper configuration and payload schema
            await loop.run_in_executor(
                None,
                lambda: client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        memmap_threshold=20000
                    ),
                    # Define payload schema for metadata structure
                    on_disk_payload=True  # Store payload on disk for large collections
                )
            )
            
            # Create payload indexes for efficient filtering
            await self._create_payload_indexes(client)
            
            logger.info(f"Collection '{self.collection_name}' created successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{self.collection_name}': {e}")
            return False
    
    async def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the main collection."""
        try:
            client = self.get_client()
            loop = asyncio.get_event_loop()
            collection_info = await loop.run_in_executor(None, lambda: client.get_collection(self.collection_name))
            return {
                "name": self.collection_name,
                "vector_size": self.vector_size,
                "points_count": collection_info.points_count
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    async def _create_payload_indexes(self, client: QdrantClient) -> None:
        """Create payload indexes for efficient filtering and search."""
        try:
            # Define indexes for key metadata fields that will be frequently filtered
            indexes_to_create = [
                ("source_doc_id", KeywordIndexParams(type=KeywordIndexType.KEYWORD)),
                ("journal", KeywordIndexParams(type=KeywordIndexType.KEYWORD)),
                ("publish_year", IntegerIndexParams(type=IntegerIndexType.INTEGER)),
                ("usage_count", IntegerIndexParams(type=IntegerIndexType.INTEGER)),
                ("chunk_index", IntegerIndexParams(type=IntegerIndexType.INTEGER)),
                ("section_heading", TextIndexParams(type=TextIndexType.TEXT)),
                ("attributes", KeywordIndexParams(type=KeywordIndexType.KEYWORD))
            ]
            
            logger.info("Creating payload indexes for efficient metadata filtering...")
            loop = asyncio.get_event_loop()
            
            for field_name, index_params in indexes_to_create:
                try:
                    # Create payload index for the field
                    await loop.run_in_executor(
                        None,
                        lambda: client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema=index_params
                        )
                    )
                    logger.info(f"Created payload index for field: {field_name}")
                    
                except Exception as e:
                    # Log warning but continue with other indexes
                    logger.warning(f"Failed to create index for {field_name}: {e}")
                    continue
            
            logger.info("Payload index creation completed.")
            
        except Exception as e:
            logger.error(f"Error setting up payload indexes: {e}")
    
    async def increment_usage_count(self, chunk_id: str) -> bool:
        """
        Increment the usage count for a specific chunk.
        
        Args:
            chunk_id: The ID of the chunk to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            client = self.get_client()
            loop = asyncio.get_event_loop()
            
            # First, get the current usage count
            current_points = await loop.run_in_executor(
                None,
                lambda: client.retrieve(
                    collection_name=self.collection_name,
                    ids=[chunk_id],
                    with_payload=True
                )
            )
            
            if not current_points:
                logger.warning(f"Chunk {chunk_id} not found")
                return False
            
            current_count = current_points[0].payload.get("usage_count", 0) if current_points[0].payload else 0
            new_count = current_count + 1
            
            # Update with the new incremented value
            await loop.run_in_executor(
                None,
                lambda: client.set_payload(
                    collection_name=self.collection_name,
                    payload={"usage_count": new_count},
                    points=[chunk_id],
                    wait=True
                )
            )
            
            logger.debug(f"Incremented usage count for chunk {chunk_id}: {current_count} -> {new_count}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to increment usage count for chunk {chunk_id}: {e}")
            return False
    
    async def save_chunks_with_embeddings(self, chunks_with_embeddings: List[Dict[str, Any]]) -> bool:
        """
        Save chunks with their embeddings and metadata to Qdrant.
        
        Args:
            chunks_with_embeddings: List of chunk dictionaries with 'embedding' field
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not chunks_with_embeddings:
                logger.warning("No chunks provided for saving")
                return False
            
            client = self.get_client()
            loop = asyncio.get_event_loop()
            
            # Prepare points for Qdrant
            points = []
            for chunk in chunks_with_embeddings:
                # Extract required fields
                chunk_id = uuid.uuid4()
                embedding = chunk.get('embedding')
                text = chunk.get('text')
                
                if not chunk_id or embedding is None or not text:
                    logger.warning(f"Skipping chunk with missing required fields: {chunk_id}")
                    continue
                
                # Prepare payload (metadata)
                payload = {
                    "id": chunk_id,
                    "text": text,
                    "source_doc_id": chunk.get('source_doc_id', ''),
                    "chunk_index": chunk.get('chunk_index', 0),
                    "journal": chunk.get('journal', ''),
                    "publish_year": chunk.get('publish_year', 0),
                    "link": chunk.get('link', ''),
                    "section_heading": chunk.get('section_heading', ''),
                    "attributes": chunk.get('attributes', []),
                    "usage_count": chunk.get('usage_count', 0)
                }
                
                # Create point for Qdrant
                point = models.PointStruct(
                    id=str(chunk_id),
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            if not points:
                logger.warning("No valid points to save")
                return False
            
            # Upsert points to Qdrant
            await loop.run_in_executor(
                None,
                lambda: client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
            )
            
            logger.info(f"Successfully saved {len(points)} chunks to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save chunks to Qdrant: {e}")
            return False
    
    async def save_single_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        Save a single chunk with embedding to Qdrant.
        
        Args:
            chunk: Chunk dictionary with 'embedding' field
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return await self.save_chunks_with_embeddings([chunk])
        except Exception as e:
            logger.error(f"Failed to save single chunk: {e}")
            return False
    
    async def get_most_cited_chunks(self, limit: int = 10) -> list:
        """
        Get the most cited/used chunks based on usage_count.
        
        Args:
            limit: Number of results to return
            
        Returns:
            List of chunks sorted by usage count
        """
        try:
            client = self.get_client()
            loop = asyncio.get_event_loop()
            
            # Search with filter for chunks with usage_count > 0, sorted by usage_count desc
            results = await loop.run_in_executor(
                None,
                lambda: client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="usage_count",
                                range=models.Range(
                                    gte=1
                                )
                            )
                        ]
                    ),
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
            )
            
            # Sort by usage_count descending
            chunks = results[0]
            chunks.sort(key=lambda x: x.payload.get("usage_count", 0) if x.payload else 0, reverse=True)
            
            return chunks[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get most cited chunks: {e}")
            return []
    
    def validate_metadata_structure(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate that metadata structure matches our schema requirements.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Required fields from our schema
            required_fields = [
                "id", "source_doc_id", "chunk_index", "journal", 
                "publish_year", "link", "section_heading", "attributes", 
                "usage_count", "text"
            ]
            
            # Check required fields exist
            for field in required_fields:
                if field not in metadata:
                    logger.warning(f"Missing required field in metadata: {field}")
                    return False
            
            # Validate data types
            if not isinstance(metadata.get("usage_count"), int):
                logger.warning("usage_count must be an integer")
                return False
                
            if not isinstance(metadata.get("attributes"), list):
                logger.warning("attributes must be a list")
                return False
                
            if not isinstance(metadata.get("chunk_index"), int):
                logger.warning("chunk_index must be an integer")
                return False
            
            logger.debug("Metadata structure validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating metadata structure: {e}")
            return False


# Global instance for easy access
qdrant_manager = QdrantManager() 