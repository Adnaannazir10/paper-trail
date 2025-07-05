"""
Embeddings utility for creating vector embeddings from text chunks.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Manages text embeddings using the specified model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embeddings manager.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        # Don't load model immediately - load on first use
    
    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is not None:
            return
            
        try:
            logger.info(f"Loading embeddings model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading embeddings model: {e}")
            raise
    
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            if not texts:
                logger.warning("No texts provided for embedding")
                return []
            
            logger.info(f"Creating embeddings for {len(texts)} texts")
            
            # Create embeddings in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._load_model()  # Load model if not already loaded
            
            embeddings = await loop.run_in_executor(
                None, 
                lambda: self.model.encode(texts, convert_to_tensor=False) # type: ignore
            )
            
            # Convert to list of lists if needed
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            logger.info(f"Successfully created {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    async def create_single_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            logger.info("Creating single embedding")
            
            # Create embedding in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._load_model()  # Load model if not already loaded
            
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.model.encode([text], convert_to_tensor=False) # type: ignore
            )
            
            # Convert to list if needed
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Return first (and only) embedding
            return embedding[0] if isinstance(embedding, list) else embedding
            
        except Exception as e:
            logger.error(f"Error creating single embedding: {e}")
            raise
    
    async def add_embeddings_to_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            List[Dict[str, Any]]: Chunks with added 'embedding' field
        """
        try:
            if not chunks:
                logger.warning("No chunks provided for embedding")
                return chunks
            
            logger.info(f"Adding embeddings to {len(chunks)} chunks")
            
            # Extract texts from chunks
            texts = [chunk['text'] for chunk in chunks]
            
            # Create embeddings
            embeddings = await self.create_embeddings(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
            
            logger.info(f"Successfully added embeddings to {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error adding embeddings to chunks: {e}")
            raise
    
    async def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            int: Dimension of the embedding vectors
        """
        try:
            # Create a dummy embedding to get the dimension
            dummy_embedding = await self.create_single_embedding("test")
            return len(dummy_embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            raise

# Global instance for reuse
embeddings_manager = EmbeddingsManager() 