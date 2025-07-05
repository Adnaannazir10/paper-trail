"""
Similarity search utility for finding similar document chunks.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Sequence
from fastapi import HTTPException
from qdrant_client.http import models

from .embeddings import embeddings_manager
from .qdrant_client import qdrant_manager
from schemas.search_schemas import SimilaritySearchRequest, SimilaritySearchResponse, SearchResult

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """Handles similarity search operations using embeddings and Qdrant."""
    
    def __init__(self):
        self.qdrant_manager = qdrant_manager
    
    async def search_similar_chunks(
        self, 
        query: str, 
        k: int = 10, 
        min_score: float = 0.25
    ) -> SimilaritySearchResponse:
        """
        Search for similar chunks using query embedding.
        
        Args:
            query: Search query text
            k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            SimilaritySearchResponse with search results
        """
        try:
            logger.info(f"Starting similarity search for query: '{query}' (k={k}, min_score={min_score})")
            
            # Step 1: Create embedding for the query
            logger.info("Creating query embedding...")
            query_embedding = await embeddings_manager.create_single_embedding(query)
            
            # Step 2: Search Qdrant for similar chunks
            logger.info("Searching Qdrant for similar chunks...")
            search_results = await self._search_qdrant(query_embedding, k, min_score)
            
            # Step 3: Format results
            logger.info("Formatting search results...")
            formatted_results = self._format_search_results(search_results)
            
            # Step 4: Increment usage count for all returned chunks
            logger.info("Incrementing usage count for returned chunks...")
            await self._increment_usage_counts([result.id for result in formatted_results])
            
            # Create response
            response = SimilaritySearchResponse(
                query=query,
                k=k,
                min_score=min_score,
                total_results=len(formatted_results),
                results=formatted_results,
            )
            
            logger.info(f"Similarity search completed: {len(formatted_results)} results found")
            return response
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _search_qdrant(
        self, 
        query_embedding: List[float], 
        k: int, 
        min_score: float
    ) -> List[Any]:
        """
        Search Qdrant for similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            min_score: Minimum score threshold
            
        Returns:
            List of Qdrant search results
        """
        try:
            client = self.qdrant_manager.get_client()
            
            # Perform vector search
            search_results = client.search(
                collection_name=self.qdrant_manager.collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=min_score,
                with_payload=True,
                with_vectors=False
            )
            
            logger.info(f"Qdrant search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _format_search_results(self, search_results: List[Any]) -> List[SearchResult]:
        """
        Format Qdrant search results into SearchResult objects.
        
        Args:
            search_results: Raw Qdrant search results
            
        Returns:
            List of formatted SearchResult objects
        """
        formatted_results = []
        
        for result in search_results:
            try:
                # Extract payload data
                payload = result.payload
                
                # Create SearchResult object
                search_result = SearchResult(
                    id=result.id,
                    score=result.score,
                    text=payload.get("text", ""),
                    source_doc_id=payload.get("source_doc_id", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    journal=payload.get("journal", ""),
                    publish_year=payload.get("publish_year", 0),
                    link=payload.get("link", ""),
                    section_heading=payload.get("section_heading", ""),
                    attributes=payload.get("attributes", []),
                    usage_count=payload.get("usage_count", 0)
                )
                
                formatted_results.append(search_result)
                
            except Exception as e:
                logger.warning(f"Error formatting search result {result.id}: {e}")
                continue
        
        logger.info(f"Formatted {len(formatted_results)} search results")
        return formatted_results
    
    async def search_with_filters(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.25,
        journal_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        source_doc_filter: Optional[str] = None
    ) -> SimilaritySearchResponse:
        """
        Search with additional filters.
        
        Args:
            query: Search query text
            k: Number of results to return
            min_score: Minimum similarity score threshold
            journal_filter: Filter by journal name
            year_filter: Filter by publication year
            source_doc_filter: Filter by source document ID
            
        Returns:
            SimilaritySearchResponse with filtered search results
        """
        try:
            logger.info(f"Starting filtered similarity search for query: '{query}'")
            
            # Create query embedding
            query_embedding = await embeddings_manager.create_single_embedding(query)
            
            # Build filter conditions
            filter_conditions = []
            
            if journal_filter:
                filter_conditions.append(
                    models.FieldCondition(
                        key="journal",
                        match=models.MatchValue(value=journal_filter)
                    )
                )
            
            if year_filter:
                filter_conditions.append(
                    models.FieldCondition(
                        key="publish_year",
                        match=models.MatchValue(value=year_filter)
                    )
                )
            
            if source_doc_filter:
                filter_conditions.append(
                    models.FieldCondition(
                        key="source_doc_id",
                        match=models.MatchValue(value=source_doc_filter)
                    )
                )
            
            # Perform filtered search
            if filter_conditions:
                search_results = await self._search_qdrant_with_filter(
                    query_embedding, k, min_score, filter_conditions
                )
            else:
                search_results = await self._search_qdrant(query_embedding, k, min_score)
            
            # Format results
            formatted_results = self._format_search_results(search_results)
            
            # Increment usage count for all returned chunks
            logger.info("Incrementing usage count for returned chunks...")
            await self._increment_usage_counts([result.id for result in formatted_results])
            
            # Create response
            response = SimilaritySearchResponse(
                query=query,
                k=k,
                min_score=min_score,
                total_results=len(formatted_results),
                results=formatted_results,
            )
            
            logger.info(f"Filtered similarity search completed: {len(formatted_results)} results found")
            return response
            
        except Exception as e:
            logger.error(f"Error in filtered similarity search: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _search_qdrant_with_filter(
        self,
        query_embedding: List[float],
        k: int,
        min_score: float,
        filter_conditions: Sequence[models.FieldCondition]
    ) -> List[Any]:
        """
        Search Qdrant with filter conditions.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            min_score: Minimum score threshold
            filter_conditions: List of filter conditions
            
        Returns:
            List of Qdrant search results
        """
        try:
            client = self.qdrant_manager.get_client()
            
            # Create filter
            search_filter = models.Filter(must=list(filter_conditions))
            
            # Perform filtered vector search
            search_results = client.search(
                collection_name=self.qdrant_manager.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=k,
                score_threshold=min_score,
                with_payload=True,
                with_vectors=False
            )
            
            logger.info(f"Filtered Qdrant search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in filtered Qdrant search: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _increment_usage_counts(self, chunk_ids: List[str]) -> None:
        """
        Increment usage count for multiple chunks.
        
        Args:
            chunk_ids: List of chunk IDs to increment usage count for
        """
        try:
            logger.info(f"Incrementing usage count for {len(chunk_ids)} chunks")
            
            # Increment usage count for each chunk
            for chunk_id in chunk_ids:
                try:
                    success = await self.qdrant_manager.increment_usage_count(chunk_id)
                    if not success:
                        logger.warning(f"Failed to increment usage count for chunk {chunk_id}")
                except Exception as e:
                    logger.warning(f"Error incrementing usage count for chunk {chunk_id}: {e}")
                    continue
            
            logger.info(f"Successfully incremented usage count for chunks")
            
        except Exception as e:
            logger.error(f"Error in batch usage count increment: {e}")
            # Don't raise exception here as it shouldn't break the search


# Global instance for reuse
similarity_search_manager = SimilaritySearch() 