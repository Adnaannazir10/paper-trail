"""
Journal operations utility for retrieving and managing journal-specific data.
"""

import logging
from typing import List, Dict, Any
from fastapi import HTTPException
from qdrant_client.http import models

from .qdrant_client import qdrant_manager
from schemas.journal_schemas import JournalChunk, JournalListItem, JournalListResponse, JournalMetadata, JournalResponse

logger = logging.getLogger(__name__)


class JournalOperations:
    """Handles journal-specific operations and data retrieval."""
    
    def __init__(self):
        self.qdrant_manager = qdrant_manager
    
    async def get_journal_chunks(self, journal_id: str) -> List[JournalChunk]:
        """
        Get all chunks for a specific journal.
        
        Args:
            journal_id: Journal identifier (journal name)
            
        Returns:
            List of JournalChunk objects containing all chunks for the journal
        """
        try:
            logger.info(f"Retrieving all chunks for journal: {journal_id}")
            
            # Search for all chunks with the specified journal
            search_results = await self._search_journal_chunks(journal_id)
            
            # Format results
            formatted_results = self._format_journal_results(search_results)
            
            logger.info(f"Retrieved {len(formatted_results)} chunks for journal: {journal_id}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving journal chunks for {journal_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _search_journal_chunks(self, journal_id: str) -> List[Any]:
        """
        Search Qdrant for all chunks belonging to a specific journal.
        
        Args:
            journal_id: Journal identifier
            
        Returns:
            List of Qdrant search results
        """
        try:
            client = self.qdrant_manager.get_client()
            
            # Create a dummy vector for metadata-only search
            dummy_vector = [0.0] * self.qdrant_manager.vector_size
            
            # Search with journal filter
            search_results = client.search(
                collection_name=self.qdrant_manager.collection_name,
                query_vector=dummy_vector,
                query_filter=self._create_journal_filter(journal_id),
                limit=1000,  # Large limit to get all chunks
                with_payload=True,
                with_vectors=False
            )
            
            logger.info(f"Found {len(search_results)} chunks for journal: {journal_id}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching journal chunks: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _create_journal_filter(self, journal_id: str) -> Any:
        """
        Create a filter for journal search.
        
        Args:
            journal_id: Journal identifier
            
        Returns:
            Qdrant filter object
        """
        
        # Try exact match first
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="journal",
                    match=models.MatchValue(value=journal_id)
                )
            ]
        )
    
    def _format_journal_results(self, search_results: List[Any]) -> List[JournalChunk]:
        """
        Format Qdrant search results into JournalChunk objects for journal chunks.
        
        Args:
            search_results: Raw Qdrant search results
            
        Returns:
            List of formatted JournalChunk objects
        """
        formatted_results = []
        
        for result in search_results:
            try:
                # Extract payload data
                payload = result.payload
                
                # Create JournalChunk object
                journal_chunk = JournalChunk(
                    chunk_id=str(result.id),
                    content=payload.get("text", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    usage_count=payload.get("usage_count", 0),
                    source_doc_id=payload.get("source_doc_id", "unknown")
                )
                
                formatted_results.append(journal_chunk)
                
            except Exception as e:
                logger.warning(f"Error formatting journal result {result.id}: {e}")
                continue
        
        # Sort by chunk_index for logical document order
        formatted_results.sort(key=lambda x: x.chunk_index)
        
        logger.info(f"Formatted {len(formatted_results)} journal results")
        return formatted_results
    
    async def get_journal_metadata(self, journal_id: str) -> JournalMetadata:
        """
        Get metadata about a specific journal.
        
        Args:
            journal_id: Journal identifier
            
        Returns:
            JournalMetadata object containing journal metadata
        """
        try:
            logger.info(f"Retrieving metadata for journal: {journal_id}")
            
            # Get all chunks for the journal
            chunks = await self.get_journal_chunks(journal_id)
            
            if not chunks:
                raise HTTPException(status_code=404, detail=f"Journal not found: {journal_id}")
            
            # Calculate statistics
            total_chunks = len(chunks)
            
            # Create JournalMetadata object
            metadata = JournalMetadata(
                journal_id=journal_id,
                total_chunks=total_chunks
            )
            
            logger.info(f"Retrieved metadata for journal: {journal_id}")
            return metadata
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving journal metadata for {journal_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_journal_response(self, journal_id: str) -> JournalResponse:
        """
        Get complete journal response with metadata and chunks.
        
        Args:
            journal_id: Journal identifier
            
        Returns:
            JournalResponse object containing metadata and all chunks for the journal
        """
        try:
            logger.info(f"Retrieving complete journal response for: {journal_id}")
            
            # Get journal metadata
            metadata = await self.get_journal_metadata(journal_id)
            
            # Get all chunks for the journal
            chunks = await self.get_journal_chunks(journal_id)
            
            # Create complete response
            response = JournalResponse(
                metadata=metadata,
                chunks=chunks
            )
            
            logger.info(f"Successfully created journal response for {journal_id} with {len(chunks)} chunks")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating journal response for {journal_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_available_journals(self) -> JournalListResponse:
        """
        Get list of all available journals.
        
        Returns:
            List of journal metadata dictionaries
        """
        try:
            logger.info("Retrieving list of available journals")
            
            client = self.qdrant_manager.get_client()
            
            # Get all unique journal names
            search_results = client.scroll(
                collection_name=self.qdrant_manager.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            # Extract unique journals
            journals = {}
            for result in search_results[0]:
                if result.payload:
                    journal_name = result.payload.get("journal", "")
                    if journal_name and journal_name not in journals:
                        journals[journal_name] = {
                            "journal_name": journal_name,
                        }
            
            journal_list = list(journals.values())
            journal_list.sort(key=lambda x: x["journal_name"])
            
            logger.info(f"Found {len(journal_list)} available journals")
            journal_items = [
                JournalListItem(
                    journal_name=journal["journal_name"],
                )
                for journal in journal_list
            ]
            response = JournalListResponse(
                journals=journal_items,
                total_count=len(journal_items)
            )

            return response
            
        except Exception as e:
            logger.error(f"Error retrieving available journals: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_doc_chunks(self, source_doc_id: str) -> List[JournalChunk]:
        """
        Get all chunks for a specific document by source_doc_id.
        """
        try:
            logger.info(f"Retrieving all chunks for document: {source_doc_id}")
            client = self.qdrant_manager.get_client()
            dummy_vector = [0.0] * self.qdrant_manager.vector_size
            # Search with source_doc_id filter
            search_results = client.search(
                collection_name=self.qdrant_manager.collection_name,
                query_vector=dummy_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_doc_id",
                            match=models.MatchValue(value=source_doc_id)
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            formatted_results = self._format_journal_results(search_results)
            logger.info(f"Retrieved {len(formatted_results)} chunks for document: {source_doc_id}")
            return formatted_results
        except Exception as e:
            logger.error(f"Error retrieving doc chunks for {source_doc_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Global instance for reuse
journal_operations = JournalOperations() 