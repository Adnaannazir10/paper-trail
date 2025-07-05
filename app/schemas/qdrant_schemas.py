"""
Qdrant Vector Database Schemas
Defines the metadata structure for document chunks stored in Qdrant.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ChunkMetadata:
    """Metadata structure for document chunks stored in Qdrant."""
    
    # Core identification
    id: str  # Unique chunk identifier (e.g., "mucuna_01_intro")
    source_doc_id: str  # Original document ID (e.g., "extension_brief_mucuna.pdf")
    chunk_index: int  # Position of chunk in document (1-based)
    
    # Document information
    journal: str  # Journal or publication name (e.g., "ILRI extension brief")
    publish_year: int  # Publication year
    link: str  # Source URL or DOI
    
    # Content organization
    section_heading: str  # Section title or heading
    attributes: List[str]  # Tags/categories (e.g., ["Botanical description", "Morphology"])
    
    # Usage tracking
    usage_count: int = 0  # Number of times this chunk has been accessed
    



@dataclass
class ChunkPayload:
    """Complete chunk payload for Qdrant storage including vector and metadata."""
    
    id: str
    vector: List[float]  # Embedding vector
    payload: ChunkMetadata
    text: str  # Original text content


class QdrantCollectionSchema:
    """Schema definition for Qdrant collection structure."""
    
    # Collection configuration
    COLLECTION_NAME = "research_documents"
    VECTOR_SIZE = 384  # NoInstruct-small-Embedding-v0 dimension
    DISTANCE_METRIC = "Cosine"
    
    # Payload field definitions for Qdrant
    PAYLOAD_FIELDS = {
        "id": {"type": "keyword", "index": True},
        "source_doc_id": {"type": "keyword", "index": True},
        "chunk_index": {"type": "integer", "index": True},
        "journal": {"type": "keyword", "index": True},
        "publish_year": {"type": "integer", "index": True},
        "link": {"type": "keyword", "index": False},
        "section_heading": {"type": "text", "index": True},
        "attributes": {"type": "keyword", "index": True},
        "usage_count": {"type": "integer", "index": True},
        "text": {"type": "text", "index": False}  # Full text content
    }
    
    @classmethod
    def get_payload_schema(cls) -> Dict[str, Any]:
        """Get the payload schema for Qdrant collection creation."""
        return {
            "text": {"type": "text"},
            "source_doc_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "journal": {"type": "keyword"},
            "publish_year": {"type": "integer"},
            "link": {"type": "keyword"},
            "section_heading": {"type": "text"},
            "attributes": {"type": "keyword"},
            "usage_count": {"type": "integer"}
        }


def create_chunk_id(source_doc_id: str, chunk_index: int, section_heading: str = "") -> str:
    """Generate a unique chunk ID based on document and position."""
    # Clean the section heading for use in ID
    clean_heading = section_heading.lower().replace(" ", "_").replace("-", "_")
    clean_heading = "".join(c for c in clean_heading if c.isalnum() or c == "_")
    
    # Remove file extension from source_doc_id
    doc_name = source_doc_id.replace(".pdf", "").replace(".PDF", "")
    
    if clean_heading:
        return f"{doc_name}_{chunk_index:02d}_{clean_heading}"
    else:
        return f"{doc_name}_{chunk_index:02d}"


def metadata_to_dict(metadata: ChunkMetadata) -> Dict[str, Any]:
    """Convert ChunkMetadata to dictionary for Qdrant storage."""
    return {
        "id": metadata.id,
        "source_doc_id": metadata.source_doc_id,
        "chunk_index": metadata.chunk_index,
        "journal": metadata.journal,
        "publish_year": metadata.publish_year,
        "link": metadata.link,
        "section_heading": metadata.section_heading,
        "attributes": metadata.attributes,
        "usage_count": metadata.usage_count
    } 