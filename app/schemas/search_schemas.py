"""
Search request schemas for similarity search and other search operations.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class SimilaritySearchRequest(BaseModel):
    """Schema for similarity search request."""
    
    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    k: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    min_score: float = Field(default=0.25, description="Minimum similarity score threshold", ge=0.0, le=1.0)
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('query cannot be empty or whitespace only')
        return v.strip()
    
    @field_validator('k')
    @classmethod
    def validate_k(cls, v):
        if v < 1:
            raise ValueError('k must be at least 1')
        if v > 100:
            raise ValueError('k cannot exceed 100')
        return v
    
    @field_validator('min_score')
    @classmethod
    def validate_min_score(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('min_score must be between 0.0 and 1.0')
        return v


class SearchResult(BaseModel):
    """Schema for individual search result."""
    
    id: str = Field(..., description="Chunk ID")
    score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)
    text: str = Field(..., description="Chunk text content")
    source_doc_id: str = Field(..., description="Source document ID")
    chunk_index: int = Field(..., description="Chunk index in document")
    journal: str = Field(..., description="Journal name")
    publish_year: int = Field(..., description="Publication year")
    link: str|None = Field(None, description="Source document link")
    section_heading: str|None = Field(None, description="Section heading")
    attributes: List[str] = Field(..., description="Chunk attributes/tags")
    usage_count: int = Field(..., description="Usage count")


class SimilaritySearchResponse(BaseModel):
    """Schema for similarity search response."""
    
    query: str = Field(..., description="Original search query")
    k: int = Field(..., description="Number of results requested")
    min_score: float = Field(..., description="Minimum score threshold used")
    total_results: int = Field(..., description="Total number of results found")
    results: List[SearchResult] = Field(..., description="Search results")