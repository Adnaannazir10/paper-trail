"""
Upload request schemas for the /api/upload endpoint.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
import re
from fastapi import UploadFile


class ChunkObject(BaseModel):
    """Schema for individual chunk objects when client provides pre-chunked content."""
    
    id: str = Field(..., description="Unique identifier for the chunk")
    source_doc_id: str = Field(..., description="Source document identifier")
    chunk_index: int = Field(..., description="Index of the chunk in the document")
    section_heading: str = Field(..., description="Section heading for the chunk")
    doi: str = Field(..., description="Digital Object Identifier")
    journal: str = Field(..., description="Journal name")
    publish_year: int = Field(..., description="Publication year")
    usage_count: int = Field(default=0, description="Usage count for the chunk")
    attributes: List[str] = Field(..., description="List of attributes/tags for the chunk")
    link: str = Field(..., description="Link to the source document")
    text: str = Field(..., description="Text content of the chunk")
    
    @field_validator('publish_year')
    @classmethod
    def validate_publish_year(cls, v):
        if v < 1900 or v > 2100:
            raise ValueError('publish_year must be between 1900 and 2100')
        return v
    
    @field_validator('usage_count')
    @classmethod
    def validate_usage_count(cls, v):
        if v < 0:
            raise ValueError('usage_count must be non-negative')
        return v
    
    @field_validator('link')
    @classmethod
    def validate_link(cls, v):
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        if not url_pattern.match(v):
            raise ValueError('link must be a valid URL')
        return v


class UploadRequest(BaseModel):
    """Schema for the /api/upload request body (JSON payload)."""
    chunks: Optional[List[ChunkObject]] = Field(None, description="Pre-chunked content from client")