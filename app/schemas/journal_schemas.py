from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class JournalChunk(BaseModel):
    """Schema for a single chunk in a journal"""
    chunk_id: str
    content: str
    chunk_index: int
    usage_count: int = 0
    source_doc_id: str = 'unknown'


class JournalMetadata(BaseModel):
    """Schema for journal metadata"""
    journal_id: str
    total_chunks: int


class JournalResponse(BaseModel):
    """Schema for journal response containing metadata and chunks"""
    metadata: JournalMetadata
    chunks: List[JournalChunk]


class JournalListItem(BaseModel):
    """Schema for a single journal in the list"""
    journal_name: str

class JournalListResponse(BaseModel):
    """Schema for journal list response"""
    journals: List[JournalListItem]
    total_count: int 