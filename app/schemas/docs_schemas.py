from pydantic import BaseModel, Field
from typing import List

class DocumentListItem(BaseModel):
    """Single document item for docs/list response."""
    source_doc_id: str = Field(..., description="Source document ID (filename)")
    paper_name: str = Field(..., description="Extracted paper name (title)")

class DocumentListResponse(BaseModel):
    """Response model for /docs/list endpoint."""
    documents: List[DocumentListItem] = Field(..., description="List of available documents")
    total_count: int = Field(..., description="Total number of documents available") 