from pydantic import BaseModel, Field
from typing import List, Optional


class SummaryResponse(BaseModel):
    """Response model for journal summary endpoint."""
    paper_name: str = Field(..., description="Paper name")
    summary: str = Field(..., description="Summary of the journal content")