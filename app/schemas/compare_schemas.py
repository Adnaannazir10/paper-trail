from pydantic import BaseModel, Field
from typing import List


class ComparePapersRequest(BaseModel):
    """Request model for comparing papers."""
    paper_names: List[str] = Field(..., description="List of paper names to compare", min_length=2, max_length=5)


class ComparePapersResponse(BaseModel):
    """Response model for compare papers endpoint."""
    comparison_text: str = Field(..., description="Comparison analysis as a single text response") 