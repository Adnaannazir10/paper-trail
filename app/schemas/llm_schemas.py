from pydantic import BaseModel, Field
from typing import Optional


class AskLLMRequest(BaseModel):
    """Request model for asking the LLM a question."""
    query: str = Field(..., description="The question to ask the LLM")
    journal: Optional[str] = Field(None, description="Optional journal filter to limit context")


class AskLLMResponse(BaseModel):
    """Response model for LLM answers."""
    llm_response: str