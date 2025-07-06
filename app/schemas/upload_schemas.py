from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    message: str = Field(..., description="Status message")
    schema_version: str = Field(..., description="Schema version used")
    processing_mode: str = Field(..., description="Processing mode (background)")
    upload_type: str = Field(..., description="Type of upload (file, url, or chunks)")
    task_id: str = Field(..., description="Unique task identifier for tracking") 