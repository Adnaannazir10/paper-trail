from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, Form, File
from fastapi.responses import JSONResponse
from schemas.search_schemas import SimilaritySearchRequest
from schemas.journal_schemas import JournalResponse, JournalListResponse, JournalListItem
from typing import List, Dict, Any
from utils.similarity_search import similarity_search_manager
from utils.background_tasks import add_document_processing_task
from utils.journal_operations import journal_operations
from typing import Optional
import logging
import json
import uuid

logger = logging.getLogger(__name__)
router = APIRouter()

@router.put("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    schema_version: str = Form(..., description="Schema version"),
    file_url: Optional[str] = Form(None, description="URL to fetch the journal document"),
    upload_req: Optional[str] = Form(None, description="JSON string of UploadRequest"),
    file: Optional[UploadFile] = File(None, description="PDF file to upload")
):
    """
    Upload endpoint for journal documents.
    
    Accepts multipart form data with:
    - schema_version: Schema version
    - upload_req: JSON string containing chunks
    - file_url: URL to fetch the journal document
    - file: PDF file upload (optional if file_url provided)
    
    Returns 202 Accepted and processes document in background.
    """
    try:
        logger.info(f"Received upload request with schema_version: {schema_version}")
        
        # Parse upload_req if provided
        chunks = None
        if upload_req:
            try:
                upload_data = json.loads(upload_req)
                chunks = upload_data.get('chunks')
                logger.info(f"Parsed {len(chunks) if chunks else 0} chunks from upload_req")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in upload_req field")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid upload request data: {str(e)}")
        
        # Validate that exactly one option is provided: file, file_url, or chunks
        options_provided = 0
        if file is not None:
            options_provided += 1
        if file_url is not None:
            options_provided += 1
        if chunks is not None:
            options_provided += 1
        
        if options_provided == 0:
            raise HTTPException(
                status_code=400, 
                detail="Must provide exactly one of: file upload, file_url, or chunks"
            )
        elif options_provided > 1:
            raise HTTPException(
                status_code=400, 
                detail="Must provide exactly one of: file upload, file_url, or chunks (not multiple)"
            )
        # Generate unique document ID
        task_id = str(uuid.uuid4())
        
        # If file is provided, read its content immediately to avoid I/O issues in background task
        file_content = None
        if file:
            try:
                file_content = file.file.read()
                # Reset file pointer for potential future reads
                file.file.seek(0)
            except Exception as e:
                logger.error(f"Error reading file content: {e}")
                raise HTTPException(status_code=400, detail="Failed to read uploaded file")
        
        # Add background task for processing
        add_document_processing_task(
            background_tasks=background_tasks,
            task_id=task_id,
            file_url=file_url,
            file=file,
            file_content=file_content,
            chunks=chunks
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "Upload accepted for processing",
                "schema_version": schema_version,
                "processing_mode": "background",
                "upload_type": "file" if file else "url",
                "task_id": task_id
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/similarity_search")
async def similarity_search(
    request: SimilaritySearchRequest
):
    """Endpoint for similarity search."""
    response = await similarity_search_manager.search_similar_chunks(
        query=request.query,
        k=request.k,
        min_score=request.min_score
    )
    return response

@router.get("/{journal_id}", response_model=JournalResponse)
async def get_journal(journal_id: str):
    """
    Get all chunks for a specific journal.
    
    Args:
        journal_id: Journal identifier
        
    Returns:
        JournalResponse containing metadata and all chunks for the journal
    """
    try:
        logger.info(f"Received request for journal: {journal_id}")
        
        response = await journal_operations.get_journal_response(journal_id)
        
        logger.info(f"Successfully retrieved journal {journal_id} with {len(response.chunks)} chunks")
        return response
        
    except HTTPException:
        raise HTTPException(status_code=404, detail="Journal not found")
    except Exception as e:
        logger.error(f"Error retrieving journal {journal_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/journals/list", response_model=JournalListResponse)
async def list_journals():
    """
    Get list of all available journals.
    
    Returns:
        JournalListResponse containing list of journals and total count
    """
    try:
        logger.info("Received request to list all journals")
        
        # Get list of available journals
        response = await journal_operations.get_available_journals()
        
        logger.info(f"Successfully retrieved {response.total_count} journals")
        return response
        
    except Exception as e:
        logger.error(f"Error listing journals: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    