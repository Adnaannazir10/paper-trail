from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, Form, File
from fastapi.responses import JSONResponse
from schemas.docs_schemas import DocumentListResponse
from utils.compare import compare_manager
from utils.summary import summary_manager
from schemas.upload_schemas import UploadResponse
from utils.ask_llm import llm_manager
from schemas.search_schemas import SimilaritySearchRequest, SimilaritySearchResponse
from schemas.journal_schemas import JournalResponse, JournalListResponse
from schemas.llm_schemas import AskLLMRequest, AskLLMResponse
from schemas.summary_schemas import SummaryResponse
from schemas.compare_schemas import ComparePapersRequest, ComparePapersResponse
from utils.similarity_search import similarity_search_manager
from utils.background_tasks import add_document_processing_task
from utils.journal_operations import journal_operations
from typing import Optional
import logging
import json
import uuid
from utils.qdrant_client import qdrant_manager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.put("/upload", status_code=202, response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    schema_version: str = Form(..., description="Schema version", examples=["v1"]),
    file_url: Optional[str] = Form(None, description="URL to fetch the journal document", examples=["https://www.example.com/journal.pdf"]),
    chunks: Optional[str] = Form(None, description="JSON string of chunks"),
    file: Optional[UploadFile] = File(None, description="PDF file to upload"),
    journal: Optional[str] = Form(None, description="Journal name"),
    publish_year: Optional[int] = Form(None, description="Publish year")
):
    """
    Upload endpoint for journal documents.
    
    Accepts multipart form data with:
    - schema_version: Schema version
    - chunks: JSON string containing chunks (optional)
    Structure of chunks (JSON format):<br>
    [
      {
        "id": "unique_chunk_identifier",
        "source_doc_id": "source_document_id",
        "chunk_index": 1,
        "text": "The actual text content of the chunk",
        "section_heading": "Introduction",
        "attributes": ["topic1", "topic2", "topic3"],
        "journal": "Journal Name",
        "publish_year": 2024,
        "usage_count": 0,
        "link": "https://example.com/document.pdf"
      }
    ]
    - file_url: URL to fetch the journal document
    - file: PDF file upload (optional if file_url provided)
    - journal: Journal name
    - publish_year: Publish year
    
    Returns 202 Accepted and processes document in background.
    """
    try:
        logger.info(f"Received upload request with schema_version: {schema_version}")
        # Parse upload_req if provided
        chunks_list = None
        if chunks:
            try:
                chunks_list = json.loads(chunks)
                print(chunks_list)
                logger.info(f"Parsed {len(chunks_list) if chunks_list else 0} chunks from chunks")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in chunks field")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid upload request data: {str(e)}")
        
        # Validate that exactly one option is provided: file, file_url, or chunks
        options_provided = 0
        if file is not None:
            options_provided += 1
        if file_url is not None:
            options_provided += 1
        if chunks_list is not None:
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
            chunks=chunks_list,
            journal=journal,
            publish_year=publish_year
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


@router.post("/similarity_search", response_model=SimilaritySearchResponse)
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


@router.post("/ask_llm", response_model=AskLLMResponse)
async def ask_llm(request: AskLLMRequest):
    """
    Ask the LLM a question with optional journal context.
    
    Args:
        request: AskLLMRequest containing the query and optional journal filter
        
    Returns:
        AskLLMResponse containing the LLM's response
    """
    try:
        logger.info(f"Received LLM query: {request.query[:100]}...")
        if request.journal:
            logger.info(f"Journal filter: {request.journal}")
        
        return await llm_manager.ask_llm(request.query, request.journal)
                
    except Exception as e:
        logger.error(f"Error in ask_llm endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/docs/list", response_model=DocumentListResponse)
async def list_docs():
    """
    Get list of all available documents (source_doc_id and paper_name).
    """
    try:
        logger.info("Received request to list all documents in full_doc collection")
        return await qdrant_manager.list_full_docs()
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/summary/{paper_name}", response_model=SummaryResponse)
async def get_summary(paper_name: str):
    """
    Get the summary for a document by its paper name.

    Args:
        paper_name: The name of the paper to summarize.

    Returns:
        SummaryResponse containing the summary.
    """
    try:
        logger.info(f"Received request to summarize paper: {paper_name}")
        summary = await summary_manager.summarize_doc_by_paper_name(paper_name)
        return summary
    except Exception as e:
        logger.error(f"Error summarizing paper '{paper_name}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/compare", response_model=ComparePapersResponse)
async def compare_papers(request: ComparePapersRequest):
    """
    Compare two or more papers.
    """
    try:
        logger.info(f"Received request to compare papers: {request.paper_names}")
        comparison = await compare_manager.compare_papers_by_names(request.paper_names)
        return comparison
    except Exception as e:
        logger.error(f"Error comparing papers: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")