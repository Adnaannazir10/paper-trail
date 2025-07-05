"""
Background Tasks Utility
Simple background task processing using FastAPI BackgroundTasks.
"""

from fastapi import BackgroundTasks, UploadFile
from typing import Dict, Any, List, Optional
import logging

from .handle_file_url import process_pdf_from_url
from .handle_upload_file import process_uploaded_pdf_from_content
from .extract_metadata import extract_metadata_from_first_page
from .create_chunks import process_text_to_chunks
from .embeddings import embeddings_manager
from .qdrant_client import qdrant_manager

logger = logging.getLogger(__name__)


async def process_document_background(
    file_url: Optional[str] = None,
    file: Optional[UploadFile] = None,
    file_content: Optional[bytes] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    task_id: Optional[str] = None
):
    """
    Background task to process a document: download, chunk, embed, and store in Qdrant.
    
    Args:
        file_url: URL to fetch the document (optional)
        file: Uploaded file (optional)
        chunks: Pre-chunked content (optional)
        document_id: Unique identifier for the document
    """
    try:
        logger.info(f"Starting background document processing for: {task_id}")
        
        # Validate that exactly one input method is provided
        input_count = sum([file_url is not None, file is not None, chunks is not None])
        if input_count != 1:
            raise ValueError("Must provide exactly one of: file_url, file, or chunks")
        
        # Scenario 1: Process file from URL
        if file_url:
            logger.info(f"Processing file from URL: {file_url}")
            
            # Download and extract text from PDF
            pdf_result = await process_pdf_from_url(file_url)
            
            # Extract metadata from first page
            metadata = await extract_metadata_from_first_page(pdf_result["first_page_text"])
            
            # Create chunks with metadata
            chunks = await process_text_to_chunks(
                text=pdf_result["full_text"],
                filename=pdf_result['filename'],
                journal=metadata.journal, # type: ignore
                publish_year=metadata.publish_year, # type: ignore
                link=file_url
            )
            
            logger.info(f"Created {len(chunks)} chunks from URL file")
        
        # Scenario 2: Process uploaded file
        elif file and file_content:
            logger.info(f"Processing uploaded file: {file.filename}")
            
            # Extract text from uploaded PDF using the content
            if file.filename is None:
                raise ValueError("File filename is required")
            pdf_result = await process_uploaded_pdf_from_content(file_content, file.filename)
            
            # Extract metadata from first page
            metadata = await extract_metadata_from_first_page(pdf_result["first_page_text"])
                        
            # Create chunks with metadata
            chunks = await process_text_to_chunks(
                text=pdf_result["full_text"],
                filename=pdf_result['filename'],
                journal=metadata.journal, # type: ignore
                publish_year=metadata.publish_year, # type: ignore
                link=None
            )
            
            logger.info(f"Created {len(chunks)} chunks from uploaded file")
        
        # Scenario 3: Process pre-chunked content
        elif chunks:
            logger.info(f"Processing {len(chunks)} pre-chunked items")
            # Chunks are already provided, no need to create them
        
        # Add embeddings to chunks
        logger.info("Adding embeddings to chunks...")
        if chunks is None:
            raise ValueError("No chunks available for processing")
        chunks_with_embeddings = await embeddings_manager.add_embeddings_to_chunks(chunks)
        
        # Store in Qdrant
        logger.info("Storing chunks in Qdrant...")
        success = await qdrant_manager.save_chunks_with_embeddings(chunks_with_embeddings)
        
        if success:
            logger.info(f"Successfully stored {len(chunks_with_embeddings)} chunks in Qdrant for task: {task_id}")
        else:
            logger.error(f"Failed to store chunks in Qdrant for task: {task_id}")
        
        logger.info(f"Background document processing completed: {task_id}")
        return success
    except Exception as e:
        logger.error(f"Error in background document processing {task_id}: {e}")
        raise


async def health_check_background():
    """
    Background health check task.
    """
    try:
        logger.info("Running background health check")
        # TODO: Add actual health checks for Qdrant, etc.
        logger.info("Background health check completed")
    except Exception as e:
        logger.error(f"Background health check failed: {e}")


async def cleanup_background():
    """
    Background cleanup task.
    """
    try:
        logger.info("Running background cleanup")
        # TODO: Add cleanup logic
        logger.info("Background cleanup completed")
    except Exception as e:
        logger.error(f"Background cleanup failed: {e}")


def add_document_processing_task(
    background_tasks: BackgroundTasks, 
    task_id: str,
    file_url: Optional[str] = None,
    file: Optional[UploadFile] = None,
    file_content: Optional[bytes] = None,
    chunks: Optional[List[Dict[str, Any]]] = None
):
    """
    Add document processing task to background tasks.
    
    Args:
        background_tasks: FastAPI BackgroundTasks instance
        task_id: Unique identifier for the task
        file_url: URL to fetch the document (optional)
        file: Uploaded file (optional)
        chunks: Pre-chunked content (optional)
    """
    background_tasks.add_task(
        process_document_background, 
        file_url=file_url,
        file=file,
        file_content=file_content,
        chunks=chunks,
        task_id=task_id
    )
    logger.info(f"Added document processing task for: {task_id}")


def add_health_check_task(background_tasks: BackgroundTasks):
    """
    Add health check task to background tasks.
    
    Args:
        background_tasks: FastAPI BackgroundTasks instance
    """
    background_tasks.add_task(health_check_background)
    logger.info("Added health check task")


def add_cleanup_task(background_tasks: BackgroundTasks):
    """
    Add cleanup task to background tasks.
    
    Args:
        background_tasks: FastAPI BackgroundTasks instance
    """
    background_tasks.add_task(cleanup_background)
    logger.info("Added cleanup task") 