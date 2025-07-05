"""
Handle uploaded PDF file text extraction.
"""

import io
import PyPDF2
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)


async def extract_text_from_pdf(file: UploadFile) -> tuple[str, str]:
    """
    Extract text content from uploaded PDF file.
    
    Args:
        file: UploadFile object containing PDF
        
    Returns:
        tuple[str, str]: (full_text, first_page_text)
        
    Raises:
        HTTPException: If PDF processing fails
    """
    try:
        logger.info(f"Starting text extraction from PDF: {file.filename}")
        
        # Read the uploaded file content
        file_content = file.file.read()
        file.file.seek(0)  # Reset file pointer for future use
        
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        
        # Extract text from all pages
        full_text = ""
        first_page_text = ""
        total_pages = len(pdf_reader.pages)
        
        logger.info(f"Processing PDF with {total_pages} pages")
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                    # Store first page text separately
                    if page_num == 0:
                        first_page_text = page_text
                logger.debug(f"Extracted text from page {page_num + 1}")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                # Continue with other pages even if one fails
        
        if not full_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted from the PDF"
            )
        
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF: {file.filename}")
        return full_text, first_page_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF file: {str(e)}"
        )


async def process_uploaded_pdf(file: UploadFile) -> dict:
    """
    Process uploaded PDF file and extract text and filename.
    
    Args:
        file: UploadFile object containing PDF
        
    Returns:
        dict: Contains full text, first page text, and filename
        
    Raises:
        HTTPException: If PDF processing fails
    """
    try:
        logger.info(f"Processing uploaded PDF: {file.filename}")
        
        # Extract text content
        full_text, first_page_text = await extract_text_from_pdf(file)
        
        result = {
            "full_text": full_text,
            "first_page_text": first_page_text,
            "filename": file.filename
        }
        
        logger.info(f"Successfully processed PDF: {file.filename}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF file: {str(e)}"
        )


async def extract_text_from_pdf_content(file_content: bytes) -> tuple[str, str]:
    """
    Extract text content from PDF bytes.
    
    Args:
        file_content: PDF file content as bytes
        
    Returns:
        tuple[str, str]: (full_text, first_page_text)
        
    Raises:
        HTTPException: If PDF processing fails
    """
    try:
        logger.info("Starting text extraction from PDF content")
        
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        
        # Extract text from all pages
        full_text = ""
        first_page_text = ""
        total_pages = len(pdf_reader.pages)
        
        logger.info(f"Processing PDF with {total_pages} pages")
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                    # Store first page text separately
                    if page_num == 0:
                        first_page_text = page_text
                logger.debug(f"Extracted text from page {page_num + 1}")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                # Continue with other pages even if one fails
        
        if not full_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted from the PDF"
            )
        
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF content")
        return full_text, first_page_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF content: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF content: {str(e)}"
        )


async def process_uploaded_pdf_from_content(file_content: bytes, filename: str) -> dict:
    """
    Process PDF content from bytes.
    
    Args:
        file_content: PDF file content as bytes
        filename: Name of the file
        
    Returns:
        dict: Contains full text, first page text, and filename
        
    Raises:
        HTTPException: If PDF processing fails
    """
    try:
        logger.info(f"Processing PDF content for file: {filename}")
        
        # Extract text content
        full_text, first_page_text = await extract_text_from_pdf_content(file_content)
        
        result = {
            "full_text": full_text,
            "first_page_text": first_page_text,
            "filename": filename
        }
        
        logger.info(f"Successfully processed PDF content for: {filename}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF content for {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF content: {str(e)}"
        ) 