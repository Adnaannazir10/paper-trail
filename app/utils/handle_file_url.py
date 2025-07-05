"""
Handle PDF file download from URL and text extraction.
"""

import io
import aiohttp # type: ignore
import PyPDF2
from urllib.parse import urlparse
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


async def download_pdf_from_url(url: str) -> tuple[bytes, str]:
    """
    Download PDF file from URL.
    
    Args:
        url: URL to download PDF from
        
    Returns:
        tuple[bytes, str]: (file_content, filename)
        
    Raises:
        HTTPException: If download fails
    """
    try:
        logger.info(f"Downloading PDF from URL: {url}")
        
        # Download the file
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                
                # Get filename from URL or Content-Disposition header
                filename = get_filename_from_response(response, url)
                
                # Read the content
                file_content = await response.read()
        
        logger.info(f"Successfully downloaded {len(file_content)} bytes from {url}")
        return file_content, filename
        
    except aiohttp.ClientError as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download PDF from URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error downloading PDF from {url}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading PDF: {str(e)}"
        )


def get_filename_from_response(response: aiohttp.ClientResponse, url: str) -> str:
    """
    Extract filename from response headers or URL.
    
    Args:
        response: HTTP response object
        url: Original URL
        
    Returns:
        str: Filename
    """
    # Try to get filename from Content-Disposition header
    content_disposition = response.headers.get('Content-Disposition', '')
    if 'filename=' in content_disposition:
        filename = content_disposition.split('filename=')[1].strip('"\'')
        if filename:
            return filename
    
    # Try to get filename from URL path
    parsed_url = urlparse(url)
    path = parsed_url.path
    if path and '.' in path:
        filename = path.split('/')[-1]
        if filename and '.' in filename:
            return filename
    
    # Default filename
    return "downloaded_document.pdf"


def extract_text_from_pdf_content(file_content: bytes, filename: str) -> tuple[str, str]:
    """
    Extract text content from PDF file content.
    
    Args:
        file_content: PDF file content as bytes
        filename: Name of the file
        
    Returns:
        tuple[str, str]: (full_text, first_page_text)
        
    Raises:
        HTTPException: If PDF processing fails
    """
    try:
        logger.info(f"Starting text extraction from PDF: {filename}")
        
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
        
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF: {filename}")
        return full_text, first_page_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF file: {str(e)}"
        )


async def process_pdf_from_url(url: str) -> dict:
    """
    Download PDF from URL and extract text and filename.
    
    Args:
        url: URL to download PDF from
        
    Returns:
        dict: Contains full text, first page text, and filename
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        logger.info(f"Processing PDF from URL: {url}")
        
        # Download the PDF
        file_content, filename = await download_pdf_from_url(url)
        
        # Extract text content
        full_text, first_page_text = extract_text_from_pdf_content(file_content, filename)
        
        result = {
            "full_text": full_text,
            "first_page_text": first_page_text,
            "filename": filename
        }
        
        logger.info(f"Successfully processed PDF from URL: {url}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF from URL {url}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF from URL: {str(e)}"
        ) 