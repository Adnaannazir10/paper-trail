"""
Simple file validation utilities for upload endpoint.
"""

import re
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)


def validate_upload_file(file: UploadFile) -> bool:
    """
    Simple file validation - check if it's a PDF file.
    
    Args:
        file: UploadFile object
        
    Returns:
        bool: True if file is valid PDF
        
    Raises:
        HTTPException: If file is not a valid PDF
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Check if file has .pdf extension
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )
        
        logger.info(f"File validation passed for: {file.filename}")
        return True
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error validating file: {e}")
        raise HTTPException(status_code=400, detail="Unable to validate file")


def validate_file_url(url: str) -> bool:
    """
    Simple URL validation - assume it's a PDF file.
    
    Args:
        url: File URL to validate
        
    Returns:
        bool: True if URL is valid
        
    Raises:
        HTTPException: If URL is invalid
    """
    try:
        
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(url):
            raise HTTPException(status_code=400, detail="Invalid file URL format")
        
        logger.info(f"URL validation passed for: {url}")
        return True
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error validating file URL: {e}")
        raise HTTPException(status_code=400, detail="Unable to validate file URL") 