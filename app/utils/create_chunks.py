"""
Create chunks from text with LLM-based metadata extraction.
"""

from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class ChunkMetadata(BaseModel):
    section_heading: str|None = Field(description="The section heading for this chunk")
    attributes: List[str] = Field(description="List of attributes/topics that represent this chunk")


def create_chunks(text: str, chunk_size: int = 256, chunk_overlap: int = 64) -> List[str]:
    """
    Create chunks from text using word-based splitting.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in words
        chunk_overlap: Overlap between chunks in words
        
    Returns:
        List[str]: List of text chunks
    """
    try:
        logger.info(f"Creating chunks with size {chunk_size} words and overlap {chunk_overlap} words")
        
        # Create text splitter with word-based length function
        def word_count(text: str) -> int:
            return len(text.split())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=word_count, 
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split the text
        chunks = text_splitter.split_text(text)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error creating chunks: {e}")
        raise


async def extract_chunk_metadata(chunk_text: str, previous_chunk: Optional[str] = None) -> dict:
    """
    Extract section heading and attributes from a chunk using LLM.
    
    Args:
        chunk_text: Current chunk text
        previous_chunk: Previous chunk text (for context)
        
    Returns:
        dict: Contains section_heading and attributes
    """
    try:
        logger.info("Extracting metadata for chunk")
        
        # Create parser
        parser = PydanticOutputParser(pydantic_object=ChunkMetadata)
        
        # Create prompt
        context = ""
        if previous_chunk:
            context = f"Previous chunk context: {previous_chunk}\n\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that extracts metadata from text chunks."),
            ("user", (
                f"{context}Extract the section heading and attributes from this text chunk. "
                "Rules:\n"
                "1. If the chunk contains a clear section heading, use that\n"
                "2. Try to extract the section heading from the chunk text\n"
                "3. Their will always be a section heading, like it may be 'Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion', or a question or a statement\n"
                "4. Attributes should be a list of 2-5 key topics/themes from the chunk\n"
                "5. Return as JSON with 'section_heading' (string or null) and 'attributes' (list of strings)\n\n"
                f"Current chunk:\n{chunk_text}\n\n"
                "Return only the JSON object."
            ))
        ])
        
        # Create chain
        chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.1) | parser
        
        # Get result
        result = await chain.ainvoke({"chunk_text": chunk_text, "previous_chunk": previous_chunk})
        
        logger.info(f"Extracted metadata: {result}")
        return result.dict()
        
    except Exception as e:
        logger.error(f"Error extracting chunk metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_text_to_chunks(text: str, filename: str, journal: str, publish_year: int, link: str|None = None) -> List[dict]:
    """
    Process text into chunks with metadata.
    
    Args:
        text: Full text to process
        filename: Name of the source file
        journal: Name of the journal
        publish_year: Year of publication
        link: Link to the source document
    Returns:
        List[dict]: List of chunks with metadata
    """
    try:
        logger.info(f"Processing text from {filename} into chunks")
        
        # Create chunks
        text_chunks = create_chunks(text)
        
        # Process each chunk with metadata
        processed_chunks = []
        previous_chunk = None
        
        for i, chunk in enumerate(text_chunks):
            # Extract metadata
            metadata = await extract_chunk_metadata(chunk, previous_chunk)
            if metadata["section_heading"] is None and previous_chunk:
                metadata["section_heading"] = processed_chunks[-1]["section_heading"]
            # Create chunk object
            chunk_obj = {
                "id": f"{filename.replace('.pdf', '')}_{i+1}",
                "source_doc_id": filename,
                "chunk_index": i + 1,
                "text": chunk,
                "section_heading": metadata["section_heading"],
                "attributes": metadata["attributes"],
                "journal": journal,
                "publish_year": publish_year,
                "usage_count": 0,
                "link": link
            }
            
            processed_chunks.append(chunk_obj)
            previous_chunk = chunk
            
            logger.debug(f"Processed chunk {i+1}/{len(text_chunks)}")
        
        logger.info(f"Successfully processed {len(processed_chunks)} chunks")
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error processing text to chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))