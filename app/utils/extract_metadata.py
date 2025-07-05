"""
Extract metadata from first page text using LLM and LangChain JSON parser.
"""

import logging
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI # type: ignore
from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class JournalMetadata(BaseModel):
    publish_year: int | None = Field(None, description="The year the document was published.")
    journal: str | None = Field(None, description="The name of the journal.")


async def extract_metadata_from_first_page(first_page_text: str) -> JournalMetadata:
    """
    Extract publish year and journal name from first page text using LLM and LangChain parser.
    
    Args:
        first_page_text: Text from the first page of the document
    Returns:
        dict: Contains publish_year and journal_name
    """
    try:
        logger.info("Extracting metadata from first page text using LangChain parser")
        
        parser = PydanticOutputParser(pydantic_object=JournalMetadata)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that extracts metadata from academic papers."),
            ("user", (
                "Extract the publish year and journal name from the following document text. "
                "Return only a JSON object with these two fields: publish_year (integer or null) and journal name as journal (string or null). "
                "If you cannot find the information, use null for that field.\n\nDocument text:\n{first_page_text}\n\nReturn only the JSON object, nothing else."
            ))
        ])
        
        chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.1) | parser
        result = await chain.ainvoke({"first_page_text": first_page_text[:2000]})
        logger.info(f"Extracted metadata: {result}")
        return result
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return JournalMetadata(publish_year=None, journal=None)
