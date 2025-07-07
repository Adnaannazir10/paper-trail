import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperName(BaseModel):
    """Pydantic model for paper name extraction."""
    paper_name: str = Field(description="The exact title of the paper")

class PaperNameExtractor:
    """Utility class to extract paper names from the first page of documents using LLM."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize the paper name extractor."""
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name)
        self.output_parser = JsonOutputParser(pydantic_object=PaperName)
        
        self.prompt = ChatPromptTemplate.from_template(
            "Extract the paper title from this first page content: {first_page_content}"
            "Return the paper title in JSON format with key 'paper_name'"
        )
        
        self.chain = self.prompt | self.llm | self.output_parser
    
    async def extract_paper_name(self, first_page_content: str) -> Optional[str]:
        """
        Extract the paper name from the first page content.
        
        Args:
            first_page_content: The text content of the first page
            
        Returns:
            The extracted paper name or None if extraction fails
        """
        try:
            if not first_page_content or not first_page_content.strip():
                logger.warning("Empty or None first page content provided")
                return None
            
            # Clean the content - take only the first 2000 characters to avoid token limits
            cleaned_content = first_page_content
            
            logger.info(f"Extracting paper name using {self.model_name}")
            
            # Extract the paper name
            result = await self.chain.ainvoke({"first_page_content": cleaned_content})
            # Extract paper name from JSON result
            paper_name = result['paper_name']
            
            logger.info(f"Successfully extracted paper name: {paper_name}")
            return paper_name
            
        except Exception as e:
            logger.error(f"Error extracting paper name: {str(e)}")
            return None


paper_name_extractor = PaperNameExtractor()