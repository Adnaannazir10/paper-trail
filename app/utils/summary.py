"""
Summary utility for generating journal summaries using LangChain and StrParser.
"""

import logging
from typing import List
from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import math

from schemas.summary_schemas import SummaryResponse

from utils.journal_operations import journal_operations
from utils.qdrant_client import qdrant_manager

logger = logging.getLogger(__name__)


class SummaryManager:
    """Handles journal summarization using LangChain and StrParser."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500,
            length_function=self._count_words,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        # Create summarization chain
        self.summary_chain = self._create_summary_chain()
        
    def _count_words(self, text: str) -> int:
        """Count words in text (not characters)."""
        words = re.findall(r'\b\w+\b', text.lower())
        return len(words)
    
    def _create_summary_chain(self):
        """Create the summarization chain with prompt and parser."""
            
        prompt = ChatPromptTemplate.from_template("""
        You are an expert research analyst. Please provide a comprehensive summary of the following academic text.
        
        Text to summarize:
        {text}
        
        Please provide a detailed summary that includes:
        1. Main research objectives and questions
        2. Key findings and results
        3. Methodology used
        4. Important conclusions
        5. Significance and implications
        
        Make the summary clear, well-structured, and comprehensive while maintaining academic rigor.
        """)
        
        return prompt | self.llm | StrOutputParser()
    
    async def summarize_doc_by_paper_name(self, paper_name: str) -> SummaryResponse:
        """
        Summarize a document by fetching its full text from Qdrant by paper_name and applying summarization.
        If the word count is > 80,000, split into N equal parts, summarize each, then summarize the summaries.
        Args:
            paper_name: The name of the paper (title)
        Returns:
            Final summary string
        """
        try:
            logger.info(f"Starting summarization for paper: {paper_name}")
            doc = await qdrant_manager.get_full_doc_by_paper_name(paper_name)
            if not doc or not doc.get("full_text"):
                raise ValueError(f"No document found for paper name: {paper_name}")
            full_text = doc["full_text"]
            words = full_text.split()
            word_count = len(words)
            logger.info(f"Document '{paper_name}' has {word_count} words")
            if word_count > 80000:
                # Split into N equal parts
                n_parts = math.ceil(word_count / 80000)
                chunk_size = word_count // n_parts
                logger.info(f"Splitting into {n_parts} parts of ~{chunk_size} words each")
                summaries = []
                for i in range(n_parts):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size if i < n_parts - 1 else word_count
                    chunk_text = " ".join(words[start:end])
                    logger.info(f"Summarizing chunk {i+1}/{n_parts} ({end-start} words)")
                    chunk_summary = await self._summarize_text(chunk_text)
                    summaries.append(chunk_summary)
                combined_summaries = "\n\n".join(summaries)
                logger.info("Summarizing combined chunk summaries")
                summary = await self._summarize_text(combined_summaries)
            else:
                summary = await self._summarize_text(full_text)
            logger.info(f"Successfully summarized document {paper_name}")
            return SummaryResponse(
                paper_name=paper_name,
                summary=summary,
            )
        except Exception as e:
            logger.error(f"Error in summarize_doc_by_paper_name: {e}")
            raise
    
    async def _summarize_text(self, text: str) -> str:
        """Summarize text directly using the LLM chain."""
        try:
            result = await self.summary_chain.ainvoke({"text": text})
            return result
        except Exception as e:
            logger.error(f"Error in direct summarization: {e}")
            raise HTTPException(status_code=500, detail=str(e))

summary_manager = SummaryManager()