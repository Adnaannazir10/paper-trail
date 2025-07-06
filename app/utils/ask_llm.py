"""
LLM manager for processing queries and generating responses using similar chunks.
"""

import logging
from typing import List, Optional
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

from schemas.llm_schemas import AskLLMResponse
from utils.similarity_search import similarity_search_manager
from schemas.search_schemas import SimilaritySearchResponse

logger = logging.getLogger(__name__)


class LLMManager:
    """Handles LLM operations using similarity search for context."""
    
    def __init__(self):
        self.similarity_search = similarity_search_manager
    
    async def ask_llm(
        self, 
        query: str, 
        journal: Optional[str] = None,
    ) -> AskLLMResponse:
        """
        Ask the LLM a question using similar chunks as context.
        
        Args:
            query: The question to ask
            journal: Optional journal filter
            k: Number of similar chunks to retrieve
            min_score: Minimum similarity score threshold
            
        Returns:
            LLM response string
        """
        try:
            logger.info(f"Processing LLM query: '{query[:100]}...'")
            if journal:
                logger.info(f"Using journal filter: {journal}")
            
            # Step 1: Search for similar chunks using filters
            search_response = await self.similarity_search.search_with_filters(
                query=query,
                journal_filter=journal,
            )
            
            # Step 2: Extract context from search results
            context_chunks = self._extract_context_from_results(search_response)
            
            # Step 3: Generate LLM response using context
            llm_response = await self._generate_response(query, context_chunks)
            
            logger.info("Successfully generated LLM response")
            return AskLLMResponse(llm_response=llm_response)
            
        except Exception as e:
            logger.error(f"Error in ask_llm: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _extract_context_from_results(self, search_response: SimilaritySearchResponse) -> List[str]:
        """
        Extract context text from search results.
        
        Args:
            search_response: Similarity search response
            
        Returns:
            List of context text chunks
        """
        context_chunks = []
        
        for result in search_response.results:
            context_chunks.append(result.text)
        
        logger.info(f"Extracted {len(context_chunks)} context chunks")
        return context_chunks
    
    async def _generate_response(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate LLM response using query and context chunks.
        
        Args:
            query: The original query
            context_chunks: List of relevant context chunks
            
        Returns:
            Generated LLM response
        """
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            context_text = "\n\n".join(context_chunks) 
            system_prompt = """You are a helpful assistant that answers questions based only on the provided context. You must use only the information given in the context to answer the user's query. Do not use any external knowledge or information not present in the provided context. If the context does not contain enough information to answer the question, clearly state that you cannot answer based on the available information."""
            
            
            system_template = SystemMessagePromptTemplate.from_template(system_prompt)
            human_template = HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context provided:")
            
            chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])
            
            chain = chat_prompt | llm | StrOutputParser()
            
            response = await chain.ainvoke({
                "context": context_text,
                "question": query
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return f"Error generating response: {str(e)}"


# Create singleton instance
llm_manager = LLMManager()