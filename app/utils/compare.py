"""
Compare utility for comparing multiple papers by analyzing their summaries.
"""

import logging
from typing import List
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.summary import summary_manager
from schemas.compare_schemas import ComparePapersResponse

logger = logging.getLogger(__name__)


class CompareManager:
    """Handles comparison of multiple papers using their summaries."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.compare_chain = self._create_compare_chain()
    
    def _create_compare_chain(self):
        """Create the comparison chain with prompt and parser."""
        prompt = ChatPromptTemplate.from_template("""
        You are an expert research analyst specializing in comparative analysis of academic papers.
        
        Please provide a comprehensive comparison of the following paper summaries:
        
        {summaries}
        
        Please analyze and compare these papers, focusing on:
        
        1. **Similarities**: What common themes, methodologies, or findings do these papers share?
        2. **Differences**: What are the key differences in approach, findings, or conclusions?
        3. **Research Gaps**: What gaps or areas for further research are revealed by comparing these papers?
        4. **Methodological Approaches**: How do the research methods differ or align?
        5. **Contributions**: What unique contributions does each paper make to the field?
        6. **Implications**: What are the broader implications of these findings when considered together?
        
        Provide a detailed, well-structured comparison that would be valuable for researchers in this field.
        Use clear headings and organize your analysis logically.
        """)
        
        return prompt | self.llm | StrOutputParser()
    
    async def get_paper_summaries(self, paper_names: List[str]) -> List[dict]:
        """
        Get summaries for multiple papers by paper name.
        Args:
            paper_names: List of paper names to summarize
        Returns:
            List of dictionaries containing paper name and summary
        """
        try:
            logger.info(f"Getting summaries for {len(paper_names)} papers: {paper_names}")
            summaries = []
            for paper_name in paper_names:
                try:
                    logger.info(f"Getting summary for paper: {paper_name}")
                    summary_response = await summary_manager.summarize_doc_by_paper_name(paper_name)
                    summaries.append({
                        "paper_name": paper_name,
                        "summary": summary_response.summary
                    })
                    logger.info(f"Successfully got summary for {paper_name}")
                except Exception as e:
                    logger.error(f"Error getting summary for {paper_name}: {e}")
                    summaries.append({
                        "paper_name": paper_name,
                        "summary": f"Error retrieving summary for {paper_name}: {str(e)}"
                    })
            logger.info(f"Retrieved summaries for {len(summaries)} papers")
            return summaries
        except Exception as e:
            logger.error(f"Error in get_paper_summaries: {e}")
            raise HTTPException(status_code=500, detail=f"Error getting paper summaries: {str(e)}")
    
    def _format_summaries_for_comparison(self, summaries: List[dict]) -> str:
        """
        Format summaries into a structured text for comparison.
        Args:
            summaries: List of paper summaries
        Returns:
            Formatted text for comparison
        """
        formatted_text = ""
        for i, summary_data in enumerate(summaries, 1):
            paper_name = summary_data["paper_name"]
            summary = summary_data["summary"]
            formatted_text += f"## Paper {i}: {paper_name}\n\n"
            formatted_text += f"{summary}\n\n"
            formatted_text += "---\n\n"
        return formatted_text
    
    async def compare_papers(self, paper_names: List[str]) -> str:
        """
        Compare multiple papers by analyzing their summaries.
        Args:
            paper_names: List of paper names to compare
        Returns:
            Comparison analysis as text
        """
        try:
            logger.info(f"Starting comparison of {len(paper_names)} papers")
            summaries = await self.get_paper_summaries(paper_names)
            if not summaries:
                raise HTTPException(status_code=400, detail="No summaries could be retrieved for comparison")
            formatted_summaries = self._format_summaries_for_comparison(summaries)
            logger.info("Generating comparison analysis")
            comparison_text = await self.compare_chain.ainvoke({
                "summaries": formatted_summaries
            })
            logger.info("Comparison analysis completed successfully")
            return comparison_text
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in compare_papers: {e}")
            raise HTTPException(status_code=500, detail=f"Error comparing papers: {str(e)}")
    
    async def compare_papers_by_names(self, paper_names: List[str]) -> ComparePapersResponse:
        """
        Complete workflow to compare papers and return structured response.
        Args:
            paper_names: List of paper names to compare
        Returns:
            ComparePapersResponse with comparison text
        """
        try:
            logger.info(f"Starting paper comparison for papers: {paper_names}")
            if len(paper_names) < 2:
                raise HTTPException(status_code=400, detail="At least 2 papers are required for comparison")
            if len(paper_names) > 5:
                raise HTTPException(status_code=400, detail="Maximum 5 papers allowed for comparison")
            comparison_text = await self.compare_papers(paper_names)
            response = ComparePapersResponse(
                comparison_text=comparison_text
            )
            logger.info("Paper comparison completed successfully")
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in compare_papers_by_names: {e}")
            raise HTTPException(status_code=500, detail=f"Error in paper comparison: {str(e)}")


# Create singleton instance
compare_manager = CompareManager() 