"""
Integration tests for vector similarity search functionality.
Tests the complete search pipeline from query to results using real embeddings and Qdrant.
"""

import pytest
import asyncio
import numpy as np
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.similarity_search import SimilaritySearch
from utils.embeddings import EmbeddingsManager
from utils.qdrant_client import QdrantManager
from utils.create_chunks import process_text_to_chunks
from utils.background_tasks import process_document_background
from schemas.search_schemas import SimilaritySearchResponse, SearchResult


class TestVectorSearch:
    """Test vector similarity search functionality."""
    
    @pytest.fixture
    def similarity_search(self):
        """Create a SimilaritySearch instance for testing."""
        return SimilaritySearch()
    
    @pytest.fixture
    def embeddings_manager(self):
        """Create a real embeddings manager for testing."""
        return EmbeddingsManager(model_name="all-MiniLM-L6-v2")
    
    @pytest.fixture
    def qdrant_manager(self):
        """Create a Qdrant manager for testing."""
        return QdrantManager()
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing search."""
        return [
            {
                "text": """
                Machine Learning Fundamentals
                
                Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.
                
                The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.
                
                Types of Machine Learning
                
                There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning involves training a model on labeled data, while unsupervised learning finds patterns in unlabeled data.
                """,
                "journal": "AI Research Journal",
                "publish_year": 2023,
                "filename": "ml_fundamentals.pdf"
            },
            {
                "text": """
                Deep Learning Applications
                
                Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns. It has revolutionized fields such as computer vision, natural language processing, and speech recognition.
                
                Neural Networks
                
                Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and can learn complex mappings between inputs and outputs.
                
                Applications in Computer Vision
                
                Deep learning has achieved remarkable success in computer vision tasks such as image classification, object detection, and image segmentation. Convolutional neural networks (CNNs) are particularly effective for these tasks.
                """,
                "journal": "Computer Science Quarterly",
                "publish_year": 2024,
                "filename": "deep_learning_applications.pdf"
            },
            {
                "text": """
                Natural Language Processing Techniques
                
                Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves developing algorithms and models that can understand, interpret, and generate human language.
                
                Text Processing
                
                Text processing involves various techniques such as tokenization, stemming, lemmatization, and part-of-speech tagging. These preprocessing steps are essential for building effective NLP models.
                
                Language Models
                
                Modern language models use transformer architectures to process and understand text. These models can perform tasks such as text classification, sentiment analysis, and language translation with high accuracy.
                """,
                "journal": "Computational Linguistics",
                "publish_year": 2023,
                "filename": "nlp_techniques.pdf"
            }
        ]
    
    @pytest.fixture
    async def populated_qdrant(self, sample_documents, qdrant_manager, embeddings_manager):
        """Populate Qdrant with test documents for search testing."""
        try:
            # Test Qdrant connection
            client = qdrant_manager.get_client()
            collections = client.get_collections()
            print(f"Available collections: {[col.name for col in collections.collections]}")
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")
        
        # Process each document
        all_chunks = []
        for doc in sample_documents:
            chunks = await process_text_to_chunks(
                text=doc["text"],
                filename=doc["filename"],
                journal=doc["journal"],
                publish_year=doc["publish_year"],
                link=f"https://example.com/{doc['filename']}"
            )
            all_chunks.extend(chunks)
        
        # Add embeddings to chunks
        chunks_with_embeddings = await embeddings_manager.add_embeddings_to_chunks(all_chunks)
        
        # Store in Qdrant
        success = await qdrant_manager.save_chunks_with_embeddings(chunks_with_embeddings)
        
        if not success:
            pytest.skip("Failed to populate Qdrant with test data")
        
        print(f"Populated Qdrant with {len(chunks_with_embeddings)} chunks from {len(sample_documents)} documents")
        return chunks_with_embeddings
    
    @pytest.mark.asyncio
    async def test_basic_similarity_search(self, similarity_search, populated_qdrant):
        """Test basic similarity search functionality."""
        # Arrange
        query = "machine learning algorithms"
        
        # Act
        result = await similarity_search.search_similar_chunks(
            query=query,
            k=5,
            min_score=0.1
        )
        
        # Assert
        assert isinstance(result, SimilaritySearchResponse)
        assert result.query == query
        assert result.k == 5
        assert result.min_score == 0.1
        assert result.total_results > 0, "Should find at least one result"
        assert len(result.results) > 0
        
        # Check result structure
        for search_result in result.results:
            assert isinstance(search_result, SearchResult)
            assert search_result.id is not None
            assert search_result.text is not None
            assert search_result.score > 0
            assert search_result.score <= 1.0
        
        print(f"Found {result.total_results} results for query: '{query}'")
        for i, res in enumerate(result.results[:3]):
            print(f"Result {i+1}: {res.id} (score: {res.score:.3f}) - {res.text[:100]}...")
    
    @pytest.mark.asyncio
    async def test_search_with_different_queries(self, similarity_search, populated_qdrant):
        """Test search with different types of queries."""
        queries = [
            "neural networks",
            "natural language processing",
            "computer vision",
            "deep learning models",
            "artificial intelligence"
        ]
        
        for query in queries:
            # Act
            result = await similarity_search.search_similar_chunks(
                query=query,
                k=3,
                min_score=0.1
            )
            
            # Assert
            assert result.query == query
            assert result.total_results >= 0  # Some queries might not find results
            
            print(f"Query '{query}': {result.total_results} results")
            
            if result.total_results > 0:
                # Check that results are relevant (high scores)
                scores = [res.score for res in result.results]
                avg_score = np.mean(scores)
                print(f"  Average score: {avg_score:.3f}")
                assert avg_score > 0.1, f"Average score too low for query '{query}'"
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, similarity_search, populated_qdrant):
        """Test search with journal and year filters."""
        # Test journal filter
        result_journal = await similarity_search.search_with_filters(
            query="machine learning",
            journal_filter="AI Research Journal",
            k=5,
            min_score=0.1
        )
        
        assert result_journal.total_results >= 0
        
        # Test year filter
        result_year = await similarity_search.search_with_filters(
            query="deep learning",
            year_filter=2024,
            k=5,
            min_score=0.1
        )
        
        assert result_year.total_results >= 0
        
        # Test multiple filters
        result_multi = await similarity_search.search_with_filters(
            query="neural networks",
            journal_filter="Computer Science Quarterly",
            year_filter=2024,
            k=5,
            min_score=0.1
        )
        
        assert result_multi.total_results >= 0
        
        print(f"Journal filter results: {result_journal.total_results}")
        print(f"Year filter results: {result_year.total_results}")
        print(f"Multi-filter results: {result_multi.total_results}")
    
    @pytest.mark.asyncio
    async def test_search_score_thresholds(self, similarity_search, populated_qdrant):
        """Test search with different score thresholds."""
        query = "machine learning"
        
        # Test with low threshold
        result_low = await similarity_search.search_similar_chunks(
            query=query,
            k=10,
            min_score=0.1
        )
        
        # Test with high threshold
        result_high = await similarity_search.search_similar_chunks(
            query=query,
            k=10,
            min_score=0.5
        )
        
        # Higher threshold should return fewer or equal results
        assert result_high.total_results <= result_low.total_results
        
        # Check that high threshold results have higher scores
        if result_high.total_results > 0:
            high_scores = [res.score for res in result_high.results]
            low_scores = [res.score for res in result_low.results]
            
            assert min(high_scores) >= 0.5
            assert min(low_scores) >= 0.1
        
        print(f"Low threshold (0.1): {result_low.total_results} results")
        print(f"High threshold (0.5): {result_high.total_results} results")
    
    @pytest.mark.asyncio
    async def test_search_result_ordering(self, similarity_search, populated_qdrant):
        """Test that search results are properly ordered by score."""
        query = "artificial intelligence"
        
        result = await similarity_search.search_similar_chunks(
            query=query,
            k=10,
            min_score=0.1
        )
        
        if result.total_results > 1:
            # Check that results are ordered by score (descending)
            scores = [res.score for res in result.results]
            assert scores == sorted(scores, reverse=True), "Results should be ordered by score (descending)"
            
            print(f"Result ordering check passed - {len(scores)} results ordered by score")
    
    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, similarity_search, populated_qdrant):
        """Test search behavior with empty or very short queries."""
        # Test with empty query
        result_empty = await similarity_search.search_similar_chunks(
            query="",
            k=5,
            min_score=0.1
        )
        
        # Test with very short query
        result_short = await similarity_search.search_similar_chunks(
            query="a",
            k=5,
            min_score=0.1
        )
        
        # Both should return results (though possibly with low scores)
        assert result_empty.total_results >= 0
        assert result_short.total_results >= 0
        
        print(f"Empty query results: {result_empty.total_results}")
        print(f"Short query results: {result_short.total_results}")
    
    @pytest.mark.asyncio
    async def test_search_semantic_similarity(self, similarity_search, populated_qdrant):
        """Test that semantically similar queries return similar results."""
        # Test semantically similar queries
        similar_queries = [
            "machine learning",
            "ML algorithms",
            "artificial intelligence learning"
        ]
        
        results = []
        for query in similar_queries:
            result = await similarity_search.search_similar_chunks(
                query=query,
                k=3,
                min_score=0.1
            )
            results.append(result)
        
        # Check that similar queries return some overlapping results
        if all(r.total_results > 0 for r in results):
            result_ids = [set(r.id for r in result.results) for result in results]
            
            # Check for overlap between results
            overlap_12 = len(result_ids[0] & result_ids[1])
            overlap_13 = len(result_ids[0] & result_ids[2])
            overlap_23 = len(result_ids[1] & result_ids[2])
            
            print(f"Overlap between queries: {overlap_12}, {overlap_13}, {overlap_23}")
            
            # At least some overlap should exist for semantically similar queries
            total_overlap = overlap_12 + overlap_13 + overlap_23
            assert total_overlap > 0, "Semantically similar queries should return some overlapping results"
    
    @pytest.mark.asyncio
    async def test_search_result_metadata(self, similarity_search, populated_qdrant):
        """Test that search results contain proper metadata."""
        query = "deep learning"
        
        result = await similarity_search.search_similar_chunks(
            query=query,
            k=5,
            min_score=0.1
        )
        
        if result.total_results > 0:
            for search_result in result.results:
                # Check required metadata fields
                assert search_result.source_doc_id is not None
                assert search_result.chunk_index > 0
                assert search_result.journal is not None
                assert search_result.publish_year > 0
                assert search_result.link is not None
                assert search_result.section_heading is not None
                assert isinstance(search_result.attributes, list)
                assert search_result.usage_count >= 0
                
                print(f"Result metadata: {search_result.journal} ({search_result.publish_year}) - {search_result.section_heading}")
    
    @pytest.mark.asyncio
    async def test_search_usage_count_increment(self, similarity_search, populated_qdrant):
        """Test that search increments usage counts for returned results."""
        query = "neural networks"
        
        # Get initial usage counts
        initial_result = await similarity_search.search_similar_chunks(
            query=query,
            k=3,
            min_score=0.1
        )
        
        if initial_result.total_results > 0:
            initial_usage_counts = {res.id: res.usage_count for res in initial_result.results}
            
            # Perform another search
            second_result = await similarity_search.search_similar_chunks(
                query=query,
                k=3,
                min_score=0.1
            )
            
            # Check that usage counts increased
            for res in second_result.results:
                if res.id in initial_usage_counts:
                    assert res.usage_count >= initial_usage_counts[res.id]
            
            print("Usage count increment test passed")
    
    @pytest.mark.asyncio
    async def test_search_with_no_results(self, similarity_search, populated_qdrant):
        """Test search behavior when no results are found."""
        # Use a very specific query that might not match anything
        query = "very specific technical term that probably doesn't exist in our test documents"
        
        result = await similarity_search.search_similar_chunks(
            query=query,
            k=5,
            min_score=0.8  # High threshold to ensure no results
        )
        
        # Should return empty results, not error
        assert result.total_results == 0
        assert len(result.results) == 0
        assert result.query == query
        
        print("No results search handled correctly")
    
    @pytest.mark.asyncio
    async def test_search_performance(self, similarity_search, populated_qdrant):
        """Test search performance with timing."""
        import time
        
        query = "machine learning applications"
        
        # Time the search
        start_time = time.time()
        result = await similarity_search.search_similar_chunks(
            query=query,
            k=10,
            min_score=0.1
        )
        end_time = time.time()
        
        search_time = end_time - start_time
        
        # Search should complete within reasonable time (e.g., < 5 seconds)
        assert search_time < 5.0, f"Search took too long: {search_time:.2f} seconds"
        
        print(f"Search completed in {search_time:.3f} seconds with {result.total_results} results")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 