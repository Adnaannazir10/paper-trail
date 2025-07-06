"""
Integration tests for the complete ingestion pipeline.
Tests the full flow from file upload to vector storage in Qdrant.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any
import io

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.background_tasks import process_document_background
from utils.create_chunks import process_text_to_chunks
from utils.embeddings import EmbeddingsManager
from utils.qdrant_client import QdrantManager
from utils.handle_upload_file import process_uploaded_pdf_from_content


class TestIngestionPipeline:
    """Test the complete ingestion pipeline from file to vector storage."""
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Create a sample PDF content for testing."""
        # This is a simple PDF content that can be processed
        # In a real test, you might want to use an actual PDF file
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(This is a test PDF document for ingestion testing.) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"
    
    @pytest.fixture
    def sample_text_content(self):
        """Create sample text content for testing."""
        return """
        Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence that focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide.
        
        Types of Machine Learning
        
        There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning involves training a model on labeled data, while unsupervised learning finds patterns in unlabeled data. Reinforcement learning uses a system of rewards and penalties to guide the learning process.
        
        Applications of Machine Learning
        
        Machine learning has numerous applications across various industries. In healthcare, it's used for disease diagnosis and drug discovery. In finance, it's used for fraud detection and algorithmic trading. In transportation, it's used for autonomous vehicles and route optimization.
        
        Conclusion
        
        Machine learning continues to evolve and find new applications. As we collect more data and develop better algorithms, the potential for machine learning to solve complex problems grows exponentially.
        """
    
    @pytest.fixture
    def embeddings_manager(self):
        """Create a real embeddings manager for testing."""
        return EmbeddingsManager(model_name="all-MiniLM-L6-v2")
    
    @pytest.fixture
    def qdrant_manager(self):
        """Create a Qdrant manager for testing."""
        return QdrantManager()
    
    @pytest.mark.asyncio
    async def test_text_chunking_pipeline(self, sample_text_content):
        """Test the text chunking pipeline with real text."""
        # Arrange
        filename = "test_document.txt"
        journal = "Test Journal"
        publish_year = 2024
        
        # Act
        chunks = await process_text_to_chunks(
            text=sample_text_content,
            filename=filename,
            journal=journal,
            publish_year=publish_year,
            link="https://example.com/test"
        )
        
        # Assert
        assert len(chunks) > 0, "Should create at least one chunk"
        
        # Check chunk structure
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "source_doc_id" in chunk
            assert "chunk_index" in chunk
            assert "journal" in chunk
            assert "publish_year" in chunk
            assert "section_heading" in chunk
            assert "attributes" in chunk
            assert "usage_count" in chunk
            assert "link" in chunk
            
            # Check specific values
            assert chunk["source_doc_id"] == filename
            assert chunk["journal"] == journal
            assert chunk["publish_year"] == publish_year
            assert chunk["usage_count"] == 0
            assert chunk["link"] == "https://example.com/test"
            assert len(chunk["text"]) > 0
        
        # Check chunk ordering
        chunk_indices = [chunk["chunk_index"] for chunk in chunks]
        assert chunk_indices == list(range(1, len(chunks) + 1))
        
        print(f"Created {len(chunks)} chunks from text")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"Chunk {i+1}: {chunk['id']} - {chunk['section_heading']}")
    
    @pytest.mark.asyncio
    async def test_embeddings_creation(self, embeddings_manager, sample_text_content):
        """Test real embedding creation with the embeddings manager."""
        # Arrange
        chunks = await process_text_to_chunks(
            text=sample_text_content,
            filename="test_embeddings.txt",
            journal="Test Journal",
            publish_year=2024
        )
        
        # Act
        chunks_with_embeddings = await embeddings_manager.add_embeddings_to_chunks(chunks)
        
        # Assert
        assert len(chunks_with_embeddings) == len(chunks)
        
        for chunk in chunks_with_embeddings:
            assert "embedding" in chunk
            assert isinstance(chunk["embedding"], list)
            assert len(chunk["embedding"]) > 0
            
            # Check embedding dimension (all-MiniLM-L6-v2 should be 384)
            embedding_dim = len(chunk["embedding"])
            print(f"Embedding dimension: {embedding_dim}")
            assert embedding_dim == 384, f"Expected 384 dimensions, got {embedding_dim}"
        
        print(f"Successfully created embeddings for {len(chunks_with_embeddings)} chunks")
    
    @pytest.mark.asyncio
    async def test_single_embedding_creation(self, embeddings_manager):
        """Test creating a single embedding."""
        # Arrange
        test_text = "This is a test sentence for embedding creation."
        
        # Act
        embedding = await embeddings_manager.create_single_embedding(test_text)
        
        # Assert
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
        assert all(isinstance(val, (int, float)) for val in embedding)
        
        print(f"Created single embedding with dimension: {len(embedding)}")
    
    @pytest.mark.asyncio
    async def test_embedding_dimension_consistency(self, embeddings_manager):
        """Test that embeddings have consistent dimensions."""
        # Arrange
        texts = [
            "Short text.",
            "This is a medium length text for testing embedding consistency.",
            "This is a much longer text that should still produce the same dimension embedding as the shorter texts, demonstrating that the embedding model maintains consistent output dimensions regardless of input text length."
        ]
        
        # Act
        embeddings = await embeddings_manager.create_embeddings(texts)
        
        # Assert
        assert len(embeddings) == len(texts)
        
        dimensions = [len(emb) for emb in embeddings]
        print(f"Embedding dimensions: {dimensions}")
        
        # All embeddings should have the same dimension
        assert len(set(dimensions)) == 1, f"All embeddings should have same dimension, got: {dimensions}"
        assert dimensions[0] == 384, f"Expected 384 dimensions, got {dimensions[0]}"
        
        print(f"All {len(embeddings)} embeddings have consistent dimension: {dimensions[0]}")
    
    @pytest.mark.asyncio
    async def test_complete_ingestion_pipeline(self, sample_text_content, qdrant_manager):
        """Test the complete ingestion pipeline from text to Qdrant storage."""
        # This test requires a real Qdrant instance
        # Skip if Qdrant is not available
        try:
            # Test Qdrant connection
            client = qdrant_manager.get_client()
            collections = client.get_collections()
            print(f"Available collections: {[col.name for col in collections.collections]}")
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")
        
        # Arrange
        chunks = await process_text_to_chunks(
            text=sample_text_content,
            filename="test_ingestion.txt",
            journal="Test Journal",
            publish_year=2024,
            link="https://example.com/test"
        )
        
        # Create embeddings
        embeddings_manager = EmbeddingsManager()
        chunks_with_embeddings = await embeddings_manager.add_embeddings_to_chunks(chunks)
        
        # Act
        success = await qdrant_manager.save_chunks_with_embeddings(chunks_with_embeddings)
        
        # Assert
        assert success, "Should successfully save chunks to Qdrant"
        
        print(f"Successfully ingested {len(chunks_with_embeddings)} chunks to Qdrant")
        
        # Verify chunks are in Qdrant by searching
        search_results = client.search(
            collection_name=qdrant_manager.collection_name,
            query_vector=chunks_with_embeddings[0]["embedding"],
            limit=5,
            score_threshold=0.1
        )
        
        assert len(search_results) > 0, "Should find at least one result in Qdrant"
        print(f"Found {len(search_results)} results in Qdrant search")
    
    @pytest.mark.asyncio
    async def test_background_processing_pipeline(self, sample_text_content):
        """Test the background processing pipeline."""
        # Arrange
        chunks = await process_text_to_chunks(
            text=sample_text_content,
            filename="test_background.txt",
            journal="Test Journal",
            publish_year=2024
        )
        
        task_id = "test_task_123"
        
        # Act
        success = await process_document_background(
            chunks=chunks,
            task_id=task_id,
            journal="Test Journal",
            publish_year=2024
        )
        
        # Assert
        assert success, "Background processing should succeed"
        print(f"Background processing completed successfully for task: {task_id}")
    
    @pytest.mark.asyncio
    async def test_chunk_metadata_extraction(self, sample_text_content):
        """Test that chunk metadata is properly extracted."""
        # Arrange
        chunks = await process_text_to_chunks(
            text=sample_text_content,
            filename="test_metadata.txt",
            journal="Test Journal",
            publish_year=2024
        )
        
        # Act & Assert
        for chunk in chunks:
            # Check that section headings are extracted
            assert chunk["section_heading"] is not None
            assert isinstance(chunk["section_heading"], str)
            
            # Check that attributes are extracted
            assert isinstance(chunk["attributes"], list)
            
            # Check that chunk has meaningful content
            assert len(chunk["text"].strip()) > 0
            
            print(f"Chunk {chunk['chunk_index']}: '{chunk['section_heading']}' - {len(chunk['text'])} chars")
    
    @pytest.mark.asyncio
    async def test_embedding_similarity(self, embeddings_manager):
        """Test that similar texts produce similar embeddings."""
        # Arrange
        similar_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "ML is a part of AI that focuses on learning from data.",
            "Artificial intelligence includes machine learning techniques."
        ]
        
        different_texts = [
            "The weather is sunny today.",
            "Cooking requires following recipes carefully.",
            "Mathematics is the language of science."
        ]
        
        # Act
        similar_embeddings = await embeddings_manager.create_embeddings(similar_texts)
        different_embeddings = await embeddings_manager.create_embeddings(different_texts)
        
        # Calculate cosine similarities
        def cosine_similarity(vec1, vec2):
            import numpy as np
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2)
        
        # Similar texts should have higher similarity
        similar_similarity = cosine_similarity(similar_embeddings[0], similar_embeddings[1])
        different_similarity = cosine_similarity(similar_embeddings[0], different_embeddings[0])
        
        print(f"Similarity between similar texts: {similar_similarity:.4f}")
        print(f"Similarity between different texts: {different_similarity:.4f}")
        
        # Assert
        assert similar_similarity > different_similarity, "Similar texts should have higher similarity"
        assert similar_similarity > 0.5, "Similar texts should have reasonable similarity"
        assert different_similarity < 0.5, "Different texts should have lower similarity"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 