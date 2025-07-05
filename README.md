# Paper Trail: Research Assistant Backend

This service ingests journal PDFs, chunks and embeds them, and stores them in Qdrant for semantic search. It exposes a FastAPI backend with async ingestion (Celery + Redis) and similarity search endpoints.

## Tech Stack
- FastAPI
- Celery + Redis (background tasks)
- Qdrant (vector DB)
- OpenAI text-embedding-3-small (for embeddings)

## Setup
1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Qdrant Vector Database:
   ```bash
   # Using Docker (recommended)
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or install locally from https://qdrant.tech/documentation/guides/installation/
   ```
4. Configure environment variables:
   ```bash
   # Copy the sample environment file
   cp env.sample .env
   
   # Edit .env with your specific values
   nano .env
   ```

## Core Endpoints
- `PUT /api/upload` — Upload and enqueue PDF for ingestion
- `POST /api/similarity_search` — Semantic search over ingested chunks
- `GET /api/{journal_id}` — Retrieve all chunks + metadata for a document

## Testing Qdrant Integration
Run the test script to verify Qdrant connection:
```bash
python app/utils/test_qdrant.py
```

---
