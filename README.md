# Paper Trail: AI Research Assistant

A comprehensive research assistant that processes journal documents, creates embeddings, stores them in a vector database, and provides intelligent search and analysis capabilities through a modern web interface.

## 🏗️ Architecture

### Backend (Python FastAPI)
- **Framework**: FastAPI
- **Vector Database**: Qdrant for semantic search
- **Embeddings**: avsolatorio/NoInstruct-small-Embedding-v0
- **LLM**: OpenAI GPT-4o-mini

### Frontend (React)
- **Framework**: React with TypeScript
- **Styling**: Modern CSS with glass morphism effects

## 📁 Project Structure

```
paper-trail/
├── app/                    # Backend FastAPI application
│   ├── api/               # API routes
│   ├── config/            # Configuration files
│   ├── schemas/           # Pydantic models
│   ├── utils/             # Utility functions
│   ├── app.py             # FastAPI app initiliased
│   └── main.py           # Uvicorn server runner script
├── frontend/              # React frontend application
│   ├── src/              # React source code
│   ├── public/           # Static assets
│   └── package.json      # Frontend dependencies
└── README.md             # This file
```


## 🚀 Quick Start

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd app
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp env.sample .env
   # Edit .env and add your OpenAI API key
   ```

5. **Start Qdrant Vector Database**:
   ```bash
   # Using Docker (recommended)
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or install locally from https://qdrant.tech/documentation/guides/installation/
   ```

6. **Run the FastAPI application**:
   ```bash
   python3 main.py
   ```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Start the React development server**:
   ```bash
   npm start
   ```

The frontend will be available at `http://localhost:3000`

## 🔌 API Endpoints

### Document Management
- `PUT /api/upload` - Upload PDF documents or URLs for processing
- `GET /api/journals/list` - Get list of all available journals
- `GET /api/{journal_id}` - Get all chunks and metadata for a specific journal

### Search & Analysis
- `POST /api/similarity_search` - Semantic search across all documents
- `POST /api/ask_llm` - Ask questions with optional journal filtering

### Request Example
#### Ask LLM Question
```bash
curl -X POST "http://localhost:8000/api/ask_llm" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings about climate change?"
  }'
```