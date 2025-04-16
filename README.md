# RAG Agent with Google ADK and Qdrant MCP server

A Retrieval-Augmented Generation (RAG) system that leverages Google's Agent Development Kit (ADK) and Qdrant vector database via MCP server.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that combines the power of Google's Agent Development Kit (ADK) with Qdrant vector database (via MCP server) for efficient knowledge retrieval. The system enhances Large Language Model (LLM) responses by retrieving relevant context from a vector database before generating answers.

## Architecture
![architecture](./assets/RAG_Agent_Architecture.png)

## Features

- **Advanced Retrieval**: Semantic search powered by Qdrant vector database
- **Google ADK Integration**: Leverages Google's Agent Development Kit for LLM capabilities
- **MCP Server**: Model Context Protocol server for Qdrant vector database
- **Context Augmentation**: Enhances LLM responses with relevant retrieved information
- **Monitoring & Logging**: Comprehensive tracking of system performance

## Installation

```bash
# Clone the repository
git clone https://github.com/khoi03/RAG-MCP-Google-ADK.git
cd rag-agent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```
# Google ADK Configuration
GOOGLE_ADK_API_KEY=your_api_key
GOOGLE_ADK_PROJECT_ID=your_project_id

# MCP Server Configuration
MCP_SERVER_HOST=your_mcp_host
MCP_SERVER_PORT=your_mcp_port
MCP_SERVER_API_KEY=your_mcp_api_key

# Qdrant Configuration
QDRANT_COLLECTION_NAME=your_collection_name
QDRANT_VECTOR_SIZE=768  # Adjust based on your embedding model

# RAG Configuration
RETRIEVAL_TOP_K=5
CONTEXT_WINDOW_SIZE=4096
```

### Qdrant Collection Setup

```python
from config import QdrantConfig
from qdrant_client import QdrantClient

def setup_qdrant_collection():
    client = QdrantClient(
        url=QdrantConfig.MCP_SERVER_HOST,
        port=QdrantConfig.MCP_SERVER_PORT,
        api_key=QdrantConfig.MCP_SERVER_API_KEY
    )
    
    # Create collection if it doesn't exist
    client.recreate_collection(
        collection_name=QdrantConfig.COLLECTION_NAME,
        vectors_config={
            "embedding": {
                "size": QdrantConfig.VECTOR_SIZE,
                "distance": "Cosine"
            }
        }
    )
    
    print(f"Collection {QdrantConfig.COLLECTION_NAME} created successfully")

if __name__ == "__main__":
    setup_qdrant_collection()
```

## Usage

### Basic Usage

```python
from rag_agent import RAGAgent

# Initialize the agent
agent = RAGAgent()

# Query the agent
response = agent.query("What is the capital of France?")
print(response)
```

### Document Ingestion

```python
from rag_agent import DocumentProcessor

processor = DocumentProcessor()

# Process a single document
processor.ingest_document("path/to/document.pdf")

# Process a directory of documents
processor.ingest_directory("path/to/documents/")
```

### Custom Retrieval Parameters

```python
from rag_agent import RAGAgent

agent = RAGAgent()

# Customize retrieval parameters for a specific query
response = agent.query(
    "Explain quantum computing",
    top_k=10,              # Retrieve more documents
    reranking_enabled=True,  # Enable reranking of results
    filter={"domain": "physics"}  # Filter by metadata
)
print(response)
```

## API Reference

### RAGAgent Class

```python
class RAGAgent:
    """Main RAG agent class that coordinates retrieval and generation."""
    
    def __init__(self, config=None):
        """Initialize the RAG agent with optional custom configuration."""
        
    def query(self, query_text, **kwargs):
        """Process a query and return the augmented response."""
        
    def add_document(self, document, metadata=None):
        """Add a document to the knowledge base."""
        
    def refresh_embeddings(self):
        """Refresh document embeddings."""
```

### MCP Server Client

```python
class MCPQdrantClient:
    """Client for interacting with Qdrant via MCP Server."""
    
    def __init__(self, host, port, api_key):
        """Initialize the MCP Qdrant client."""
        
    def search(self, query_vector, top_k=5, filter=None):
        """Search for similar vectors in Qdrant."""
        
    def add_vectors(self, vectors, payloads=None, ids=None):
        """Add vectors to the Qdrant collection."""
```

## Performance Optimization

### Embedding Caching

The system implements an efficient caching mechanism for embeddings to avoid redundant computations:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text):
    # Generate embedding
    return embedding_model.encode(text)
```

### Batch Processing

For efficient document ingestion:

```python
def batch_process_documents(documents, batch_size=10):
    """Process documents in batches for efficiency."""
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        # Process batch
        # ...
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.