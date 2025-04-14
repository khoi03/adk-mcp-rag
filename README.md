```bash
QDRANT_LOCAL_PATH="/home/koi/Documents/Code/vector-db-test/qdrant/db" \
COLLECTION_NAME="demo_collection" \
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
uvx mcp-server-qdrant --transport sse
```