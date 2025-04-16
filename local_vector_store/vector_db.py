from qdrant_client import QdrantClient
from pydantic import BaseModel, ConfigDict
from uuid import uuid4
from typing import List, Tuple, Optional
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, OptimizersConfigDiff

# Initialize the vector store
class VectorDB(BaseModel):
    memory_location: str = "localhost"
    embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_embeddings_model_name: str = "Qdrant/bm25"
    collection_name: str = "demo_collection"
    vector_size: int = 384
    client: Optional[QdrantClient] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Qdrant client and set up the collection"""
        # First, create the client
        self.client = QdrantClient(self.memory_location)  # For production using url and port
        
        # Set the models - this is needed for automatic collection creation to work
        self.client.set_model(self.embeddings_model_name)
        self.client.set_sparse_model(self.sparse_embeddings_model_name)
        
    def check_collection_existence(self):
        """Check if the collection exists in the vector database"""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return collection_info is not None
        except Exception as e:
            print(f"Collection does not exist: {e}")
            return False
        
    def get_documents_from_collection(self):
        """Retrieve documents from the collection of the vector database"""
        limit = 100  # Set a high number or determine count first
        offset = 0
        all_documents = []

        # Get count of points in the collection
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            total_documents = collection_info.points_count
            print(f"Total documents in collection: {total_documents}")
            
            # Fetch in batches to handle potentially large collections
            while offset < total_documents:
                # Using scroll API for retrieving large datasets
                response = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False 
                )
                
                # Extract documents from points
                batch_points = response[0]  # First element contains points
                
                for point in batch_points:
                    # Get content
                    document_text = point.payload.get("page_content", "")
                    document_metadata = {k: v for k, v in point.payload.items() if k != "page_content"}
                    
                    all_documents.append({
                        "id": point.id,
                        "page_content": document_text,
                        "metadata": document_metadata
                    })
                
                # Update offset for next batch
                offset += len(batch_points)
                
                # If we got fewer documents than the limit, we've reached the end
                if len(batch_points) < limit:
                    break

            print(f"Retrieved {len(all_documents)} documents from Qdrant")
            return all_documents
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def add_to_vectordb(self, documents, source_ids):
        """Add documents to the vector database"""
        try:
            if len(documents) == 0:
                print("No documents to add")
                return
                
            # Prepare metadata - keep page_content separate as it's expected by Qdrant's add() method
            metadata = [{"source_id": source_id} for source_id in source_ids]
            ids = [str(uuid4()) for _ in range(len(documents))]
            
            # Use add method which handles embedding and collection creation internally
            self.client.add(
                collection_name=self.collection_name,
                documents=documents,
                metadata=metadata,
                ids=ids
            )
            print(f"Successfully added {len(documents)} documents to vector database")
        except Exception as e:
            print(f"Error adding documents to vector database: {e}")

    def query(self, query_text: str, limit: int = 5, threshold: int = 0.5):
        """Query from the vector database"""
        try:
            # Try the simplified query method first
            search_result = self.client.query(
                collection_name=self.collection_name,
                query_text=query_text,
                limit=limit,
                score_threshold=threshold
            )
            return search_result
        except Exception as e:
            print(f"Error using query(): {e}")
            return []