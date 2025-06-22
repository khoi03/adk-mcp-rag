from langchain.schema import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFDirectoryLoader

from vector_db import VectorDB

DATA_PATH = "data"

def calculate_chunk_ids(chunks):
    '''
    This will create IDs like "data/monopoly.pdf:6:2"
    Page Source : Page Number : Chunk Index
    '''

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_vectorstore(db, chunks):
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get_documents_from_collection()  # IDs are always included by default

    existing_ids = []
    if existing_items:
        for item in existing_items:
            existing_ids.append(item['metadata']['source_id'])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    new_chunk_ids = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk.page_content)
            new_chunk_ids.append(chunk.metadata["id"])

    print("CHUNK IDS:", new_chunk_ids)
    if len(new_chunks):
        batch_size = 50
        for i in range(0, len(new_chunks), batch_size):
            batch_docs = new_chunks[i:i+batch_size]
            batch_ids = new_chunk_ids[i:i+batch_size]
            print(f"Adding batch {i//batch_size + 1} with {len(batch_docs)} documents")
            db.add_to_vectordb(batch_docs, source_ids=batch_ids)
        print(f"Successfully added all {len(new_chunks)} documents")
        # db.persist()
    else:
        print("No new documents to add")

def split_text(documents: list[Document]):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 400,
        length_function = len,
        add_start_index = True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[1]
    # print(document.page_content)
    # print(document.metadata)

    return chunks

def load_documents():
    # Load documents from datapath
    documents_loader = DirectoryLoader(DATA_PATH, glob="*/*.md")
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH, extract_images=False)
    documents = documents_loader.load()
    pdfs = pdf_loader.load()
    final_documents = documents + pdfs
    print("PDF:\n", pdfs)
    return final_documents

def generate_data_store(db):
    documents = load_documents()
    chunks = split_text(documents)
    
    add_to_vectorstore(db, chunks)

if __name__ == "__main__":
    # Create a new vector store
    vector_store = VectorDB(
        embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2", 
        memory_location="localhost", 
        vector_size=384
    )
    generate_data_store(vector_store)

    query_rag = "Tran Dinh Khoi"
    print("Querying RAG")
    print(vector_store.query(query_text=query_rag, limit=2, threshold=0))