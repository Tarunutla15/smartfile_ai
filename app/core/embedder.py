# smartfile_ai/app/core/embedder.py

import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from typing import List

# Define the model and database directory
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "chroma_db"

def get_embedding_model():
    """
    Initializes and returns the embedding model.
    """
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vector_db_instance():
    """
    Returns a ChromaDB instance, creating it if it doesn't exist.
    """
    embedding_model = get_embedding_model()
    # If the directory exists, it loads the existing DB; otherwise, it creates a new one
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )
    return vector_db

def add_documents_to_db(vector_db: Chroma, documents: List[Document]):
    """
    Adds new documents to the ChromaDB instance and persists the changes.
    """
    # The add_documents method handles chunking and embedding internally
    vector_db.add_documents(documents)
    # This persists the changes to disk
    # vector_db.persist()
    print(f"Successfully added {len(documents)} documents to the vector database.")

# --- Temporary Test Block ---
# This block runs only when the file is executed directly
if __name__ == "__main__":
    print("--- Running embedder.py temporary test block ---")
    
    # 1. Create a dummy list of Document objects
    dummy_documents = [
        Document(page_content="The sun is a star at the center of our solar system."),
        Document(page_content="Earth is the third planet from the Sun and the only astronomical object known to harbor life."),
        Document(page_content="A black hole is a region of spacetime where gravity is so strong that nothing—no particles or even electromagnetic radiation such as light—can escape from it.")
    ]
    
    # Ensure the directory exists
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # 2. Get the ChromaDB instance
    print("Getting ChromaDB instance...")
    vector_db = get_vector_db_instance()
    
    # 3. Add the documents to the database
    print("Adding documents to the database...")
    add_documents_to_db(vector_db, dummy_documents)
    
    # 4. Perform a simple similarity search to verify the process
    print("\nPerforming a similarity search to verify...")
    query = "What is the third planet from the sun?"
    retrieved_docs = vector_db.similarity_search(query)
    
    print(f"Query: '{query}'")
    print(f"Retrieved document content: {retrieved_docs[0].page_content}")
    
    # 5. Clean up the test files
    # 5. Clean up the test files
    if os.path.exists(PERSIST_DIRECTORY):
        print("\nCleaning up the test database...")

        # Ensure DB is closed before deleting
        vector_db._client._system.stop()  # Gracefully stop Chroma server
        del vector_db
        import gc
        gc.collect()

        import shutil
        shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
        print("Cleanup complete.")
