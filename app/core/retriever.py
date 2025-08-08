# smartfile_ai/app/core/retriever.py

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.core import embedder, llm_handler
from langchain.schema import Document

def get_qa_chain():
    """
    Builds and returns the complete RetrievalQA chain.
    """
    # Get the vector database instance as a retriever
    vector_db = embedder.get_vector_db_instance()
    # The retriever is responsible for finding relevant documents
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Get the LLM and prompt template from our handler
    llm = llm_handler.get_llm()
    prompt = llm_handler.get_prompt_template()
    
    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# --- Temporary Test Block ---
if __name__ == "__main__":
    print("--- Running retriever.py temporary test block ---")
    
    # This test demonstrates the full RAG pipeline from a query
    print("Building a temporary vector database and QA chain...")
    
    # Dummy documents for testing
    dummy_documents = [
        Document(page_content="The sun is a star at the center of our solar system."),
        Document(page_content="Earth is the third planet from the Sun and the only astronomical object known to harbor life."),
        Document(page_content="A black hole is a region of spacetime where gravity is so strong that nothing can escape.")
    ]
    
    # Add dummy documents to a temporary database
    temp_db = embedder.get_vector_db_instance()
    embedder.add_documents_to_db(temp_db, dummy_documents)
    
    # Now, get the fully configured QA chain
    qa_chain = get_qa_chain()
    
    # Test queries
    test_queries = [
        "What is a black hole?",
        "What is the third planet from the sun?"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")
        try:
            response = qa_chain.invoke({"query": query})
            print(f"Response: {response['result']}")
        except Exception as e:
            print(f"Error during chain invocation: {e}")
    
    # Clean up the temporary database
    print("\n--- Cleaning up temporary database ---")
    try:
        temp_db._client._system.stop()
        del temp_db
        import shutil
        shutil.rmtree(embedder.PERSIST_DIRECTORY, ignore_errors=True)
        print("Cleanup complete.")
    except Exception as e:
        print(f"Error during cleanup: {e}")