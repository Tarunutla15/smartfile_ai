# smartfile_ai/app/core/pdf_parser.py

import fitz  # PyMuPDF is imported as fitz
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

def get_pdf_text_pypdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using pypdf.
    
    Args:
        pdf_path (str): The path to the PDF file.
        
    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text with pypdf: {e}")
        return ""
    return text

def get_pdf_text_pymupdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    This method is generally more robust for complex PDFs.
    
    Args:
        pdf_path (str): The path to the PDF file.
        
    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    try:
        document = fitz.open(pdf_path)
        for page in document:
            text += page.get_text()
    except Exception as e:
        print(f"Error extracting text with PyMuPDF: {e}")
        return ""
    return text

def get_text_chunks(text: str) -> List[Document]:
    """
    Splits the given text into chunks using a LangChain text splitter.
    
    Args:
        text (str): The text to be chunked.
        
    Returns:
        List[Document]: A list of LangChain Document objects, each representing a chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.create_documents([text])
    return chunks

# Add this block to the end of app/core/pdf_parser.py for testing
if __name__ == "__main__":
    # Create a dummy PDF file for testing purposes
    # In a real scenario, you'd use a real PDF from your project folder
    test_pdf_path = f"C:\\Users\\Admin\\OneDrive\\Desktop\\Lecture Notes-ANN&DL-20CS702.pdf"
    
    # Create a dummy PDF with some text
    try:
        dummy_text = "This is a test document to check the PDF parser. It contains sample information that will be extracted and chunked. This process is crucial for the RAG pipeline to work correctly. The parser must handle text extraction reliably to ensure the embeddings are accurate. The document splitter then takes this text and breaks it down into smaller, manageable pieces, which helps in efficient retrieval." * 10
        doc = fitz.open()  # Create a new, empty PDF
        page = doc.new_page()
        page.insert_text((50, 50), dummy_text)
        doc.save(test_pdf_path)
        doc.close()
        print(f"Created a temporary test PDF at {test_pdf_path}")
    except Exception as e:
        print(f"Error creating dummy PDF: {e}")
        test_pdf_path = None

    if test_pdf_path:
        # Test the PyMuPDF parser
        pdf_text_pymupdf = get_pdf_text_pymupdf(test_pdf_path)
        print("\n--- Testing PyMuPDF Parser ---")
        print(f"Extracted text length: {len(pdf_text_pymupdf)}")
        print("First 100 characters:", pdf_text_pymupdf[:100].replace('\n', ' '))
        
        # Test the chunking function
        chunks = get_text_chunks(pdf_text_pymupdf)
        print("\n--- Testing Text Chunking ---")
        print(f"Total number of chunks: {len(chunks)}")
        if chunks:
            print(f"First chunk length: {len(chunks[0].page_content)}")
            print("First chunk content:", chunks[0].page_content[:100].replace('\n', ' '))
            
        # Clean up the test file
        import os
        os.remove(test_pdf_path)
        print(f"\nCleaned up the temporary test PDF.")