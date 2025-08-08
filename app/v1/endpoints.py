# smartfile_ai/app/v1/endpoints.py

import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi import Request
from app.core import pdf_parser, embedder, retriever

# Create an API router for version 1
router = APIRouter(prefix="/v1")

# Pydantic model for the chat request body
class ChatQuery(BaseModel):
    query: str

@router.post("/upload-pdf/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files, parse their content,
    and store the text embeddings in the vector database.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    upload_dir = "data/uploaded_pdfs"
    os.makedirs(upload_dir, exist_ok=True)

    all_text = ""
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        # Save the uploaded file locally
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract text using our parser
        text = pdf_parser.get_pdf_text_pymupdf(file_path)
        if not text:
            print(f"Warning: Could not extract text from {file.filename}")
            continue
        all_text += text + "\n\n"

    if not all_text:
        raise HTTPException(status_code=500, detail="Could not extract text from any of the uploaded files.")
    
    # Chunk the combined text and embed it
    text_chunks = pdf_parser.get_text_chunks(all_text)
    vector_db = embedder.get_vector_db_instance()
    embedder.add_documents_to_db(vector_db, text_chunks)

    return {"message": f"Successfully processed {len(files)} file(s) and updated the vector database."}

@router.post("/chat/")
async def chat_with_pdf(chat_query: ChatQuery, request: Request):
    qa_chain = request.app.state.qa_chain
    response = qa_chain.invoke({"query": chat_query.query})
    return {"response": response['result']}
