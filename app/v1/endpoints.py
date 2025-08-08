import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import List
from app.core import pdf_parser, embedder, retriever, version_control

router = APIRouter(prefix="/v1")

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

    # No persist() call here because it's deprecated or handled internally
    
    # Now track and push changes with DVC
    try:
        version_control.dvc_add_commit_push(upload_dir)
        version_control.dvc_add_commit_push(embedder.PERSIST_DIRECTORY)
    except Exception as e:
        print(f"Warning: DVC tracking failed: {e}")

    return {"message": f"Successfully processed {len(files)} file(s) and updated the vector database."}

@router.post("/chat/")
async def chat_with_pdf(chat_query: ChatQuery, request: Request):
    qa_chain = request.app.state.qa_chain
    response = qa_chain.invoke({"query": chat_query.query})
    return {"response": response['result']}
