# main.py
from fastapi import FastAPI
from .v1 import endpoints as v1_endpoints
from app.core import retriever

app = FastAPI(
    title="SmartFile AI",
    description="Chat with Your PDFs using a Local RAG Pipeline",
    version="1.0.0",
)

# Preload QA chain on startup
@app.on_event("startup")
async def startup_event():
    print("Loading QA chain into memory...")
    app.state.qa_chain = retriever.get_qa_chain()
    print("QA chain loaded successfully.")

app.include_router(v1_endpoints.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to SmartFile AI! Go to /docs to use the API."}
