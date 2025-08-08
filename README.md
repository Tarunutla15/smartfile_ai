# SmartFile AI

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.0.186-orange)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🚀 Project Overview

**SmartFile AI** is an AI-powered chatbot that allows users to upload PDF documents and interact with their contents via natural language queries.  
It leverages:

- LangChain's Retrieval-Augmented Generation (RAG) pipeline
- Vector database (ChromaDB) for embeddings storage and retrieval
- Transformer-based LLM (e.g., Google FLAN-T5) for generating answers
- FastAPI backend for RESTful API
- Streamlit for user-friendly UI
- DVC for version control of PDFs and embeddings

---

## 📂 Folder Structure
```
smartfile_ai/
├── app/
│ ├── core/
│ │ ├── embedder.py
│ │ ├── llm_handler.py
│ │ ├── pdf_parser.py
│ │ └── retriever.py
│ ├── v1/
│ │ └── endpoints.py
│ └── main.py
├── data/
│ └── uploaded_pdfs/
├── logs/
├── ui.py
├── requirements.txt
├── Dockerfile
└── .env
```

---

## ⚙️ Features

- Upload multiple PDFs and parse their text content
- Chunk and embed the extracted text into vector DB
- Query PDFs using natural language via an LLM-powered chatbot
- Persistent vector DB storage using ChromaDB
- DVC integration to track data and embeddings versions
- Streamlit-based UI for interaction
- FastAPI backend with detailed logging and metrics

---

## 🛠️ Installation & Setup

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/smartfile_ai.git
cd smartfile_ai
```
2. **Set up a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. ***Install dependencies***
```bash
pip install -r requirements.txt
```
4. ***Configure environment variables***
```bash
MODEL_ID=google/flan-t5-small
VECTOR_DB=chroma
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_FILE=logs/query_log.json
UPLOAD_DIR=data/uploaded_pdfs
```
5. ***Initialize DVC and Git***
```bash
git init
dvc init
```
6. ***Run the FastAPI server***
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
7. ***Run Streamlit UI***
```bash
streamlit run ui.py
```
🧩 Usage
- Use /v1/upload-pdf/ endpoint to upload PDFs.

- Use /v1/chat/ endpoint to query the content of uploaded PDFs.

- Interact via the Streamlit UI for file upload and chat.

📝 Logging & Metrics
- API queries, responses, and latency are logged in logs/query_log.json.

- DVC tracks changes to uploaded PDFs and vector DB embeddings.

📦 Docker Deployment
Build the Docker image:
```bash
docker build -t smartfile_ai .
```
Run the container:
```bash
docker run -p 8000:8000 smartfile_ai
```
