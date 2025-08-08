FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY ./data ./data
COPY .env .

ENV MODEL_ID=google/flan-t5-small
ENV VECTOR_DB=chroma
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
