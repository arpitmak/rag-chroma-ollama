# RAG with ChromaDB and Ollama

A local Retrieval-Augmented Generation (RAG) system using:
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- Ollama (local LLM)

## Features
- PDF ingestion
- Metadata filtering
- Vector similarity search
- Local LLM inference (no API keys)

## How to Run
```bash
pip install -r requirements.txt
python ingest.py
python rag.py
