# Local RAG Assistant — Qwen 2.5:3B + Ollama + Streamlit

A beginner-friendly, fully local Retrieval-Augmented Generation (RAG) application.

## What this project does

1. Upload PDF, TXT or DOCX files.
2. Extract text from the documents.
3. Split text into smaller chunks.
4. Generate embeddings locally with `nomic-embed-text`.
5. Store embeddings in memory with NumPy.
6. Embed the user's question.
7. Retrieve the most similar chunks using cosine similarity.
8. Send the retrieved context to `qwen2.5:3b`.
9. Display the answer and retrieved sources in Streamlit.

## Architecture

```text
                ┌──────────────────────────┐
                │   Streamlit Web UI       │
                │ Upload + Chat + Status   │
                └────────────┬─────────────┘
                             │
                    Uploaded documents
                             │
                             v
                ┌──────────────────────────┐
                │   Text Extraction        │
                │ PDF / DOCX / TXT         │
                └────────────┬─────────────┘
                             │
                             v
                ┌──────────────────────────┐
                │     Chunking             │
                │ small text passages      │
                └────────────┬─────────────┘
                             │
                             v
                ┌──────────────────────────┐
                │ Ollama Embedding Model   │
                │   nomic-embed-text       │
                └────────────┬─────────────┘
                             │
                     vectors in memory
                             │
                             v
User question ───────► query embedding
                             │
                             v
                ┌──────────────────────────┐
                │ Cosine Similarity        │
                │ Top-K Retrieval           │
                └────────────┬─────────────┘
                             │
                    relevant context
                             │
                             v
                ┌──────────────────────────┐
                │ Ollama Chat API          │
                │      Qwen 2.5:3B         │
                └────────────┬─────────────┘
                             │
                             v
                      grounded answer
```

## Models

- Generator: `qwen2.5:3b`
- Embedding model: `nomic-embed-text`

The embedding model is separate because document retrieval requires vector embeddings, while Qwen is used here to generate the final natural-language answer.

## Setup

### Windows PowerShell

```powershell
ollama --version
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

### Optional environment variables

```powershell
$env:OLLAMA_MODEL="qwen2.5:3b"
$env:EMBEDDING_MODEL="nomic-embed-text"
$env:OLLAMA_API_BASE="http://localhost:11434/api"
$env:OLLAMA_TIMEOUT="180"
```

## Recommended classroom demonstration

Use one short college policy, syllabus, HR document, product specification, or training handout.

Ask:
- "What are the attendance rules?"
- "What is the minimum percentage?"
- "Who approves leave?"
- "What happens if a student misses the requirement?"

Then ask a question that does not exist in the document and observe that the assistant should say it cannot find the information.

## Important limitation

This classroom project stores vectors in memory. Restarting the Streamlit app removes the index. That is intentional: students can understand the RAG pipeline before introducing a persistent vector database such as Chroma, FAISS, or a hosted vector database.
