# Lab: Build a Local RAG Assistant

**Project:** Local RAG Assistant  
**Technology:** Qwen 2.5:3B + Ollama + Python + Streamlit + NumPy  
**Level:** Beginner to Intermediate  
**Suggested time:** 3–4 hours  
**Method:** Build first, explain each RAG stage while testing it.

---

## 1. What you will build

You will build a document-question-answering application that works locally.

The user uploads documents. The application retrieves the most relevant pieces of those documents and gives those pieces to Qwen 2.5:3B before generating the final answer.

The pipeline is:

```text
Document
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Store (in memory)
   ↓
User Question
   ↓
Query Embedding
   ↓
Similarity Search
   ↓
Top-K Chunks
   ↓
Qwen 2.5:3B
   ↓
Grounded Answer
```

The original LocalGenAI lab uses Streamlit for the interface and Ollama for the local model/API connection. This lab keeps that same local architecture and adds the retrieval layer. 
---

## 2. The most important idea

### Normal LLM application

```text
Question → Qwen → Answer
```

The model depends mainly on what it learned during training plus the prompt.

### RAG application

```text
Question
   ↓
Find relevant document chunks
   ↓
Add those chunks to the prompt
   ↓
Qwen
   ↓
Answer
```

Think:

**RAG = Retrieve first, Generate second.**

---

## 3. The four main components

| Component | Simple meaning | Role |
|---|---|---|
| Streamlit | Web UI made with Python | Uploads files and displays chat |
| Ollama | Local AI runtime + API server | Runs local models |
| `nomic-embed-text` | Embedding model | Converts text into vectors |
| `qwen2.5:3b` | Language model | Generates the final answer |

### Trainer note

Stress this distinction:

> **Embedding model = search engine brain**  
> **Generation model = explanation brain**

Qwen 2.5:3B is used for generation in this project. We use a separate embedding model for semantic retrieval.

---

## 4. Prerequisites

You need:

- Windows, macOS or Linux
- Python 3.10+
- Ollama
- Enough RAM for local model inference
- Internet access for the initial software/model downloads

---

## 5. Install Ollama models

Open a terminal:

```powershell
ollama --version
```

Download the generator:

```powershell
ollama pull qwen2.5:3b
```

Download the embedding model:

```powershell
ollama pull nomic-embed-text
```

Verify:

```powershell
ollama list
```

You should see both models.

---

## 6. Create the Python environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `Activate.ps1` is blocked, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

---

## 7. Run the application

```powershell
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 8. Trainer walkthrough — understand the RAG pipeline

## Stage A — Document ingestion

When the student uploads a PDF, DOCX or TXT file:

```python
raw_text = extract_text_from_file(
    uploaded_file.name,
    uploaded_file.getvalue(),
)
```

### Explain

A file is not directly useful to the language model.

First, we convert the file into plain text.

### Analogy

A PDF is like a sealed book.

Text extraction is like opening the book and copying the words onto a sheet.

### Ask students

> "Can Qwen directly search every PDF on my laptop?"

Expected answer:

> No. The application needs to load the document, extract its content and provide relevant text to the model.

---

## Stage B — Chunking

Important code:

```python
chunks = chunk_text(
    raw_text,
    chunk_size=900,
    overlap=150,
)
```

### Explain

Large documents are broken into smaller passages.

Why?

Because retrieval works better when the searchable units are focused pieces of text.

### Analogy

Imagine a 500-page textbook.

You would not search by carrying the entire book to every conversation.

You use chapters and sections.

Chunks are those smaller searchable sections.

### Chunk size

`chunk_size=900`

Meaning:

> approximately 900 characters per chunk in this teaching implementation.

### Overlap

`overlap=150`

Meaning:

> 150 characters can be repeated between neighboring chunks.

This reduces the chance of cutting an important sentence exactly at a chunk boundary.

### Important trainer note

This is a deliberately simple character-based chunker.

Production systems may use:

- token-aware chunking
- paragraph-aware chunking
- heading-aware chunking
- semantic chunking

Do not present the classroom chunker as the only correct strategy.

---

# 9. Stage C — Embeddings

The next step is:

```python
embeddings = embed_texts(texts)
```

The application sends chunks to:

```text
http://localhost:11434/api/embed
```

using:

```text
nomic-embed-text
```

### What is an embedding?

An embedding is a numeric representation of meaning.

For example:

```text
"Students must maintain 75% attendance."
```

becomes something like:

```text
[0.12, -0.44, 0.08, ...]
```

The real vector contains many numbers.

We do not manually interpret each number.

We compare vectors mathematically.

### Analogy

Think about a map.

Words with related meaning are placed closer together in "meaning space."

For example:

```text
attendance requirement
attendance percentage
minimum attendance
```

should be semantically closer than:

```text
attendance requirement
banana recipe
```

---

# 10. Stage D — Vector storage

This project uses:

```python
st.session_state.vectors
```

and NumPy.

### Explain

A vector store simply needs to keep:

1. the document chunk
2. its embedding
3. metadata such as file name and chunk number

This teaching version stores vectors in memory.

### Why not use a vector database immediately?

Because beginners first need to understand:

```text
TEXT → VECTOR → SEARCH → TOP RESULTS
```

After they understand this, introduce:

- FAISS
- Chroma
- Pinecone
- Qdrant
- Weaviate

---

# 11. Stage E — Query embedding

When the user asks:

```text
"What is the attendance requirement?"
```

the question is also converted into an embedding:

```python
query_vector = embed_texts([query])[0]
```

Now we have:

```text
document chunk vectors
+
query vector
```

and can compare them.

---

# 12. Stage F — Cosine similarity

The code:

```python
scores = vectors @ query_vector
```

works after the vectors are normalized.

### Explain

Cosine similarity measures how closely two vectors point in the same direction.

A simplified interpretation:

```text
1.0  → very similar
0.7  → fairly similar
0.0  → unrelated
-1.0 → opposite direction
```

In practical text retrieval, students should think:

> **Higher score = more semantically similar.**

### Analogy

Imagine two students walking.

If they walk in almost the same direction, their direction vectors are similar.

If they walk in very different directions, similarity is lower.

---

# 13. Stage G — Top-K retrieval

The UI contains:

```python
top_k = st.slider(...)
```

Example:

```text
Top-K = 3
```

means:

> Retrieve the three most similar chunks.

The code:

```python
top_indices = np.argsort(scores)[::-1][:top_k]
```

selects the highest-scoring chunks.

### Trainer challenge

Ask:

> "What happens when Top-K changes from 3 to 5?"

Expected:

> More document context is sent to Qwen, but irrelevant context may also increase.

This is the beginning of retrieval tuning.

---

# 14. Stage H — Augmentation

Now the application builds a context:

```text
[Source 1] handbook.pdf — chunk 4
...

[Source 2] handbook.pdf — chunk 7
...
```

Then it creates a prompt:

```text
DOCUMENT CONTEXT:
...

QUESTION:
...
```

### Explain

This is why the technique is called:

**Retrieval-Augmented Generation**

- Retrieval = find relevant information
- Augmented = add it to the prompt
- Generation = let Qwen produce the answer

---

# 15. Stage I — Generation with Qwen

The final request goes to:

```text
POST /api/chat
```

with:

```python
{
    "model": "qwen2.5:3b",
    "messages": [...],
    "stream": False
}
```

Qwen reads:

```text
System instructions
+
Retrieved document context
+
User question
```

and generates the answer.

The original LocalGenAI lab uses the same local `/api/chat` architecture for Qwen. 
---

# 16. Why the system prompt matters

The RAG system prompt says:

```text
Answer using the supplied document context whenever possible.

Do not invent facts that are not supported by the context.

If the answer is not present in the context, say:
"I couldn't find that information in the uploaded documents."
```

### Explain

This does not mathematically guarantee zero hallucinations.

It gives the model a strong instruction to stay grounded.

### Teaching point

RAG reduces the model's need to rely only on remembered knowledge, but retrieval quality and prompt design still matter.

---

# 17. Complete request journey

Students should be able to explain this without looking at the code:

```text
1. User uploads a document.
2. Python extracts the text.
3. Text is split into chunks.
4. Embedding model converts chunks into vectors.
5. Vectors are stored in memory.
6. User asks a question.
7. The question becomes a vector.
8. Cosine similarity ranks document chunks.
9. Top-K chunks are retrieved.
10. Retrieved chunks are placed into the prompt.
11. Qwen 2.5:3B generates the answer.
12. Streamlit displays the answer and sources.
```

---

# 18. Suggested classroom demo

Use a short real document such as:

- college syllabus
- HR policy
- company product document
- training handbook
- leave policy
- course PDF

### Demo 1 — Grounded question

Ask:

```text
What is the attendance requirement?
```

Students should see an answer based on the uploaded document.

### Demo 2 — Follow-up

Ask:

```text
Who approves the leave?
```

### Demo 3 — Missing information

Ask something clearly absent:

```text
What is the salary of the principal?
```

Expected behavior:

```text
I couldn't find that information in the uploaded documents.
```

### Demo 4 — Change Top-K

Set:

```text
Top-K = 1
```

then:

```text
Top-K = 5
```

Discuss how the retrieved context changes.

---

# 19. Hands-on student tasks

## Task 1 — Understand chunking

Change:

```python
chunk_size=900
```

to:

```python
chunk_size=500
```

Compare:

- number of chunks
- retrieval quality
- answer quality

---

## Task 2 — Experiment with overlap

Test:

```text
0
100
250
```

Ask the same question each time.

Discuss:

> Does overlap improve retrieval for your document?

---

## Task 3 — Experiment with Top-K

Test:

```text
Top-K = 1
Top-K = 3
Top-K = 5
```

Write down:

- answer relevance
- amount of context
- response speed

---

## Task 4 — Add source metadata

Modify the chunk dictionary to include:

```python
{
    "text": chunk,
    "source": uploaded_file.name,
    "chunk_id": index,
}
```

Then display that metadata in the UI.

---

## Task 5 — Create a Career RAG assistant

Upload:

- placement preparation material
- resume guide
- interview guide
- company profile

Ask:

```text
What skills should I prepare for this role?
```

---

# 20. Advanced extension ideas

After students understand the basic pipeline, introduce:

### Level 1

- streaming responses
- better chunking
- PDF page numbers
- source highlighting

### Level 2

- persistent FAISS index
- Chroma vector database
- metadata filters
- hybrid search

### Level 3

- reranking
- query rewriting
- multi-document comparison
- conversational RAG
- evaluation datasets
- RAGAS-style evaluation
- LangChain / LlamaIndex abstraction

---

# 21. Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| Ollama offline | Ollama service not running | Start Ollama |
| Qwen missing | Model not downloaded | `ollama pull qwen2.5:3b` |
| Embedding model missing | Embedding model not downloaded | `ollama pull nomic-embed-text` |
| Port 11434 unavailable | Ollama is not running | Start Ollama |
| No chunks | File contains no extractable text | Try another document or OCR-enabled workflow |
| Very slow | CPU/RAM limitations | Close other applications |
| Poor answers | Bad chunking/retrieval | Tune chunk size, overlap and Top-K |
| Answer ignores document | Weak retrieval or poor context | Inspect retrieved chunks first |
| PowerShell activation blocked | Script execution policy | Use Command Prompt activation |

---

# 22. Trainer notes — questions to ask students

### Beginner

1. What is RAG?
2. What is an embedding?
3. Why do we chunk documents?
4. What is Top-K?
5. What is cosine similarity?
6. Why do we need an embedding model?
7. What does Qwen do?

### Intermediate

1. Why can very large chunks hurt retrieval?
2. Why can very small chunks hurt context?
3. What happens when Top-K is too high?
4. Why can good retrieval still produce a bad answer?
5. What is the difference between retrieval and generation?
6. Why is this project considered local?

### Advanced

1. When would FAISS be better than an in-memory NumPy store?
2. How would you preserve the index after restart?
3. How could reranking improve results?
4. How would you evaluate RAG quality separately for retrieval and generation?
5. When would fine-tuning be preferable to RAG?

---

# 23. Core technical terms

| Term | Meaning |
|---|---|
| RAG | Retrieval-Augmented Generation |
| Chunk | Small passage of a document |
| Embedding | Numerical representation of semantic meaning |
| Vector | Numeric representation used for similarity |
| Vector store | System that stores vectors + metadata |
| Retrieval | Finding relevant chunks |
| Top-K | Number of highest-ranked chunks returned |
| Cosine similarity | Measure of vector directional similarity |
| Context | Retrieved information supplied to the LLM |
| Grounding | Keeping the answer tied to supplied evidence |
| Hallucination | Unsupported or invented model output |
| Generator | LLM that writes the final response |
| Retriever | Component that finds relevant information |

---

# 24. Mnemonic

Remember:

## **RAG = READ → ARRANGE → GENERATE**

**R — Read** the user's documents  
**A — Arrange** relevant chunks as context  
**G — Generate** the answer with Qwen

A more technical memory trick:

## **Load → Chunk → Embed → Retrieve → Prompt → Generate**

---

# 25. Quick recap for students

A normal local chatbot is:

```text
Question → Qwen → Answer
```

A local RAG chatbot is:

```text
Documents
   ↓
Chunk
   ↓
Embed
   ↓
Retrieve
   ↓
Context + Question
   ↓
Qwen
   ↓
Answer + Sources
```

The key lesson is:

> **RAG does not make the model "learn" the document permanently. It retrieves relevant information at query time and places that information into the prompt.**

---
