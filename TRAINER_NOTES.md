# Trainer Notes — Local RAG Assistant

## Teaching objective

By the end of the session, students should be able to draw and explain:

```text
Document → Chunk → Embedding → Retrieval → Context → Qwen → Answer
```

Do not begin by explaining every function in the code.

Start with the problem.

---

## Opening question

Ask students:

> "Suppose I give Qwen a 300-page company policy. How can we make the application answer questions specifically from that policy?"

Let students suggest ideas.

Then reveal:

> "We do not send the entire document every time. We retrieve the useful pieces first."

---

## Whiteboard explanation

Draw:

```text
             KNOWLEDGE BASE
             ┌─────────────┐
Document ───►│   Chunks    │
             └──────┬──────┘
                    │
                    ▼
             Embedding Model
                    │
                    ▼
               Vector Store
                    │
                    │
Question ──► Query Embedding
                    │
                    ▼
             Similarity Search
                    │
                    ▼
               Top-K Chunks
                    │
                    ▼
                 Qwen 2.5
                    │
                    ▼
                  Answer
```

---

## The three questions students must understand

### Question 1

**Where does the information come from?**

Answer:

> Uploaded documents.

### Question 2

**How does the application find relevant text?**

Answer:

> Embeddings + similarity search.

### Question 3

**Who writes the final answer?**

Answer:

> Qwen 2.5:3B.

---

## Important misconception to correct

Students may say:

> "The RAG system trained Qwen on my PDF."

Correct them.

The application did not fine-tune Qwen.

It created temporary retrieval context and placed that context into the prompt.

Use this analogy:

### RAG = open-book exam

The student does not memorize the entire book during the exam.

The student looks up the relevant page and answers using that page.

### Fine-tuning = additional training

The model's parameters are changed through training.

---

## Explain why we use two models

### Model 1 — Embedding model

Purpose:

```text
text → vector
```

### Model 2 — Qwen 2.5:3B

Purpose:

```text
context + question → natural language answer
```

Say this sentence aloud:

> "Embeddings are for finding; Qwen is for explaining."

---

## Teaching cosine similarity

Do not start with the formula.

Start with the idea:

> "We need a number telling us how related two vectors are."

Then introduce:

```text
Cosine similarity
= similarity of vector direction
```

Then show:

```python
scores = vectors @ query_vector
```

Because vectors are normalized, the dot product gives cosine similarity.

---

## Teaching Top-K

Use a search engine analogy.

If you search Google, it does not return the entire internet.

It returns the most relevant results.

Top-K means:

> "How many relevant chunks should we give Qwen?"

---

## Teaching retrieval debugging

This is one of the most important skills.

When the answer is wrong, do NOT immediately blame Qwen.

First inspect:

```text
What chunks were retrieved?
```

The correct debugging order is:

```text
Question
  ↓
Was the query embedded correctly?
  ↓
Were relevant chunks retrieved?
  ↓
Was useful context sent?
  ↓
Did Qwen follow the prompt?
```

This teaches students to debug pipelines instead of blaming the LLM.

---

## 3-stage classroom challenge

### Stage 1 — Easy

Upload a one-page document.

Ask direct questions.

Goal:

> Verify the basic pipeline.

### Stage 2 — Medium

Upload multiple documents.

Ask comparison questions.

Goal:

> Understand multi-document retrieval.

### Stage 3 — Advanced

Ask a question whose answer is not in the documents.

Goal:

> Observe grounded refusal behavior.

---

## Mini assessment

Ask students to explain these in one sentence each:

**Chunking**

> Splitting documents into smaller searchable passages.

**Embedding**

> Converting text into numerical vectors representing semantic information.

**Retrieval**

> Finding the most relevant chunks for a query.

**Context**

> Retrieved information given to the language model.

**Generation**

> Producing the final natural-language response.

**RAG**

> Retrieve relevant information and use it to augment generation.

---

## Suggested final assignment

Build a domain-specific RAG assistant for one of:

- College handbook
- HR policy
- Placement preparation
- Software documentation
- Product manuals
- Company SOPs

Deliver:

1. working Streamlit app
2. at least three documents
3. screenshots of retrieval results
4. five test questions
5. two questions whose answers are intentionally absent
6. one-page explanation of the RAG architecture

---

## Trainer closing line

End the session with:

> "A good RAG system is not just a chatbot. It is a retrieval pipeline plus a generation model. When the answer is wrong, inspect the retrieved evidence first."
