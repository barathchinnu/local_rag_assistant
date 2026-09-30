import streamlit as st

from config import (
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_API_BASE,
    OLLAMA_TIMEOUT,
)
from ollama_client import (
    check_ollama_status,
    chat_with_ollama,
)
from rag_pipeline import (
    extract_text_from_file,
    chunk_text,
    build_embeddings,
    retrieve,
)

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Local RAG Assistant")
st.caption("Qwen 2.5:3B + Ollama + Streamlit — ask questions about your own documents")

# -----------------------------
# Session state
# -----------------------------
defaults = {
    "messages": [],
    "chunks": [],
    "vectors": None,
    "documents": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ RAG Controls")

    st.write(f"**Generator:** `{OLLAMA_MODEL}`")
    st.write(f"**Embedding:** `{EMBEDDING_MODEL}`")
    st.write(f"**API:** `{OLLAMA_API_BASE}`")

    st.divider()

    st.subheader("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=300,
        max_value=1800,
        value=900,
        step=100,
        help="Approximate number of characters in each chunk.",
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=400,
        value=150,
        step=25,
        help="Characters repeated between neighboring chunks.",
    )

    top_k = st.slider(
        "Top-K retrieved chunks",
        min_value=1,
        max_value=6,
        value=3,
        help="How many relevant chunks are sent to Qwen.",
    )

    build_button = st.button(
        "🔨 Build / Rebuild Knowledge Base",
        type="primary",
        use_container_width=True,
    )

    clear_kb = st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True,
    )

    clear_chat = st.button(
        "🧹 Clear Chat",
        use_container_width=True,
    )

    if clear_kb:
        st.session_state.chunks = []
        st.session_state.vectors = None
        st.session_state.documents = []
        st.rerun()

    if clear_chat:
        st.session_state.messages = []
        st.rerun()

# -----------------------------
# Ollama status
# -----------------------------
status = check_ollama_status()

if status["ok"]:
    st.success(
        f"Ollama ready • Qwen: {status['generator']} • "
        f"Embedding: {status['embedding']}"
    )
else:
    st.error(status["message"])

# -----------------------------
# Build the knowledge base
# -----------------------------
if build_button:
    if not uploaded_files:
        st.warning("Upload at least one document first.")
    elif not status["ok"]:
        st.error("Fix the Ollama status before building the knowledge base.")
    elif chunk_overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size.")
    else:
        all_chunks = []
        document_names = []

        with st.status("Building knowledge base...", expanded=True) as build_status:
            for uploaded_file in uploaded_files:
                try:
                    raw_text = extract_text_from_file(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                    )

                    if not raw_text.strip():
                        st.warning(f"No readable text found in {uploaded_file.name}.")
                        continue

                    chunks = chunk_text(
                        raw_text,
                        chunk_size=chunk_size,
                        overlap=chunk_overlap,
                    )

                    for index, chunk in enumerate(chunks, start=1):
                        all_chunks.append(
                            {
                                "text": chunk,
                                "source": uploaded_file.name,
                                "chunk_id": index,
                            }
                        )

                    document_names.append(uploaded_file.name)
                    st.write(
                        f"✅ {uploaded_file.name}: "
                        f"{len(chunks)} chunks"
                    )
                except Exception as exc:
                    st.error(f"Could not process {uploaded_file.name}: {exc}")

            if all_chunks:
                st.write(
                    f"🧠 Creating embeddings for {len(all_chunks)} chunks..."
                )
                vectors = build_embeddings(
                    [item["text"] for item in all_chunks]
                )

                st.session_state.chunks = all_chunks
                st.session_state.vectors = vectors
                st.session_state.documents = document_names

                build_status.update(
                    label="Knowledge base ready",
                    state="complete",
                )
                st.success(
                    f"Indexed {len(document_names)} document(s) into "
                    f"{len(all_chunks)} chunks."
                )
            else:
                build_status.update(
                    label="No usable text found",
                    state="error",
                )

# -----------------------------
# Knowledge base summary
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Documents", len(st.session_state.documents))
with col2:
    st.metric("Chunks", len(st.session_state.chunks))
with col3:
    st.metric("Top-K", top_k)

if st.session_state.documents:
    st.info(
        "Loaded documents: " + ", ".join(st.session_state.documents)
    )
else:
    st.info(
        "Upload one or more documents, build the knowledge base, "
        "then ask questions below."
    )

st.divider()

# -----------------------------
# Chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("🔎 Retrieved sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['source']}** — chunk {source['chunk_id']} "
                        f"— similarity `{source['score']:.3f}`"
                    )

# -----------------------------
# Chat input
# -----------------------------
user_question = st.chat_input(
    "Ask a question about your uploaded documents..."
)

if user_question:
    if not status["ok"]:
        st.error("Ollama is not ready. Start Ollama and refresh the page.")
        st.stop()

    if not st.session_state.chunks or st.session_state.vectors is None:
        st.warning(
            "Build the knowledge base first. Otherwise there is no document "
            "context for the RAG system."
        )
        st.stop()

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Retrieve relevant chunks
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context and asking Qwen..."):
            retrieved = retrieve(
                query=user_question,
                chunks=st.session_state.chunks,
                vectors=st.session_state.vectors,
                top_k=top_k,
            )

            context_blocks = []
            for index, item in enumerate(retrieved, start=1):
                context_blocks.append(
                    f"[Source {index}] {item['source']} — "
                    f"chunk {item['chunk_id']}\n"
                    f"{item['text']}"
                )

            context = "\n\n".join(context_blocks)

            rag_system_prompt = """
You are a local document-grounded assistant.

Rules:
1. Answer using the supplied document context whenever possible.
2. Do not invent facts that are not supported by the context.
3. If the answer is not present in the context, say:
   "I couldn't find that information in the uploaded documents."
4. Give a concise, student-friendly explanation.
5. When useful, mention the source file name.
""".strip()

            user_prompt = f"""
DOCUMENT CONTEXT:
{context}

QUESTION:
{user_question}
""".strip()

            answer = chat_with_ollama(
                system_prompt=rag_system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )

        st.markdown(answer)

        with st.expander("🔎 Retrieved sources"):
            for source in retrieved:
                st.markdown(
                    f"- **{source['source']}** — chunk {source['chunk_id']} "
                    f"— similarity `{source['score']:.3f}`"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved,
        }
    )
