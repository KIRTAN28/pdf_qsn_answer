"""
RAG-Based PDF Question Answering System
Upload a PDF and chat with its contents using Mistral OCR + OpenAI + FAISS.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from pdf_processor import load_and_split
from vector_store import create_vector_store, get_retriever
from rag_chain import build_rag_chain, ask_question


load_dotenv()


st.set_page_config(
    page_title="PDF Q&A – RAG System",
    page_icon="📄",
    layout="wide",
)

# 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #e0e0e0;
}

/* Chat message styling */
.stChatMessage {
    border-radius: 12px;
    margin-bottom: 8px;
}

/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0;
}
.sub-header {
    color: #888;
    font-size: 1rem;
    margin-top: -8px;
    margin-bottom: 24px;
}

/* Success card */
.success-card {
    background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
    border-left: 4px solid #48bb78;
    border-radius: 8px;
    padding: 16px 20px;
    color: #e0e0e0;
    margin: 12px 0;
}

/* Upload area */
.upload-header {
    color: #a78bfa;
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# Sidebar: PDF upload 
with st.sidebar:
    st.markdown("### 📁 Upload Your PDF")
    st.markdown(
        "<span style='color:#94a3b8;font-size:0.85rem;'>"
        "Upload a PDF to start asking questions about its content."
        "</span>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Only re-process if a new file is uploaded
        if st.session_state.pdf_name != uploaded_file.name:
            if not os.getenv("MISTRAL_API_KEY"):
                st.error(" MISTRAL_API_KEY not found in .env file!")
            else:
                with st.spinner(" Running Mistral OCR on PDF…"):
                    try:
                        chunks = load_and_split(uploaded_file)
                    except Exception as e:
                        chunks = []
                        st.error(f" OCR Error: {str(e)}")

                if not chunks:
                    st.error(" Could not extract text from this PDF.")
                else:
                    with st.spinner(" Building vector index…"):
                        vs = create_vector_store(chunks)
                        st.session_state.vector_store = vs

                    with st.spinner(" Setting up RAG chain…"):
                        retriever = get_retriever(vs)
                        st.session_state.rag_chain = build_rag_chain(retriever)
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.chunk_count = len(chunks)
                        st.session_state.messages = []  # reset chat for new doc

    # Show status if a PDF is loaded
    if st.session_state.pdf_name:
        st.markdown(
            f"<div class='success-card'>"
            f" <strong>{st.session_state.pdf_name}</strong> ready<br>"
            f"<span style='font-size:0.85rem;'>{st.session_state.chunk_count} chunks indexed</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "<span style='color:#64748b;font-size:0.78rem;'>"
        "Powered by Mistral OCR · OpenAI · LangChain · FAISS"
        "</span>",
        unsafe_allow_html=True,
    )

# ── Main area ───────────────────────────────────────────────────────────
st.markdown('<p class="main-header"> PDF Q&A</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload a PDF on the left, then ask anything about it.</p>',
    unsafe_allow_html=True,
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDF…"):
    if st.session_state.rag_chain is None:
        st.warning(" Please upload a PDF first using the sidebar.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve and display chunks
        retriever = get_retriever(st.session_state.vector_store)
        retrieved_docs = retriever.invoke(prompt)

        with st.expander(f"📎 Retrieved Chunks ({len(retrieved_docs)})", expanded=False):
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"**Chunk {i + 1}:**")
                st.text(doc.page_content)
                if doc.metadata:
                    st.caption(f"Metadata: {doc.metadata}")
                st.divider()

        # Also print to terminal for debugging
        print(f"\n{'='*60}")
        print(f"Question: {prompt}")
        print(f"Retrieved {len(retrieved_docs)} chunks:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Chunk {i + 1} ---")
            print(doc.page_content)
            print(f"Metadata: {doc.metadata}")
        print(f"{'='*60}\n")

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = ask_question(st.session_state.rag_chain, prompt)
                except Exception as e:
                    answer = f" Error: {str(e)}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
