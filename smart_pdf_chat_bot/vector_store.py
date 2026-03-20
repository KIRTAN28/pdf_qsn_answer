"""
Vector Store Module
Creates and manages the FAISS vector store for document embeddings.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):
    """
    Embed document chunks and build an in-memory FAISS vector store.

    Args:
        chunks: List of LangChain Document objects.

    Returns:
        FAISS vector store instance.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_retriever(vector_store, k=8):
    """
    Create a retriever from the vector store.

    Args:
        vector_store: FAISS vector store instance.
        k: Number of documents to retrieve per query.

    Returns:
        LangChain retriever object.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
