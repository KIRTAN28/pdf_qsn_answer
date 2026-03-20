"""
RAG Chain Module
Builds the retrieval-augmented generation chain for question answering.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def _format_docs(docs):
    """Join retrieved document contents into a single string."""
    formatted = "\n\n".join(doc.page_content for doc in docs)
    print(f"[RAG Chain] Retrieved {len(docs)} docs, total context length: {len(formatted)} chars")
    if docs:
        print(f"[RAG Chain] First doc preview: {docs[0].page_content[:200]}...")
    return formatted


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context from a PDF document.

Instructions:
- Carefully read and understand the given context.
- Answer the question strictly using ONLY the information present in the context.
- Do not use any external knowledge or make assumptions beyond the provided text.

context:
{context}
"""


def build_rag_chain(retriever):
    """
    Build a RAG chain: retriever → prompt → LLM → output parser.

    Args:
        retriever: LangChain retriever object.

    Returns:
        Runnable RAG chain.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask_question(chain, question):
    """
    Invoke the RAG chain with a question.

    Args:
        chain: The RAG chain.
        question: User's question string.

    Returns:
        Answer string.
    """
    return chain.invoke(question)
