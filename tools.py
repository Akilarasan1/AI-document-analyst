from langchain_core.tools import tool
from retriever import get_retriever

retriever = get_retriever()

@tool
def search_documents(query: str):
    """Search documents in the vector database."""

    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])