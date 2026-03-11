from vector_store import get_vector_store

def get_retriever():
    vectordb = get_vector_store()
    return vectordb.as_retriever(search_kwargs={"k": 3})