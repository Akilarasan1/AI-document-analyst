from langchain_chroma import Chroma
from embeddings import get_embedding_model

def get_vector_store():
    embeddings = get_embedding_model()

    vectordb = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    return vectordb