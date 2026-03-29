from dotenv import load_dotenv
import os
load_dotenv()
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from chromadb.config import Settings
import chromadb, shutil, streamlit as st


def get_client():
    # return chromadb.PersistentClient(path="./chroma_db")
    return chromadb.Client()


def ingest_documents(docs, reset_db=False):

    if reset_db:
        if "vectordb" in st.session_state:
            del st.session_state["vectordb"]
        shutil.rmtree("./chroma_db", ignore_errors=True)

    client = get_client()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectordb = get_vector_store(client)
    vectordb.add_documents(chunks)
    return vectordb


def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


def get_vector_store(client):
    embeddings = get_embedding_model()
    return Chroma(
        client=client,
        collection_name="docs",
        embedding_function=embeddings,persist_directory="./chroma_db")


def get_retriever():
    vectordb = st.session_state.get("vectordb", None)
    if vectordb is None:
        return None
    return vectordb.as_retriever(search_kwargs={"k": 3})


@tool
def search_documents(query: str):
    """Search the uploaded document text to answer user questions."""
    retriever = get_retriever()
    docs = retriever.invoke(query)

    return "\n\n".join(doc.page_content[:500] for doc in docs)


