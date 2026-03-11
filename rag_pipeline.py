from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_store import get_vector_store

def ingest_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    vectordb = get_vector_store()

    vectordb.add_documents(chunks)

    # vectordb.persist()

def get_retriever():

    vectordb = get_vector_store()

    retriever = vectordb.as_retriever(search_kwargs={"k":3})

    return retriever