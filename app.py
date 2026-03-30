import os
from langchain_openai import ChatOpenAI
# from agent import search_documents
from langchain_community.chat_models import ChatOllama
from agent import get_vector_store
import streamlit as st
api_key = os.environ["OPENROUTER_API_KEY"]

local_Model = True

if local_Model:
    llm = ChatOllama(model="phi3", base_url="http://localhost:11434", temperature=0)
else:
    llm = ChatOpenAI(model="openrouter/free", base_url="https://openrouter.ai/api/v1",api_key=api_key)


def ask_question(question, docs=None):

    if docs:
        context = "\n\n".join(doc.page_content for doc in docs)
    else:
        print("Using vector DB...")
        vectordb = st.session_state.get("vectordb")
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
    You are an AI document analyst.
    RULES:
    - Answer ONLY from the context
    - If partially available, answer what you can
    - Say "Not found in document" only if completely missing


    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    response = llm.invoke(prompt)

    return response.content  