import os
from langchain_openai import ChatOpenAI
# from agent import search_documents
from langchain_community.chat_models import ChatOllama
from agent import get_vector_store
import streamlit as st
api_key = os.environ["OPENROUTER_API_KEY"]

local_Model = False

if local_Model:
    llm = ChatOllama(model="phi3", base_url="http://localhost:11434", temperature=0)
else:
    llm = ChatOpenAI(model="openrouter/free", base_url="https://openrouter.ai/api/v1",api_key=api_key)



def ask_question(question):
    try:
        vectordb = st.session_state.get("vectordb", None)
        if vectordb is None:
            return "Please upload and process a document first."

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        
        if not docs:
            return "No relevant content found in the document."

        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"""
                You are an AI document analyst.

                STRICT RULES:
                - Answer ONLY from the provided context
                - If answer is not found, say "Not found in document"

                Context:
                {context}

                Question:
                {question}

                Answer:"""

        response = llm.invoke(prompt)
        return response.content
    
    except Exception as e:
        return f"Error processing question: {str(e)}"

