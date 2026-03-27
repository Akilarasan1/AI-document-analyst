from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain import hub
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
import chromadb

client = chromadb.Client(
    Settings(anonymized_telemetry=False)
)

# api_key = os.environ["OPENROUTER_API_KEY"]

# def create_document_agent():

#     llm = ChatOpenAI(
#         model="openrouter/free",
#         base_url="https://openrouter.ai/api/v1",
#         api_key=api_key
#     )
#     tools = [search_documents]


def ingest_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    vectordb = get_vector_store()

    # ✅ Delete the entire collection safely
    vectordb.delete_collection()

    # recreate the vector store
    vectordb = get_vector_store()

    vectordb.add_documents(chunks)


def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def get_vector_store():
    embeddings = get_embedding_model()

    vectordb = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    return vectordb


def get_retriever():
    vectordb = get_vector_store()
    # retriever = vectordb.as_retriever(search_kwargs={"k":3})
    retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3})

    return retriever


@tool
def search_documents(query: str):
    """Search the uploaded document text to answer user questions."""
    retriever = get_retriever()
    docs = retriever.invoke(query)

    return "\n\n".join(doc.page_content[:500] for doc in docs)



# api_key = os.environ["OLLAMA_API_KEY"]

# def create_document_agent():
#     llm = ChatOllama(
#     model="phi3",
#     base_url="http://localhost:11434",
#     temperature=0)

#     tools = [search_documents]

#     prompt = hub.pull("hwchase17/react")

#     agent = create_react_agent(llm, tools, prompt)

#     agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     handle_parsing_errors=True,
#     max_iterations=3)


#     return agent_executor