from agent import create_document_agent

# llm = create_document_agent()
from langchain_community.chat_models import ChatOllama


from agent import get_vector_store

llm = ChatOllama(
    model="phi3",
    base_url="http://localhost:11434",
    temperature=0
)




def ask_question(question):

    vectordb = get_vector_store()

    retriever = vectordb.as_retriever(search_kwargs={"k":3})

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    print("Retrieved context:\n", context)

    prompt = f"""
        You are an AI document analyst.

        Use the context below to answer the question.

        Context:
        {context}

        Question:
        {question}

        Answer clearly based only on the document.
        """

    response = llm.invoke(prompt)

    return response.content