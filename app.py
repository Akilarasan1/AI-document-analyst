from agent import create_document_agent

agent = create_document_agent()

def ask_question(question):

    response = agent.invoke({
    "input": question})

    return response