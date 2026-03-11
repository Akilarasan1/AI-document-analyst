from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain import hub


from tools import search_documents

def create_document_agent():

    llm = ChatOpenAI(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )
    tools = [search_documents]

    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor