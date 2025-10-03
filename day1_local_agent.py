# day1_local_agent.py
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ✅ Use llama3.1 (better for agents)
llm = ChatOllama(
    model="llama3.1:8b",      # ← NEW MODEL
    temperature=0.0,
    base_url="http://localhost:11434"
)

# Tool
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# ✅ PROMPT WITHOUT MEMORY (avoids chat_history error)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided tools to answer questions accurately."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")  # ← Only scratchpad needed
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# ✅ NO MEMORY for now → avoids deprecation + input errors
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Test
if __name__ == "__main__":
    question = "Tell about nasa space biology knowledge center in short?"
    print(f"❓ Question: {question}\n")
    response = agent_executor.invoke({"input": question})
    print(f"\n✅ Final Answer: {response['output']}")