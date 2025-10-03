# test_ollama.py
from langchain_community.chat_models import ChatOllama

# Make sure Ollama app is RUNNING in background!
llm = ChatOllama(model="llama3", temperature=0)

response = llm.invoke("Say 'Local AI Forge Ready!' in 4 words.")
print("✅ SUCCESS:", response.content.strip())