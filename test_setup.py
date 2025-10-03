# test_setup.py
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

# Test 1: Env loaded?
if not os.getenv("OPENAI_API_KEY"):
    print("❌ FAILED: OpenAI key missing in .env")
    exit(1)

# Test 2: Can we talk to OpenAI?
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke("Say 'AI Forge Ready!' in 3 words.")
    print("✅ SUCCESS:", response.content.strip())
except Exception as e:
    print("❌ FAILED:", str(e))