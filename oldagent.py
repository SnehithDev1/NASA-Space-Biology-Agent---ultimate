# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv
import os
import re

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ Missing GROQ_API_KEY in .env")

app = FastAPI(title="MCP Server - Free & Stable (Oct 2025)")

class QuestionRequest(BaseModel):
    question: str

# Global memory (for demo)
chat_history = []

# Tool functions
def search_web(query: str) -> str:
    wrapper = DuckDuckGoSearchAPIWrapper(region='wt-wt', max_results=3)
    return str(wrapper.results(query, 3))

def calculate_math(expr: str) -> str:
    # Safe math evaluation (only numbers and operators)
    if re.match(r'^[\d+\-*/().\s]+$', expr):
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)
        except:
            return "Invalid math expression"
    return "Only basic math allowed: +, -, *, /, (), numbers"

# ReAct-style agent
def react_agent(question: str, history: list) -> str:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # ✅ Current, non-deprecated, free
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Build history string
    history_str = "\n".join([f"User: {h.content}" if isinstance(h, HumanMessage) else f"AI: {h.content}" 
                             for h in history])
    
    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the following tools when needed:
- To search the web: respond with "SEARCH: <query>"
- To calculate math: respond with "CALC: <expression>"

Otherwise, answer directly.

Conversation history:
{history}

User: {input}
AI:""")
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"input": question, "history": history_str})
    
    # Parse tool calls
    response = response.strip()
    if response.startswith("SEARCH:"):
        query = response[7:].strip()
        result = search_web(query)
        # Re-ask with result
        followup = f"User: {question}\nTool Result: {result}\nAI:"
        final = chain.invoke({"input": followup, "history": history_str})
        return final.strip()
    elif response.startswith("CALC:"):
        expr = response[5:].strip()
        result = calculate_math(expr)
        return f"The result is {result}."
    else:
        return response

@app.post("/ask")
async def ask(request: QuestionRequest):
    global chat_history
    
    try:
        answer = react_agent(request.question, chat_history)
        
        # Update memory
        chat_history.append(HumanMessage(content=request.question))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
            
        return {"answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/")
async def health():
    return {"status": "✅ MCP Server LIVE", "model": "llama-3.3-70b-versatile (Groq)"}

# Needed for memory
from langchain_core.messages import HumanMessage, AIMessage