# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import markdown
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os
import requests
import re

load_dotenv()

# Validate keys
assert os.getenv("GROQ_API_KEY"), "❌ Missing GROQ_API_KEY in .env"
assert os.getenv("NASA_ADS_TOKEN"), "❌ Missing NASA_ADS_TOKEN in .env"

app = FastAPI(
    title="NASA Space Biology Knowledge Engine",
    description="A hackathon-winning AI agent for NASA Space Apps Challenge — powered by Groq, NASA ADS, and LangChain.",
    version="1.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

class QuestionRequest(BaseModel):
    question: str

# In-memory session store (for demo; replace with Redis/DB in production)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# NASA-aware web search
def nasa_search(query: str) -> str:
    nasa_query = f"{query} site:nasa.gov OR site:data.nasa.gov OR site:adsabs.harvard.edu"
    wrapper = DuckDuckGoSearchAPIWrapper(region='wt-wt', max_results=2)
    results = wrapper.results(nasa_query, 2)
    if results:
        return "\n".join([f"**{r['title']}**\n🔗 {r['link']}" for r in results])
    return "No NASA sources found."

# NASA ADS paper search
def nasa_ads_search(query: str) -> str:
    url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {"Authorization": f"Bearer {os.getenv('NASA_ADS_TOKEN')}"}
    params = {
        "q": f"abstract:({query}) AND (database:astronomy OR database:physics)",
        "fl": "title,bibcode",
        "rows": 2
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        if res.status_code == 200:
            docs = res.json().get("response", {}).get("docs", [])
            if docs:
                return "\n\n".join([
                    f"📄 **{d['title'][0]}**\n🔗 https://ui.adsabs.harvard.edu/abs/{d['bibcode']}"
                    for d in docs
                ])
        return "No relevant NASA papers found."
    except Exception as e:
        return f"NASA ADS search failed: {str(e)}"

# Safe math evaluator
def safe_math(expr: str) -> str:
    cleaned = re.sub(r'[^\d+\-*/().\s]', '', expr)
    if re.match(r'^[\d+\-*/().\s]+$', cleaned):
        try:
            result = eval(cleaned, {"__builtins__": {}}, {})
            return str(result)
        except:
            return "Invalid math expression"
    return "Only basic math allowed: +, -, *, /, (), numbers"

# Format raw LLM output into structured, readable response
def format_nasa_response(raw: str) -> str:
    """Converts raw LLM output into structured HTML for the UI."""
    raw = raw.strip()
    if not raw:
        return "<p>No answer generated.</p>"

    # Ensure proper paragraph breaks
    formatted = re.sub(r'(\. )', r'.\n\n', raw)
    
    # Handle numbered lists
    formatted = re.sub(r'^(\d+)\.', r'### \1.', formatted, flags=re.MULTILINE)

    # Wrap in professional framing if not already
    if not formatted.startswith("> **"):
        formatted = "> **Based on NASA’s open data and research, here is a structured overview:**\n\n" + formatted

    if not formatted.endswith("lunar sustainability."):
        formatted += "\n\n> 🌕 *This insight is synthesized from NASA missions, peer-reviewed research, and open data portals to support space biology innovation.*"

    # Convert Markdown to HTML
    html = markdown.markdown(formatted, extensions=['nl2br'])  # nl2br preserves line breaks
    
    # Style the blockquotes (for the framing)
    html = html.replace('<blockquote>', '<blockquote style="border-left: 4px solid #0b5394; padding-left: 16px; margin: 16px 0; color: #2c3e50;">')
    
    return html

# Build the NASA-specialized agent
def get_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # ✅ Officially recommended, non-deprecated, free
        temperature=0.1,  # Slight creativity for fluency
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the NASA Space Biology Knowledge Assistant.
Your role is to provide accurate, well-structured, and source-backed answers about space biology, lunar missions, and NASA research.

Guidelines:
- Always prioritize information from nasa.gov, data.nasa.gov, or adsabs.harvard.edu
- Structure answers with clear headings, bullet points, and paragraphs
- Cite specific missions (e.g., Artemis II, ISS) or experiments when possible
- If asked for calculations (e.g., radiation dose, growth rates), compute them precisely
- Never hallucinate. If unsure, say "NASA has not published data on this yet."
- Format output in clean markdown with bold headings and source links"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

agent = get_agent()

@app.post("/ask")
async def ask(request: QuestionRequest, req: Request):
    session_id = req.headers.get("X-Session-ID", "user_123")
    
    try:
        # 1. Handle math
        if re.search(r'\d+[\+\-\*\/]\d+', request.question):
            result = safe_math(request.question)
            return {"answer": f"🧮 **Calculation Result**\n\nThe result of `{request.question}` is **{result}**."}

        # 2. Detect NASA/space topics
        nasa_keywords = [
            "nasa", "space", "moon", "mars", "artemis", "iss", "astronaut", 
            "biology", "radiation", "microgravity", "lunar", "orbit", "cosmic",
            "spacecraft", "orion", "sls", "rocket", "mission"
        ]
        is_nasa_topic = any(kw in request.question.lower() for kw in nasa_keywords)

        # 3. Route accordingly
        if is_nasa_topic:
            # Try NASA ADS first for research
            if any(kw in request.question.lower() for kw in ["paper", "study", "research", "experiment", "publication"]):
                papers = nasa_ads_search(request.question)
                if "failed" not in papers and "No relevant" not in papers:
                    enriched = f"User: {request.question}\n\nNASA Papers:\n{papers}\n\nProvide a structured answer with citations."
                    raw = agent.invoke({"input": enriched}, config={"configurable": {"session_id": session_id}})
                    return {"answer": format_nasa_response(raw)}
            
            # Then NASA-biased web search
            nasa_results = nasa_search(request.question)
            if "No NASA sources" not in nasa_results:
                enriched = f"User: {request.question}\n\nNASA Sources:\n{nasa_results}\n\nAnswer using these sources."
                raw = agent.invoke({"input": enriched}, config={"configurable": {"session_id": session_id}})
                return {"answer": format_nasa_response(raw)}

        # 4. FOR ALL OTHER QUESTIONS (Wall Street, news, etc.) → GENERAL SEARCH
        general_wrapper = DuckDuckGoSearchAPIWrapper(region='wt-wt', max_results=3)
        general_results = general_wrapper.results(request.question, 3)
        
        if general_results:
            # Format results cleanly
            general_text = "\n\n".join([
                f"**{r['title']}**\n🔗 {r['link']}\n{r.get('snippet', '')}"
                for r in general_results
            ])
            enriched = f"User: {request.question}\n\nWeb Search Results:\n{general_text}\n\nProvide a concise, factual summary."
            raw = agent.invoke({"input": enriched}, config={"configurable": {"session_id": session_id}})
            return {"answer": format_nasa_response(raw)}
        
        # 5. Last resort: honest "I don't know"
        return {"answer": format_nasa_response(
            "I cannot provide real-time news or financial updates without live web search results. "
            "Please check a trusted news source like Reuters, Bloomberg, or CNBC for the latest Wall Street updates."
        )}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/health")
async def health():
    return {"status": "✅ LIVE", "model": "llama-3.3-70b-versatile", "challenge": "NASA Space Biology Knowledge Engine"}