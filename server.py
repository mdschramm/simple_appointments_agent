import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from healthcare_agent import HealthcareConversationAgent, ConversationStates, ConversationState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.responses import HTMLResponse
import uuid

load_dotenv()

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conversation state store (in-memory, for demo)
user_sessions = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    user_id: str
    ai_messages: list[str]
    state: str
    end: bool

# LLM selection logic (mirrors test_agent.py)
def get_chat_model():
    open_ai_api_key = os.environ.get("OPENAI_API_KEY")
    if not open_ai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found")
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
        max_retries=2,
        openai_api_key=open_ai_api_key
    )
    return llm

# Create agent instance (stateless, but can reuse chain)
llm = get_chat_model()

async def update_state(state: ConversationState, user_message: str, agent: HealthcareConversationAgent) -> ConversationState:
    # Append user message
    state["messages"].append(HumanMessage(content=user_message))
    state["session_metadata"]["last_message_read"] += 1
    
    # Resume the state machine from the last state
    # result = await agent.graph.ainvoke(state, config=thread_config)
    result = await agent.graph.ainvoke(state)
    if "__interrupt__" in result:
        state = result["__interrupt__"][-1].value
    else:
        state = result
    return state

# Async chat endpoint (single endpoint, resets session on end)
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    END_MESSAGE = "It was good chatting!"
    # Generate a random user ID if none provided
    user_id = request.user_id
    user_message = request.message

    # Retrieve or initialize conversation state
    user_session = user_sessions.get(user_id)
    if not user_session:
        # initialize user session
        user_id = uuid.uuid4().hex
        user_state = HealthcareConversationAgent.get_initial_state()
        user_state["session_metadata"]["last_message_read"] += 1
        agent = HealthcareConversationAgent(llm_chat=llm)
        user_session = {"state": user_state, "agent": agent}
        user_sessions[user_id] = user_session

        # Append user message
        user_state = await update_state(user_state, user_message, agent)
        user_session["state"] = user_state
        if user_state["current_state"] == ConversationStates.END_CONVERSATION:
            del user_sessions[user_id]
            return ChatResponse(
                user_id=user_id,
                ai_messages=[END_MESSAGE],
                state=user_state["current_state"],
                end=True
            )
    else:
        user_state = user_session["state"]
        agent = user_session["agent"]
        # Append user message
        user_state = await update_state(user_state, user_message, agent)
        user_session["state"] = user_state
        
        if user_state["current_state"] == ConversationStates.END_CONVERSATION:
            del user_sessions[user_id]
            return ChatResponse(
                user_id=user_id,
                ai_messages=[END_MESSAGE],
                state=user_state["current_state"],
                end=True
            )

    # Track unread AI messages
    FALLBACK_MESSAGE_INDEX = 0
    last_message_read = user_state["session_metadata"].get("last_message_read", FALLBACK_MESSAGE_INDEX)
    new_ai_messages = [
        msg.content for msg in user_state["messages"][last_message_read:]
        if isinstance(msg, AIMessage)
    ]
    user_state["session_metadata"]["last_message_read"] = len(user_state["messages"]) - 1

    end = user_state["current_state"] == ConversationStates.END_CONVERSATION
    if end:
        del user_sessions[user_id]

    return ChatResponse(
        user_id=user_id,
        ai_messages=new_ai_messages,
        state=user_state["current_state"],
        end=end
    )

@app.get("/")
def root():
    return {"status": "ok", "message": "Healthcare Agent FastAPI server running."}


# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML page
@app.get("/chat-ui", response_class=HTMLResponse)
def chat_ui():
    html_path = Path("static/index.html")
    return HTMLResponse(content=html_path.read_text(), status_code=200)
