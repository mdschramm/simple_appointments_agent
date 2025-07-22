import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from healthcare_agent import HealthcareConversationAgent, ConversationStates
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

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
agent = HealthcareConversationAgent(llm_chat=llm)
thread_config = {"configurable": {"thread_id": "some_id"}}

# Async chat endpoint (single endpoint, resets session on end)
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Retrieve or initialize conversation state
    state = user_sessions.get(user_id)
    if not state:
        state = agent.get_initial_state()
        user_sessions[user_id] = state

        # Append user message
        state["messages"].append(HumanMessage(content=user_message))
        result = await agent.graph.ainvoke(state, config=thread_config)
        if "__interrupt__" in result:
            state = result["__interrupt__"][-1].value
        else:
            state = result
        if state["current_state"] == ConversationStates.END_CONVERSATION:
            del user_sessions[user_id]
            return ChatResponse(
                user_id=user_id,
                ai_messages=["It was nice talking to you!",],
                state=state["current_state"],
                end=True
            )
    else:
        # Append user message
        state["messages"].append(HumanMessage(content=user_message))
        state["session_metadata"]["last_message_read"] += 1
        # Resume the state machine from the last state
        result = await agent.graph.ainvoke(state, config=thread_config)
        if "__interrupt__" in result:
            state = result["__interrupt__"][-1].value
        else:
            state = result
        if state["current_state"] == ConversationStates.END_CONVERSATION:
            del user_sessions[user_id]
            return ChatResponse(
                user_id=user_id,
                ai_messages=["It was nice talking to you!",],
                state=state["current_state"],
                end=True
            )

    # Track unread AI messages
    last_message_read = state["session_metadata"].get("last_message_read", 0)
    new_ai_messages = [
        msg.content for msg in state["messages"][last_message_read:]
        if isinstance(msg, AIMessage)
    ]
    state["session_metadata"]["last_message_read"] = len(state["messages"])

    end = state["current_state"] == ConversationStates.END_CONVERSATION
    if end:
        del user_sessions[user_id]

    # Save user state
    user_sessions[user_id] = state
    return ChatResponse(
        user_id=user_id,
        ai_messages=new_ai_messages,
        state=state["current_state"],
        end=end
    )

@app.get("/")
def root():
    return {"status": "ok", "message": "Healthcare Agent FastAPI server running."}
