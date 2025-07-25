#!/usr/bin/env python3
import os
import asyncio
from dotenv import load_dotenv
from healthcare_agent import HealthcareConversationAgent, ConversationState, ConversationStates
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def get_chat_model():
    # Use Groq's native integration instead of OpenAI compatibility
    # groq_api_key = os.environ.get("GROQ_API_KEY")
    # if not groq_api_key:
    #     raise ValueError("GROQ_API_KEY environment variable not found")
    
    # print(f"Using Groq API key: {groq_api_key[:5]}...{groq_api_key[-5:] if groq_api_key else None}")

    # llm = ChatGroq(
    #     model=llm_model,
    #     temperature=0.1,
    #     max_retries=2,
    #     groq_api_key=groq_api_key  # Explicitly pass the API key
    # )
    
    open_ai_api_key = os.environ.get("OPENAI_API_KEY")
    if not open_ai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not found")
    print(f"Using OpenAI API key: {open_ai_api_key[:5]}...{open_ai_api_key[-5:] if open_ai_api_key else None}")
    
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
        max_retries=2,
        openai_api_key=open_ai_api_key
    )
    return llm

async def interactive_chat():
    llm = get_chat_model()
    agent = HealthcareConversationAgent(llm)
    thread_config = {"configurable": {"thread_id": "some_id"}}
    state = None
    print("🏥 Healthcare Agent Test")
    print("=" * 30)
    state = agent.get_initial_state()
    state["messages"].append(AIMessage(content="""Hello! I'm an appointment management assistant that can help you
    list, confirm, and cancel your healthcare appointments. You'll first need to provide your full name, phone number, and date of
    birth to verify your identity before I can assist you."""))
    print(f"Agent: {state['messages'][0].content}")
    state["session_metadata"]["last_message_read"] += 1
    while True:
        user_input = input("You: ")
        state["messages"].append(HumanMessage(content=user_input))
        state["session_metadata"]["last_message_read"] += 1
        # Resume the state machine from the last state
        result = await agent.graph.ainvoke(state, config=thread_config)
        if "__interrupt__" in result:
            state = result["__interrupt__"][-1].value
        else:
            state = result
        # Print the latest AI message
        last_message_read = state["session_metadata"]["last_message_read"]
        for i in range(last_message_read, len(state["messages"])):
            if isinstance(state["messages"][i], AIMessage):
                ai_message = state["messages"][i].content
                print(f"Agent: {ai_message}")

        state["session_metadata"]["last_message_read"] = len(state["messages"]) - 1
        if state["current_state"] == ConversationStates.END_CONVERSATION:
            break

if __name__ == "__main__":
    asyncio.run(interactive_chat())