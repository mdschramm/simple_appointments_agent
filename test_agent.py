#!/usr/bin/env python3
"""
Quick test script for the Healthcare Conversation Agent
Run this after installing the minimal requirements to test the agent.
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key if not already set
if not os.getenv("OPENAI_API_KEY"):
    api_key = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key

# Import the agent (assumes the main code is saved as healthcare_agent.py)
try:
    from healthcare_agent import HealthcareConversationAgent
except ImportError:
    print("Error: Make sure the healthcare agent code is saved as 'healthcare_agent.py'")
    exit(1)

async def interactive_test():
    """Interactive test of the healthcare agent"""
    print("🏥 Healthcare Conversation Agent Test")
    print("=" * 50)
    print("Test patient credentials:")
    print("- Name: John Doe")
    print("- Phone: 555-0123") 
    print("- DOB: 1990-01-01")
    print("=" * 50)
    
    # Initialize agent
    try:
        agent = HealthcareConversationAgent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Start conversation
    state = None
    print("\n💬 Starting conversation...")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            # Get user input
            if state is None:
                user_input = "Hello, I need help with my appointments"
                print(f"👤 User: {user_input}")
            else:
                user_input = input("👤 User: ")
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
            
            # Process message
            state = await agent.process_message(user_input, state)
            
            # Display agent response
            last_message = state["messages"][-1]
            print(f"🤖 Agent: {last_message.content}\n")
            
            # Display current state for debugging
            print(f"🔍 State: {state['current_state']}, Verified: {state['verified']}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

async def automated_test():
    """Automated test with predefined conversation flow"""
    print("🤖 Running automated test...")
    
    agent = HealthcareConversationAgent()
    
    # Test conversation flow
    test_messages = [
        "Hi, I need to check my appointments",
        "My name is John Doe, phone is 555-0123, and I was born on January 1st, 1990", 
        "I'd like to see my appointments",
        "I want to confirm the first appointment",
        "Thank you!"
    ]
    
    state = None
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Step {i} ---")
        print(f"👤 User: {message}")
        
        state = await agent.process_message(message, state)
        
        last_message = state["messages"][-1]
        print(f"🤖 Agent: {last_message.content}")
        print(f"🔍 State: {state['current_state']}, Verified: {state['verified']}")

def main():
    """Main function to run tests"""
    print("Choose test mode:")
    print("1. Interactive test (chat with the agent)")
    print("2. Automated test (predefined conversation)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(interactive_test())
    elif choice == "2":
        asyncio.run(automated_test())
    else:
        print("Invalid choice. Running interactive test...")
        asyncio.run(interactive_test())

if __name__ == "__main__":
    main()