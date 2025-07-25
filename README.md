# Appointment Management Agent

This is a chat agent that can list, confirm, or cancel appointments.

## Running the Agent

The chat can currently be run in the `test_agent.py` file. After a language model and the `HealthcareConversationAgent` are initialized,
the user interact through a `while True` loop. The loop returns the new state of the conversation, including the AI's new responses in the
chat history. You will need an `OPENAI_API_KEY` or for slightly reduced chat intelligence `GROQ_API_KEY`

The chat can also be run as a server in `server.py` from the `/chat` endpoint. To run locally,
run `uvicorn server:app --reload`. Then make requests containing a user_id and message to the `/chat` endpoint.
Example Request:

```
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "message": "Hi how are you"}'
```

Response:

```json
{
  "user_id": "alice",
  "ai_messages": [
    "Hi! I'm here to help you with your healthcare appointments. You can ask me to list, confirm, or cancel your appointments. To get started, I'll need to verify your identity for security purposes. Could you please provide your full name, phone number, and date of birth?"
  ],
  "state": "verification",
  "end": false
}
```

This is also deployed on https://simple-appointments-agent.onrender.com - however it may be quite slow.

## Storage of User Data

Currently user data is hard-coded in memory.

## Overview of Design

The chat is designed as a state-machine using LangGraph containing 6 nodes:

```python
class ConversationStates(str, Enum):
    VERIFICATION = "verification"
    LIST_APPOINTMENTS = "list_appointments"
    CONFIRM_APPOINTMENT = "confirm_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    ERROR_RECOVERY = "error_recovery"
    END_CONVERSATION = "end_conversation"
```

Each node has a corresponding handler function that updates the state of the conversation:

```python
class ConversationState(TypedDict):
    """Complete conversation state with all necessary tracking"""
    messages: Annotated[List[BaseMessage], "conversation history"]
    last_user_message: Optional[HumanMessage]
    current_state: ConversationStates
    verified: bool
    patient_id: Optional[str]
    verification_attempts: int
    verification_data: Dict[str, Optional[str]]  # collected verification info
    pending_action: Optional[PendingActions]
    selected_appointment_id: Optional[str] # Use later for confirming before confirm/cancel
    appointments: List[Dict[str, Any]]
    last_error: Optional[str]
    session_metadata: Dict[str, Any]
```

To access and modify data to verify the user and perform actions, the chat uses tools in `tools.py` which in turn
call services in `services.py`.

## Agent Class

The chat and state machine flow are managed within the HealthcareConversationAgent class in `healthcare_agent.py`.

This class passes state around to each handler function and back to the user. Handler
functions have full control over the routing by setting "current_state" to the next state
based on it's handling of user input.

In this state machine:
1. The user initially enters a verification handling state
2. Here the user can also declare a pending action
```python
class PendingActions(str, Enum):
    LIST = "list"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    NONE = "none"
```
   which is queued for after the user is verified
 3. After verification the user is directed to the pending action, or if none has been declared, is prompted to declare their intent
 4. From here the user reaches the action states to list/confirm/cancel appointments
 5. After actions succeed, the user is prompted for future actions.
 6. If the user fails to verify after max attempts or there are data inconsistencies, the chat enters an error_recovery state

 * user verification loops until all of the necessary data is given from the user
 * confirm/cancel loops if the user intends to confirm/cancel but hasn't indicated which appointment they want to act on


## Technologies

- Langchain
- LangGraph
- Groq
- OpenAI

## Notes

Upgrading from groq to openai's gpt-4.1 improved the chat's handling of instructions and user input, leading me to
rely more on it than manual parsing of the user's input.

