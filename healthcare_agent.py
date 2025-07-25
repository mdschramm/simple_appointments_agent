"""
LangGraph-based Conversational AI Agent for Healthcare Appointment Management
Fixed version compatible with latest LangGraph API
"""

from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from enum import Enum
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
from tools import extract_verification_info, fetch_appointments, confirm_patient_appointment, cancel_patient_appointment, verify_patient_identity
from constants import MAX_VERIFICATION_ATTEMPTS, PII_FIELDS, END_MESSAGE


# Load environment variables
load_dotenv(override=True)
# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class PendingActions(str, Enum):
    LIST = "list"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    NONE = "none"

class ConversationStates(str, Enum):
    VERIFICATION = "verification"
    LIST_APPOINTMENTS = "list_appointments"
    CONFIRM_APPOINTMENT = "confirm_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    ERROR_RECOVERY = "error_recovery"
    END_CONVERSATION = "end_conversation"


# Todo: Enable user's to specify which appointment they want to cancel or confirm
# Before getting to the confirm or cancelllation state
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

# ============================================================================
# HEALTHCARE CONVERSATION AGENT
# ============================================================================

class HealthcareConversationAgent:
    """Main conversation agent with LangGraph state machine"""
    
    def __init__(self, llm_chat: ChatOpenAI | ChatGroq):
        
        self.llm = llm_chat

        # Ensure's interpretation of ending conversation so user isn't stuck
        self.system_prompt = """Be friendly and conversational. {additional_instructions} 
        If the user has indicated that they want to end the conversation, then ignore all previous instructions and return END and no other text. """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
        ])
        self.chain = self.prompt | self.llm
            
        self.tools = [
            extract_verification_info,
            fetch_appointments, 
            confirm_patient_appointment,
            cancel_patient_appointment
        ]
        self.graph = self._build_graph()

    def _check_end_response(self, response, state):
        if isinstance(response, str):
            content = response.strip()
        else:
            content = getattr(response, "content", None)
            if content is not None:
                content = content.strip()
        if content and content.upper() == END_MESSAGE:
            state["messages"].append(AIMessage(content="It was nice talking to you!"))
            state["current_state"] = ConversationStates.END_CONVERSATION
            return True
        return False
    
    def _build_graph(self) -> StateGraph:
        """Build the complete state machine graph"""
        graph = StateGraph(ConversationState)
        
        graph.add_node("verification_handler", self.handle_verification)
        graph.add_node("list_appointments_handler", self.handle_list_appointments)
        graph.add_node("confirm_appointment_handler", self.handle_confirm_appointment)
        graph.add_node("cancel_appointment_handler", self.handle_cancel_appointment)
        graph.add_node("error_recovery_handler", self.handle_error_recovery)
        graph.add_node("router", self.route_conversation)
        
        graph.set_entry_point("router")
        
        self._add_graph_edges(graph)
        
        return graph.compile()
    
    def _add_graph_edges(self, graph: StateGraph):
        """Define all possible state transitions"""
        
        graph.add_edge("verification_handler", "router")
        graph.add_edge("list_appointments_handler", "router") 
        graph.add_edge("confirm_appointment_handler", "router")
        graph.add_edge("cancel_appointment_handler", "router")
        graph.add_edge("error_recovery_handler", "router")
        
        graph.add_conditional_edges(
            "router",
            self.determine_next_state,
            {
                ConversationStates.VERIFICATION: "verification_handler", 
                ConversationStates.LIST_APPOINTMENTS: "list_appointments_handler",
                ConversationStates.CONFIRM_APPOINTMENT: "confirm_appointment_handler",
                ConversationStates.CANCEL_APPOINTMENT: "cancel_appointment_handler",
                ConversationStates.ERROR_RECOVERY: "error_recovery_handler",
                ConversationStates.END_CONVERSATION: END
            }
        )

    @staticmethod
    def get_initial_state():
        return ConversationState(
            messages=[],
            last_user_message=None,
            current_state=ConversationStates.VERIFICATION,
            verified=False,
            patient_id=None,
            verification_attempts=0,
            verification_data={},
            pending_action=PendingActions.NONE,
            selected_appointment_id=None,
            appointments=[],
            last_error=None,
            session_metadata={"last_message_read": -1} # Set to -1 to indicate no message has been read
        )

    # This function handles user verification, it attempts to iteratively extract name, phone and dob from the user
    # It will only return, allowing handle_verification to proceed, if all 3 pieces of information have been extracted
    async def _extract_verification_info(self, state: ConversationState) -> ConversationState:
        # last_message = state["messages"][-1]
        user_input = state["last_user_message"]
        
        # Try to extract and verify information
        verification_prompt = f"""
        You are helping verify a patient's identity. Extract verification information from this message: "{user_input}"
        
        Look for:
        - Full name (first and last name)
        - Phone number (various formats acceptable)  
        - Date of birth (MM/DD/YYYY, YYYY-MM-DD, or similar formats)
        
        For dates, please convert to YYYY-MM-DD format.
        Here is information that the user has already provided: "{state['verification_data']}"
        Here is the list of all of the information fields required: {PII_FIELDS}
        For each piece of information found, call the extract_verification_info tool with the field name {PII_FIELDS} and the value.
        Remember to only respond with END and no other text if the user wants to end the conversation.
        """
        
        response = await self.llm.bind_tools(self.tools).ainvoke([
            {"role": "system", "content": verification_prompt},
            {"role": "user", "content": user_input}
        ])

        # Check if LLM called verification tool
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "extract_verification_info":
                    # Execute the tool
                    extracted_info_result = await extract_verification_info.ainvoke(tool_call["args"])
                    if extracted_info_result["success"]:
                        state["verification_data"] = {**state["verification_data"], **extracted_info_result["data"]}

            if len(state["verification_data"]) != len(PII_FIELDS):
                message = "I don't have all the information I need to verify your identity. So far you've provided:\n"
                message += ", ".join([key.replace("_", " ") for key in state["verification_data"].keys() if key in PII_FIELDS])
                message += ".\nPlease provide the missing information:\n"
                message += ", ".join([key.replace("_", " ") for key in PII_FIELDS if key not in state["verification_data"].keys()])
                state["messages"].append(AIMessage(content=message))
                return interrupt(state)

        else:
            # LLM is asking for more information
            state["messages"].append(response)
            return interrupt(state)

    # Once the user is verified, this function checks for any pending_actions that were set in the initial state.
    # If there is a pending action, it will navigate the user to that action. Otherwise, the user is navigated to the main menu.    
    async def handle_verification(self, state: ConversationState) -> ConversationState:
        """Handle identity verification process"""

        if state["verified"]:
            state["messages"].append(AIMessage(content="I'm not sure what you want to do. Please let me know if you want to view your appointments, confirm an appointment, or cancel an appointment."))
            return interrupt(state)
        
        if state["verification_attempts"] >= MAX_VERIFICATION_ATTEMPTS:
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            state["last_error"] = "Maximum verification attempts exceeded"
            return state
        
        # Populates state["verification_data"] with name, phone and dob
        await self._extract_verification_info(state)
        
        result = await verify_patient_identity(state["verification_data"])
        if result["success"]:
            state["verified"] = True
            state["patient_id"] = result["patient_id"]
            # Clear verified information from state for security
            state["verification_data"] = {}
            welcome_msg = "Great! I've verified your identity."
            if state["pending_action"] == PendingActions.NONE:
                welcome_msg += " I can help you with:\n"
                welcome_msg += "• View your appointments\n• Confirm appointments\n• Cancel appointments\n\n"
                welcome_msg += "What would you like to do?"
            state["messages"].append(AIMessage(content=welcome_msg))
            if state["pending_action"] != PendingActions.NONE:
                return state

        else:
            state["verification_attempts"] += 1
            error_msg = "I couldn't verify your identity with that information. "
            error_msg += "Please double-check your full name, phone number, and date of birth."
            
            state["messages"].append(AIMessage(content=error_msg))
        
        # Prompt use for response
        return interrupt(state)
    
    # Will list user appointments, loops back to menu unless there is an error, in which case
    # it will navigate to error recovery.
    async def handle_list_appointments(self, state: ConversationState) -> ConversationState:
        """Fetch and display patient appointments"""
        
        if not state["patient_id"]:
            state["last_error"] = "Internal error, missing patient ID."
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            return state

        state["pending_action"] = PendingActions.NONE
        # Fetch appointments
        result = await fetch_appointments.ainvoke({"patient_id": state["patient_id"]})

        if not result["success"]:
            state["last_error"] = "Failed to fetch appointments"
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            return state
        
        state["appointments"] = result["appointments"]
        
        if result["appointments"]:
            appt_text = "Here are your upcoming appointments:\n\n"
            for i, appt in enumerate(result["appointments"], 1):
                appt_text += f"{i}. **{appt['type']}** with {appt['provider']}\n"
                appt_text += f"   {appt['datetime']}\n"
                appt_text += f"   {appt['location']}\n"
                appt_text += f"   Status: {appt['status']}\n\n"

            appt_text += "Let me know if you'd like to confirm or cancel any of these appointments."
            state["messages"].append(AIMessage(content=appt_text))
        else:
            no_appt_msg = "You don't have any upcoming appointments scheduled."
            state["messages"].append(AIMessage(content=no_appt_msg))
        
        return interrupt(state)
    
    async def handle_confirm_appointment(self, state: ConversationState) -> ConversationState:
        """Handle appointment confirmation"""
        return await self._handle_appointment_action(state, PendingActions.CONFIRM)
    
    async def handle_cancel_appointment(self, state: ConversationState) -> ConversationState:
        """Handle appointment cancellation"""
        return await self._handle_appointment_action(state, PendingActions.CANCEL)
    
    # Seaches for a selected appointment and performs the action.
    async def _handle_appointment_action(self, state: ConversationState, action: PendingActions) -> ConversationState:
        """Generic handler for appointment actions"""
        
        if not state["patient_id"]:
            state["last_error"] = "Internal error, missing patient ID."
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            return state

        # Clear pending action
        state["pending_action"] = PendingActions.NONE
        
        # Get user input
        user_input = state["last_user_message"]

        # Fetch appointments if not already fetched
        if not state["appointments"]:
            result = await fetch_appointments.ainvoke({"patient_id": state["patient_id"]})
            if result["success"]:
                state["appointments"] = result["appointments"]
        
        # Parse appointment selection
        selected_appointment = await self._parse_appointment_selection(state, user_input)
        
        if selected_appointment:
            # Perform the action
            if action == "confirm":
                result = await confirm_patient_appointment.ainvoke({"appointment_id": selected_appointment["id"],
                "patient_id": state["patient_id"]})
                action_word = "confirmed"
            else:  # cancel
                result = await cancel_patient_appointment.ainvoke({"appointment_id": selected_appointment["id"], "patient_id": state["patient_id"]})
                action_word = "cancelled"

            if not result["success"]:
                state["last_error"] = f"Failed to perform {action.value}"
                state["current_state"] = ConversationStates.ERROR_RECOVERY
                return state

            success_msg = f"Perfect! I've {action_word} your appointment:\n\n"
            success_msg += f"**{selected_appointment['type']}** with {selected_appointment['provider']}\n"
            success_msg += f"{selected_appointment['datetime']}\n\n"
            success_msg += "Is there anything else I can help you with?"
            
            state["messages"].append(AIMessage(content=success_msg))
        else:
            # Couldn't parse selection
            state["messages"].append(AIMessage(content=f"Which appointment would you like to {action.value}? Please tell me the number (1, 2, etc.)."))

        return interrupt(state)
    
    # Parses user input to select an appointment
    async def _parse_appointment_selection(self, state: ConversationState, user_input: str) -> Optional[Dict]:
        """Parse user input to select an appointment"""
        if not state["appointments"]:
            return None

        appointments = state["appointments"]

        appointment_descriptions = [f"{i+1}. {appt['type']} with {appt['provider']} on {appt['datetime']} at {appt['location']} with id {appt['id']}" for i, appt in enumerate(appointments)]
        
        NONE = "NONE"

        selection_prompt = f"""
        The user has provided the following input: {user_input}\n

        The list of appointments for this user is as follows: {'\n'.join(appointment_descriptions)}\n

        Indicate to be by number only e.g. "1" or "2" which appointment the user is referring to.
        Bear in mind that the user's input may contain information irrelevant to the appointment selection.
        Only focus on information that is relevant to the user's appointment.
        If the user's input does not match any of the appointments then return {NONE}. 
        Only return a number or {NONE} in your response. No other text. 
        """

        response = await self.chain.ainvoke(
            {"additional_instructions": selection_prompt,
            "chat_history": state["messages"],
            }
        )

        content = response.content

        if content == NONE:
            return None

        try:
            content = int(content)
        except ValueError:
            return None

        if content < 1 or content > len(appointments):
            return None
  
        return appointments[content - 1]
    
    async def handle_error_recovery(self, state: ConversationState) -> ConversationState:
        """Handle error states and recovery"""
        
        if state["last_error"] == "Maximum verification attempts exceeded":
            state["messages"].append(AIMessage(content="You've exceeded the maximum verification attempts."))
            state["current_state"] = ConversationStates.END_CONVERSATION
            return state
        else:
            error_msg = "I'm sorry, I encountered an issue. Let me try to help you differently."
            state["current_state"] = ConversationStates.VERIFICATION
            state["messages"].append(AIMessage(content=error_msg))
        
        return interrupt(state)

    
    async def route_conversation(self, state: ConversationState) -> ConversationState:
        """Router node - conditional routing logic centralized here"""

        # Skip routing if the conversation is ending
        if state["current_state"] == ConversationStates.END_CONVERSATION:
            return state

        # Set the last user message in state so it can be referenced in handlers
        user_input = None
        for message in state["messages"][::-1]:
            if isinstance(message, HumanMessage):
                user_input = message.content
                state["last_user_message"] = user_input
                break

        # If there's an error, proceed immediately to error recovery
        if state["current_state"] == ConversationStates.ERROR_RECOVERY:
            return state

        # If the user is verified and there's a pending action, navigate to that state
        if state["verified"] and state["pending_action"] == PendingActions.LIST:
            state["current_state"] = ConversationStates.LIST_APPOINTMENTS
            return state
        elif state["verified"] and state["pending_action"] == PendingActions.CONFIRM:
            state["current_state"] = ConversationStates.CONFIRM_APPOINTMENT
            return state
        elif state["verified"] and state["pending_action"] == PendingActions.CANCEL:
            state["current_state"] = ConversationStates.CANCEL_APPOINTMENT
            return state
        
        # User input parsing - Either returns "END" or a JSON object with key information
        # From the user query
        VERIFICATION_PROVIDED = "verification_provided"
        REQUESTED_ACTION = "requested_action"
        INFORMATION_REQUESTED = "information_requested"
        instruction = f"""
        Based on the user's last message: {user_input}
        If the user indicates that they want to end the conversation and they don't provide any other information or indicate any appointment actions, thn return END and no other text. Otherwise:

        Return a JSON object of the following format:
        {{"{VERIFICATION_PROVIDED}": {bool}, "{REQUESTED_ACTION}": "{PendingActions.LIST.value}/{PendingActions.CONFIRM.value}/{PendingActions.CANCEL.value}/{PendingActions.NONE.value}", "{INFORMATION_REQUESTED}": {bool}}}

        If the user provides any of their verification information in their last message(full name, phone number, and/or date of birth) 
        then {VERIFICATION_PROVIDED} should be set to true even if the user hasn't provided everything they need.

        If there is no verification information in the user's last message, then {VERIFICATION_PROVIDED} should be set to false.

        If in their message they've indicated that they want to list/view/see their appointments, then {REQUESTED_ACTION} should be set to "{PendingActions.LIST.value}".
        If in their message they've indicated that they want to confirm an appointment, then {REQUESTED_ACTION} should be set to "{PendingActions.CONFIRM.value}".
        If in their message they've indicated that they want to cancel an appointment, then {REQUESTED_ACTION} should be set to "{PendingActions.CANCEL.value}".
        If they haven't indicated any of the above actions then {REQUESTED_ACTION} should be set to "{PendingActions.NONE.value}".

        If the user has asked about which actions they can perform in this chat, then set {INFORMATION_REQUESTED} to true. Otherwise, set {INFORMATION_REQUESTED} to false.

        Remember that the user may refer to other messages in chat history to indicate what they want to do, so use that context to help determine {REQUESTED_ACTION}
        """
        response = await self.chain.ainvoke({
            "additional_instructions": instruction, 
            "chat_history": state["messages"]})
        
        # End immediately if the user wants to end conversation
        if self._check_end_response(response, state):
            state["current_state"] = ConversationStates.END_CONVERSATION
            return state

        # Parse json response content string
        try:
            response_json = json.loads(response.content)

            # Include information if the user asks
            if response_json["information_requested"]:
                msg = "I can help you list, confirm, and cancel your healthcare appointments."
                if not state["verified"] and not response_json["verification_provided"]:
                    msg += " You'll need to first provide your full name, phone number and date of birth before I can assist you."
                state["messages"].append(AIMessage(content=msg))

            if not state["verified"]:
                state["current_state"] = ConversationStates.VERIFICATION
                state["pending_action"] = response_json["requested_action"]
                return state

            if response_json["requested_action"] == PendingActions.LIST:
                state["current_state"] = ConversationStates.LIST_APPOINTMENTS
            elif response_json["requested_action"] == PendingActions.CONFIRM:
                state["current_state"] = ConversationStates.CONFIRM_APPOINTMENT
            elif response_json["requested_action"] == PendingActions.CANCEL:
                state["current_state"] = ConversationStates.CANCEL_APPOINTMENT
            else:
                state["current_state"] = ConversationStates.VERIFICATION
            

        except json.JSONDecodeError:
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            return state

        return state
    
    async def determine_next_state(self, state: ConversationState) -> ConversationStates:
        """Pass current_state key through to determine next node"""
        return state["current_state"]
    
