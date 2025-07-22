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
    UNSURE = "unsure"

class ConversationStates(str, Enum):
    INITIAL = "initial"
    VERIFICATION = "verification"
    AUTHENTICATED = "authenticated"
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

        # There's an issue with the parsing of system prompt so I can't interpolate END_MESSAGE here
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
        # Helper to check if the LLM signaled to end the conversation
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
        
        # Add all nodes
        graph.add_node("initial_handler", self.handle_initial)
        graph.add_node("verification_handler", self.handle_verification)
        graph.add_node("authenticated_handler", self.handle_authenticated)
        graph.add_node("list_appointments_handler", self.handle_list_appointments)
        graph.add_node("confirm_appointment_handler", self.handle_confirm_appointment)
        graph.add_node("cancel_appointment_handler", self.handle_cancel_appointment)
        graph.add_node("error_recovery_handler", self.handle_error_recovery)
        graph.add_node("router", self.route_conversation)
        
        # Set entry point
        graph.set_entry_point("router")
        
        # Add edges
        self._add_graph_edges(graph)
        
        return graph.compile()
    
    def _add_graph_edges(self, graph: StateGraph):
        """Define all possible state transitions"""
        
        # From each handler to router
        graph.add_edge("initial_handler", "router")
        graph.add_edge("verification_handler", "router")
        graph.add_edge("authenticated_handler", "router")
        graph.add_edge("list_appointments_handler", "router") 
        graph.add_edge("confirm_appointment_handler", "router")
        graph.add_edge("cancel_appointment_handler", "router")
        graph.add_edge("error_recovery_handler", "router")
        
        # Router conditional edges
        graph.add_conditional_edges(
            "router",
            self.determine_next_state,
            {
                ConversationStates.INITIAL: "initial_handler",
                ConversationStates.VERIFICATION: "verification_handler", 
                ConversationStates.AUTHENTICATED: "authenticated_handler",
                ConversationStates.LIST_APPOINTMENTS: "list_appointments_handler",
                ConversationStates.CONFIRM_APPOINTMENT: "confirm_appointment_handler",
                ConversationStates.CANCEL_APPOINTMENT: "cancel_appointment_handler",
                ConversationStates.ERROR_RECOVERY: "error_recovery_handler",
                ConversationStates.END_CONVERSATION: END
            }
        )

    # This handles the initial state and has 3 responsibilities:
    # 1. Respond to the user's initial message and inform them of the verification requirements
    # 2. Immediately send the user to the verification state if they've provided their verification information in their first message
    # 3. Save an action as pending_action if the user intends to perform an action upon verification, if verification is successful - the user will be navigated to that action

    """
    Example response 1-
        You: Hello
        Agent: Hello! I can help you list, confirm, or cancel your healthcare appointments. To get started, 
        I'll need to verify your identity for security purposes. Could you please provide your full name, phone number, 
        and date of birth? Once verified, we can proceed with your request.

    Example response 2-
        You: my name is john doe my phone number is 5550123 and my dob is 01/01/1990
        Agent: Great! I've verified your identity. I can help you with:
        • View your appointments
        • Confirm appointments
        • Cancel appointments

        What would you like to do?

    Example response 3-
        You: my name is john doe my phone number is 5550123 and my dob is 01/01/1990 and I'd like to confirm appointment 1
        Agent: Great! I've verified your identity.
        Agent: Perfect! I've confirmed your appointment:

        **Annual Checkup** with Dr. Smith
        📅 2025-07-25 10:00:00

        Is there anything else I can help you with?
    """
    
    async def handle_initial(self, state: ConversationState) -> ConversationState:

        user_input = state["messages"][-1].content.lower()

        instruction = f"""
        You are a helpful healthcare assistant helping patients list, confirm, and/or cancel their appointments (only these 3 actions).
        The user is not verified yet, so you can't perform these actions, however you can still save their intent in pending_action if they express it
        in the user_input: {user_input}

        Before you can help a patient with their appointments, you need to verify their identity for security purposes.
        This requires their:
        1. Full name (first and last)
        2. Phone number 
        3. Date of birth (MM/DD/YYYY or YYYY-MM-DD format)

        Let the user know what they can do, but be clear about the verification requirement.
        Your entire response should be a JSON object in the following format: {{\"message\": \"<your main response>\", \"pending_action\": {PendingActions}, \"verification_provided\": {bool}}}
        and based on the user_input.
        
        If the user provides verification information then verification_provided should be set to true, otherwise false. 
        If they've specified confirming, listing/viewing, or cancelling appointments then pending_action should be set to that action.
        Finally if you're unsure, put unsure for pending_action.
        Remember to only respond with END if the user wants to end the conversation.
        """
        # Get response
        response = await self.chain.ainvoke({
            "additional_instructions": instruction, 
            "chat_history": state["messages"]})

        if self._check_end_response(response, state):
            return state

        # Attempt to parse as json
        try:
            response_json = json.loads(response.content)
            if response_json["pending_action"] in [PendingActions.LIST, PendingActions.CONFIRM, PendingActions.CANCEL]:
                state["pending_action"] = response_json["pending_action"]
            if response_json["verification_provided"]:
                state["current_state"] = ConversationStates.VERIFICATION
                return state
            state["messages"].append(AIMessage(content=response_json["message"]))
        except json.JSONDecodeError:
            state["messages"].append(AIMessage(content=response.content))

        state["current_state"] = ConversationStates.VERIFICATION
        return interrupt(state)


    # This function handles user verification, it attempts to iteratively extract name, phone and dob from the user
    # It will only return, allowing handle_verification to proceed, if all 3 pieces of information have been extracted
    async def _extract_verification_info(self, state: ConversationState) -> ConversationState:
        last_message = state["messages"][-1]
        user_input = last_message.content if isinstance(last_message, HumanMessage) else ""
        
        # Try to extract and verify information
        verification_prompt = f"""
        You are helping verify a patient's identity. Extract verification information from this message: "{user_input}"
        
        Look for:
        - Full name (first and last name)
        - Phone number (various formats acceptable)  
        - Date of birth (MM/DD/YYYY, YYYY-MM-DD, or similar formats)
        
        For each piece of information found, call the extract_verification_info tool with the field name {PII_FIELDS} and the value.
        For dates, please convert to YYYY-MM-DD format.
        Here is information that the user has already provided: "{state['verification_data']}"
        Be conversational and let the user know if inputs are missing or malformed. Remember to only respond with END and no other text 
        if the user wants to end the conversation.
        """
        
        response = await self.llm.bind_tools(self.tools).ainvoke([
            {"role": "system", "content": verification_prompt},
            {"role": "user", "content": user_input}
        ])

        if self._check_end_response(response, state):
            return state

        # Check if LLM called verification tool
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "extract_verification_info":
                    # Execute the tool
                    extracted_info_result = await extract_verification_info.ainvoke(tool_call["args"])
                    if extracted_info_result["success"]:
                        state["verification_data"] = {**state["verification_data"], **extracted_info_result["data"]}

            if len(state["verification_data"]) != len(PII_FIELDS):
                message = "I don't have all the information I need to verify your identity. So far you've provided: "
                message += ", ".join([key.replace("_", " ") for key in state["verification_data"].keys() if key in PII_FIELDS])
                message += "\nPlease provide the missing information: "
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
        
        if state["verification_attempts"] >= MAX_VERIFICATION_ATTEMPTS:
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            state["last_error"] = "Maximum verification attempts exceeded"
            return state
        
        # Populates state["verification_data"] with name, phone and dob
        await self._extract_verification_info(state)
        if state["current_state"] == ConversationStates.END_CONVERSATION:
            return state
        
        result = await verify_patient_identity(state["verification_data"])
        if result["success"]:
            state["verified"] = True
            state["patient_id"] = result["patient_id"]
            state["current_state"] = ConversationStates.AUTHENTICATED
            # Clear verified information from state for security
            state["verification_data"] = {}
            pending_action =state["pending_action"]
                
            welcome_msg = "Great! I've verified your identity."
            if not pending_action:
                welcome_msg += " I can help you with:\n"
                welcome_msg += "• View your appointments\n• Confirm appointments\n• Cancel appointments\n\n"
                welcome_msg += "What would you like to do?"
            state["messages"].append(AIMessage(content=welcome_msg))

            # Navigate to the appropriate state based on the pending action
            if pending_action == PendingActions.CONFIRM:
                state["current_state"] = ConversationStates.CONFIRM_APPOINTMENT
                return state
            elif pending_action == PendingActions.CANCEL:
                state["current_state"] = ConversationStates.CANCEL_APPOINTMENT
                return state
            elif pending_action == PendingActions.LIST:
                state["current_state"] = ConversationStates.LIST_APPOINTMENTS
                return state

        else:
            state["verification_attempts"] += 1
            error_msg = "I couldn't verify your identity with that information. "
            error_msg += "Please double-check your full name, phone number, and date of birth."
            
            state["messages"].append(AIMessage(content=error_msg))
        
        # Prompt use for response
        return interrupt(state)
    
    # This function handles the main menu after successful verification
    # It's purpose is parse what the user wants to do after they've been
    # presented with the options to view, confirm, or cancel appointments.
    # It will loop back to itself and reprompt the use if unsure.
    async def handle_authenticated(self, state: ConversationState) -> ConversationState:
        """Handle main menu after successful verification"""
        
        menu_prompt = """
        The user is verified and can access appointment features. Prioritizing their most recent chat messages first,
        determine the next action based on the user's intent. If you don't think the user wants
        any of the first 4, then respond with Unsure:
        - View/list appointments
        - Confirm appointments  
        - Cancel appointments
        - End conversation
        - Unsure

        The only text in your response should be one of the 5 listed actions."
        """
        
        response = await self.chain.ainvoke({
            "additional_instructions": menu_prompt,
            "chat_history": state["messages"]
        })

        if self._check_end_response(response, state):
            return state
        
        # Determine next action based on user intent
        intent_lower = response.content.lower()
        if any(word in intent_lower for word in ["list", "view", "view/list", "see"]):
            state["current_state"] = ConversationStates.LIST_APPOINTMENTS
            state["pending_action"] = PendingActions.LIST
        elif any(word in intent_lower for word in ["confirm", "confirmation"]):
            state["current_state"] = ConversationStates.CONFIRM_APPOINTMENT
            state["pending_action"] = PendingActions.CONFIRM
        elif any(word in intent_lower for word in ["cancel", "cancellation"]):
            state["current_state"] = ConversationStates.CANCEL_APPOINTMENT  
            state["pending_action"] = PendingActions.CANCEL
        elif any(word in intent_lower for word in ["unsure"]):
            state["messages"].append(AIMessage(content="I'm not sure what you want to do. Please let me know if you want to view your appointments, confirm an appointment, or cancel an appointment."))
            state["current_state"] = ConversationStates.AUTHENTICATED
            return interrupt(state)
        
        return state
    
    # Will list user appointments, loops back to menu unless there is an error, in which case
    # it will navigate to error recovery.
    async def handle_list_appointments(self, state: ConversationState) -> ConversationState:
        """Fetch and display patient appointments"""
        
        if not state["patient_id"]:
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            return state
        
        # Fetch appointments
        result = await fetch_appointments.ainvoke({"patient_id": state["patient_id"]})
        state["pending_action"] = None
        
        if result["success"]:
            state["appointments"] = result["appointments"]
            
            if result["appointments"]:
                appt_text = "Here are your upcoming appointments:\n\n"
                for i, appt in enumerate(result["appointments"], 1):
                    appt_text += f"{i}. **{appt['type']}** with {appt['provider']}\n"
                    appt_text += f"   {appt['datetime']}\n"
                    appt_text += f"   {appt['location']}\n"
                    appt_text += f"   Status: {appt['status']}\n\n"

                appt_text += "Let me know if you'd like to confirm or cancel any of these appointments."
                state["current_state"] = ConversationStates.AUTHENTICATED
                state["messages"].append(AIMessage(content=appt_text))
            else:
                no_appt_msg = "You don't have any upcoming appointments scheduled."
                state["messages"].append(AIMessage(content=no_appt_msg))
                state["current_state"] = ConversationStates.AUTHENTICATED
        else:
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            state["last_error"] = "Failed to fetch appointments"
        
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
        
        last_message = state["messages"][-1]
        user_input = last_message.content if isinstance(last_message, HumanMessage) else ""

        if not state["appointments"]:
            result = await fetch_appointments.ainvoke({"patient_id": state["patient_id"]})
            if result["success"]:
                state["appointments"] = result["appointments"]
        
        # Parse appointment selection
        selected_appointment = await self._parse_appointment_selection(state, user_input)
        if state["current_state"] == ConversationStates.END_CONVERSATION:
            return state
        
        if selected_appointment:
            # Perform the action
            if action == "confirm":
                result = await confirm_patient_appointment.ainvoke({"appointment_id": selected_appointment["id"]})
                action_word = "confirmed"
            else:  # cancel
                result = await cancel_patient_appointment.ainvoke({"appointment_id": selected_appointment["id"]})
                action_word = "cancelled"

            # Clear pending action
            state["pending_action"] = None
            if result["success"]:
                success_msg = f"Perfect! I've {action_word} your appointment:\n\n"
                success_msg += f"**{selected_appointment['type']}** with {selected_appointment['provider']}\n"
                success_msg += f"{selected_appointment['datetime']}\n\n"
                success_msg += "Is there anything else I can help you with?"
                
                state["messages"].append(AIMessage(content=success_msg))
                state["current_state"] = ConversationStates.AUTHENTICATED
            else:
                state["current_state"] = ConversationStates.ERROR_RECOVERY
                return state
        else:
            # Couldn't parse selection
            state["last_error"] = f"Which appointment would you like to {action.value}? Please tell me the number (1, 2, etc.)."
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            return state
        
        return interrupt(state)
    
    # Parses user input to select an appointment
    async def _parse_appointment_selection(self, state: ConversationState, user_input: str) -> Optional[Dict]:
        """Parse user input to select an appointment"""
        if not state["appointments"]:
            return None

        appointments = state["appointments"]
        
        user_input_lower = user_input.lower()

        appointment_descriptions = [f"{i+1}. {appt['type']} with {appt['provider']} on {appt['datetime']} at {appt['location']} with id {appt['id']}" for i, appt in enumerate(appointments)]
        
        NONE = "NONE"

        selection_prompt = f"""
        Given the list of appointments as described here {'\n'.join(appointment_descriptions)}
        and the user's input as {user_input_lower}, indicate to be by number only e.g. "1" or "2" which
        appointment the user is referring to. If the user's input does not match any of the appointments
        then return {NONE}. Only return a number or {NONE} in your response. No other text. 
        """

        response = await self.chain.ainvoke(
            {"additional_instructions": selection_prompt,
            "chat_history": state["messages"],
            }
        )

        if self._check_end_response(response, state):
            return None

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
        recovered_state = ConversationStates.AUTHENTICATED if state["verified"] else ConversationStates.INITIAL
        
        if state["last_error"] == "Maximum verification attempts exceeded":
            error_msg = "You've exceeded the maximum verification attempts. "
            error_msg += "Please contact our office directly for assistance."
            state["current_state"] = ConversationStates.END_CONVERSATION
        elif "Which appointment would you like" in state["last_error"]:
            UNPARSEABLE = "UNPARSEABLE"
            SOMETHING_ELSE = "SOMETHING_ELSE"
            # attempt to see if use is asking for something else, if not then go back to authenticated state
            recovery_prompt = """
            Analyze the user's last chat message and respond with only "UNPARSEABLE" if the user was asking to confirm or
            cancel an appointment but none was specified. Respond with only "SOMETHING_ELSE" if the user was asking for or
            referencing something else. Your only response to this message should be "UNPARSEABLE" or "SOMETHING_ELSE", nothing else.
            """

            response = await self.chain.ainvoke(
            {"additional_instructions": recovery_prompt,
            "chat_history": state["messages"]}
            )

            if self._check_end_response(response, state):
                return state
            
            if response.content.lower() == UNPARSEABLE.lower():
                action_state = ConversationStates.CONFIRM_APPOINTMENT if state["pending_action"] == "confirm" else ConversationStates.CANCEL_APPOINTMENT
                state["current_state"] = action_state
                state["messages"].append(AIMessage(content=state["last_error"]))
                return interrupt(state)
            elif response.content.lower() == SOMETHING_ELSE.lower():
                # Determine next action based on user intent
                state["pending_action"] = None
                state["current_state"] = recovered_state
                return state
            else:
                state["pending_action"] = None
                state["messages"].append(AIMessage(content="Apologies I don't understand your request. Please let me know if you want to view your appointments, confirm an appointment, or cancel an appointment."))
                state["current_state"] = ConversationStates.AUTHENTICATED
        else:
            error_msg = "I'm sorry, I encountered an issue. Let me try to help you differently."
            state["current_state"] = recovered_state
            state["messages"].append(AIMessage(content=error_msg))
        
        return interrupt(state)

    
    def route_conversation(self, state: ConversationState) -> ConversationState:
        """Router node - pass-through for conditional routing"""
        return state
    
    def determine_next_state(self, state: ConversationState) -> ConversationStates:
        """Determine the next state based on current conversation state"""
        return state["current_state"]
    
