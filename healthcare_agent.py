"""
LangGraph-based Conversational AI Agent for Healthcare Appointment Management
Fixed version compatible with latest LangGraph API
"""

from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from datetime import datetime, date
import hashlib
import re
from enum import Enum
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class ConversationStates(str, Enum):
    INITIAL = "initial"
    VERIFICATION = "verification"
    AUTHENTICATED = "authenticated"
    LIST_APPOINTMENTS = "list_appointments"
    CONFIRM_APPOINTMENT = "confirm_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    ERROR_RECOVERY = "error_recovery"
    END_CONVERSATION = "end_conversation"

class ConversationState(TypedDict):
    """Complete conversation state with all necessary tracking"""
    messages: Annotated[List[BaseMessage], "conversation history"]
    current_state: ConversationStates
    verified: bool
    patient_id: Optional[str]
    verification_attempts: int
    verification_data: Dict[str, Optional[str]]  # collected verification info
    pending_action: Optional[str]
    selected_appointment_id: Optional[str]
    appointments: List[Dict[str, Any]]
    error_count: int
    last_error: Optional[str]
    session_metadata: Dict[str, Any]

# ============================================================================
# MOCK DATABASE SERVICES
# ============================================================================

class MockPatientService:
    """Mock service for patient verification and data retrieval"""
    
    @staticmethod
    def hash_pii(value: str) -> str:
        """Hash PII for secure storage and comparison"""
        return hashlib.sha256(value.lower().strip().encode()).hexdigest()
    
    @staticmethod
    async def verify_patient(full_name: str, phone: str, dob: str) -> Optional[str]:
        """Verify patient identity and return patient_id if successful"""
        # Mock verification - replace with actual database logic
        mock_patients = {
            ("john doe", "5550123", "1990-01-01"): "patient_123",
            ("jane smith", "5550124", "1985-05-15"): "patient_456",
        }
        
        # Normalize inputs
        name_normalized = full_name.lower().strip()
        phone_normalized = re.sub(r'\D', '', phone)[-10:] if phone else ""
        dob_normalized = dob.strip() if dob else ""
        
        normalized_key = (name_normalized, phone_normalized, dob_normalized)
        
        return mock_patients.get(normalized_key)

class MockAppointmentService:
    """Mock service for appointment management"""
    
    @staticmethod
    async def get_appointments(patient_id: str) -> List[Dict[str, Any]]:
        """Get all appointments for a patient"""
        mock_appointments = {
            "patient_123": [
                {
                    "id": "appt_001",
                    "datetime": "2025-07-25 10:00:00",
                    "provider": "Dr. Smith",
                    "type": "Annual Checkup",
                    "status": "scheduled",
                    "location": "Main Clinic"
                },
                {
                    "id": "appt_002", 
                    "datetime": "2025-08-15 14:30:00",
                    "provider": "Dr. Jones",
                    "type": "Follow-up",
                    "status": "scheduled",
                    "location": "Cardiology Wing"
                }
            ]
        }
        return mock_appointments.get(patient_id, [])
    
    @staticmethod
    async def confirm_appointment(appointment_id: str) -> bool:
        """Confirm an appointment"""
        return True
    
    @staticmethod
    async def cancel_appointment(appointment_id: str) -> bool:
        """Cancel an appointment"""
        return True

# ============================================================================
# LANGGRAPH TOOLS
# ============================================================================

@tool
async def verify_patient_identity(full_name: str, phone: str, date_of_birth: str) -> Dict[str, Any]:
    """
    Verify patient identity using provided credentials
    
    Args:
        full_name: Patient's full name
        phone: Patient's phone number  
        date_of_birth: Patient's date of birth (YYYY-MM-DD format)
    """
    try:
        patient_id = await MockPatientService.verify_patient(full_name, phone, date_of_birth)
        return {
            "success": patient_id is not None,
            "patient_id": patient_id,
            "message": "Identity verified successfully" if patient_id else "Identity verification failed"
        }
    except Exception as e:
        return {"success": False, "patient_id": None, "message": f"Verification error: {str(e)}"}

@tool
async def fetch_appointments(patient_id: str) -> Dict[str, Any]:
    """
    Fetch all appointments for a verified patient
    
    Args:
        patient_id: The verified patient's ID
    """
    try:
        appointments = await MockAppointmentService.get_appointments(patient_id)
        return {
            "success": True,
            "appointments": appointments,
            "count": len(appointments)
        }
    except Exception as e:
        return {"success": False, "appointments": [], "message": f"Error fetching appointments: {str(e)}"}

@tool
async def confirm_patient_appointment(appointment_id: str) -> Dict[str, Any]:
    """
    Confirm a specific appointment
    
    Args:
        appointment_id: The ID of the appointment to confirm
    """
    try:
        success = await MockAppointmentService.confirm_appointment(appointment_id)
        return {
            "success": success,
            "message": "Appointment confirmed successfully" if success else "Failed to confirm appointment"
        }
    except Exception as e:
        return {"success": False, "message": f"Error confirming appointment: {str(e)}"}

@tool
async def cancel_patient_appointment(appointment_id: str) -> Dict[str, Any]:
    """
    Cancel a specific appointment
    
    Args:
        appointment_id: The ID of the appointment to cancel
    """
    try:
        success = await MockAppointmentService.cancel_appointment(appointment_id)
        return {
            "success": success,
            "message": "Appointment cancelled successfully" if success else "Failed to cancel appointment"
        }
    except Exception as e:
        return {"success": False, "message": f"Error cancelling appointment: {str(e)}"}

# ============================================================================
# HEALTHCARE CONVERSATION AGENT
# ============================================================================

class HealthcareConversationAgent:
    """Main conversation agent with LangGraph state machine"""
    
    def __init__(self, llm_model: str = "llama-3.3-70b-versatile"):
        # Use Groq's native integration instead of OpenAI compatibility
        groq_api_key = os.environ.get("GROQ_API_KEY")
        print(f"Using Groq API key: {groq_api_key[:5]}...{groq_api_key[-5:] if groq_api_key else None}")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not found")
            
        self.llm = ChatGroq(
            model=llm_model,
            temperature=0.1,
            max_retries=2,
            groq_api_key=groq_api_key  # Explicitly pass the API key
        )
        self.tools = [
            verify_patient_identity,
            fetch_appointments, 
            confirm_patient_appointment,
            cancel_patient_appointment
        ]
        self.graph = self._build_graph()
    
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
        graph.set_entry_point("initial_handler")
        
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
    
    async def handle_initial(self, state: ConversationState) -> ConversationState:
        """Handle initial conversation state"""
        
        system_prompt = """
        You are a helpful healthcare assistant helping patients manage their appointments.
        
        Before I can help you with your appointments, I need to verify your identity for security purposes.
        This requires your:
        1. Full name (first and last)
        2. Phone number 
        3. Date of birth (MM/DD/YYYY or YYYY-MM-DD format)
        
        Please provide this information so I can help you with your appointments.
        Be conversational and friendly, but clear about the verification requirement.
        """
        
        last_message = state["messages"][-1] if state["messages"] else None
        user_message = last_message.content if last_message else "Hello"
        
        response = await self.llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ])

        
        state["messages"].append(AIMessage(content=response.content))
        state["current_state"] = ConversationStates.VERIFICATION
        
        return state
    
    async def handle_verification(self, state: ConversationState) -> ConversationState:
        """Handle identity verification process"""
        
        if state["verification_attempts"] >= 3:
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            state["last_error"] = "Maximum verification attempts exceeded"
            return state
        
        last_message = state["messages"][-1]
        user_input = last_message.content if isinstance(last_message, HumanMessage) else ""
        
        # Try to extract and verify information
        verification_prompt = f"""
        You are helping verify a patient's identity. Extract verification information from this message: "{user_input}"
        
        Look for:
        - Full name (first and last name)
        - Phone number (various formats acceptable)  
        - Date of birth (MM/DD/YYYY, YYYY-MM-DD, or similar formats)
        
        If you have all three pieces of information, call the verify_patient_identity tool.
        If missing information, ask for what's still needed conversationally.
        """
        
        response = await self.llm.bind_tools(self.tools).ainvoke([
            {"role": "system", "content": verification_prompt},
            {"role": "user", "content": user_input}
        ])
        
        # Check if LLM called verification tool
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "verify_patient_identity":
                    # Execute the tool
                    result = await verify_patient_identity.ainvoke(tool_call["args"])
                    
                    if result["success"]:
                        state["verified"] = True
                        state["patient_id"] = result["patient_id"]
                        state["current_state"] = ConversationStates.AUTHENTICATED
                        
                        welcome_msg = "Great! I've verified your identity. I can now help you with:\n"
                        welcome_msg += "• View your appointments\n• Confirm appointments\n• Cancel appointments\n\n"
                        welcome_msg += "What would you like to do?"
                        
                        state["messages"].append(AIMessage(content=welcome_msg))
                    else:
                        state["verification_attempts"] += 1
                        error_msg = "I couldn't verify your identity with that information. "
                        error_msg += "Please double-check your full name, phone number, and date of birth."
                        
                        state["messages"].append(AIMessage(content=error_msg))
        else:
            # LLM is asking for more information
            state["messages"].append(response)
        
        return state
    
    async def handle_authenticated(self, state: ConversationState) -> ConversationState:
        """Handle main menu after successful verification"""
        
        last_message = state["messages"][-1]
        user_input = last_message.content if isinstance(last_message, HumanMessage) else ""
        
        menu_prompt = """
        The user is verified and can access appointment features. Based on their message,
        determine what they want to do and respond helpfully.
        
        Available actions:
        - View/list appointments
        - Confirm appointments  
        - Cancel appointments
        
        Be natural and conversational.
        """
        
        response = await self.llm.ainvoke([
            {"role": "system", "content": menu_prompt},
            {"role": "user", "content": user_input}
        ])
        
        state["messages"].append(response)
        
        # Determine next action based on user intent
        intent_lower = user_input.lower()
        if any(word in intent_lower for word in ["list", "show", "view", "see", "appointments"]):
            state["current_state"] = ConversationStates.LIST_APPOINTMENTS
        elif any(word in intent_lower for word in ["confirm", "confirmation"]):
            state["current_state"] = ConversationStates.LIST_APPOINTMENTS
            state["pending_action"] = "confirm"
        elif any(word in intent_lower for word in ["cancel", "cancellation"]):
            state["current_state"] = ConversationStates.LIST_APPOINTMENTS  
            state["pending_action"] = "cancel"
        
        return state
    
    async def handle_list_appointments(self, state: ConversationState) -> ConversationState:
        """Fetch and display patient appointments"""
        
        if not state["patient_id"]:
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            return state
        
        # Fetch appointments
        result = await fetch_appointments.ainvoke({"patient_id": state["patient_id"]})
        
        if result["success"]:
            state["appointments"] = result["appointments"]
            
            if result["appointments"]:
                appt_text = "Here are your upcoming appointments:\n\n"
                for i, appt in enumerate(result["appointments"], 1):
                    appt_text += f"{i}. **{appt['type']}** with {appt['provider']}\n"
                    appt_text += f"   📅 {appt['datetime']}\n"
                    appt_text += f"   📍 {appt['location']}\n"
                    appt_text += f"   Status: {appt['status']}\n\n"
                
                # Add action-specific prompts
                if state["pending_action"] == "confirm":
                    appt_text += "Which appointment would you like to confirm?"
                    state["current_state"] = ConversationStates.CONFIRM_APPOINTMENT
                elif state["pending_action"] == "cancel":
                    appt_text += "Which appointment would you like to cancel?"
                    state["current_state"] = ConversationStates.CANCEL_APPOINTMENT
                else:
                    appt_text += "What would you like to do with these appointments?"
                    state["current_state"] = ConversationStates.AUTHENTICATED
                
                state["messages"].append(AIMessage(content=appt_text))
            else:
                no_appt_msg = "You don't have any upcoming appointments scheduled."
                state["messages"].append(AIMessage(content=no_appt_msg))
                state["current_state"] = ConversationStates.AUTHENTICATED
        else:
            state["current_state"] = ConversationStates.ERROR_RECOVERY
            state["last_error"] = "Failed to fetch appointments"
        
        # Clear pending action
        state["pending_action"] = None
        return state
    
    async def handle_confirm_appointment(self, state: ConversationState) -> ConversationState:
        """Handle appointment confirmation"""
        return await self._handle_appointment_action(state, "confirm")
    
    async def handle_cancel_appointment(self, state: ConversationState) -> ConversationState:
        """Handle appointment cancellation"""
        return await self._handle_appointment_action(state, "cancel")
    
    async def _handle_appointment_action(self, state: ConversationState, action: str) -> ConversationState:
        """Generic handler for appointment actions"""
        
        last_message = state["messages"][-1]
        user_input = last_message.content if isinstance(last_message, HumanMessage) else ""
        
        # Parse appointment selection
        selected_appointment = self._parse_appointment_selection(user_input, state["appointments"])
        
        if selected_appointment:
            # Perform the action
            if action == "confirm":
                result = await confirm_patient_appointment.ainvoke({"appointment_id": selected_appointment["id"]})
                action_word = "confirmed"
            else:  # cancel
                result = await cancel_patient_appointment.ainvoke({"appointment_id": selected_appointment["id"]})
                action_word = "cancelled"
            
            if result["success"]:
                success_msg = f"Perfect! I've {action_word} your appointment:\n\n"
                success_msg += f"**{selected_appointment['type']}** with {selected_appointment['provider']}\n"
                success_msg += f"📅 {selected_appointment['datetime']}\n\n"
                success_msg += "Is there anything else I can help you with?"
                
                state["messages"].append(AIMessage(content=success_msg))
                state["current_state"] = ConversationStates.AUTHENTICATED
            else:
                error_msg = f"I wasn't able to {action} that appointment. Please try again."
                state["messages"].append(AIMessage(content=error_msg))
                state["current_state"] = ConversationStates.ERROR_RECOVERY
        else:
            # Couldn't parse selection
            clarify_msg = f"Which appointment would you like to {action}? Please tell me the number (1, 2, etc.)."
            state["messages"].append(AIMessage(content=clarify_msg))
        
        return state
    
    def _parse_appointment_selection(self, user_input: str, appointments: List[Dict]) -> Optional[Dict]:
        """Parse user input to select an appointment"""
        if not appointments:
            return None
        
        user_input_lower = user_input.lower()
        
        # Check for number selection
        for i, appt in enumerate(appointments):
            if str(i + 1) in user_input or f"{i+1}" in user_input:
                return appt
        
        # Check for appointment type or provider matching
        for appt in appointments:
            if (appt["type"].lower() in user_input_lower or 
                appt["provider"].lower() in user_input_lower):
                return appt
        
        return None
    
    async def handle_error_recovery(self, state: ConversationState) -> ConversationState:
        """Handle error states and recovery"""
        
        if state["last_error"] == "Maximum verification attempts exceeded":
            error_msg = "You've exceeded the maximum verification attempts. "
            error_msg += "Please contact our office directly for assistance."
            state["current_state"] = ConversationStates.END_CONVERSATION
        else:
            error_msg = "I'm sorry, I encountered an issue. Let me try to help you differently."
            state["current_state"] = ConversationStates.AUTHENTICATED if state["verified"] else ConversationStates.INITIAL
        
        state["messages"].append(AIMessage(content=error_msg))
        return state
    
    def route_conversation(self, state: ConversationState) -> ConversationState:
        """Router node - pass-through for conditional routing"""
        return state
    
    def determine_next_state(self, state: ConversationState) -> ConversationStates:
        """Determine the next state based on current conversation state"""
        return state["current_state"]
    
    async def process_message(self, user_message: str, session_state: Optional[ConversationState] = None) -> ConversationState:
        """Main entry point for processing user messages"""
        
        if session_state is None:
            # Initialize new conversation
            session_state = ConversationState(
                messages=[],
                current_state=ConversationStates.INITIAL,
                verified=False,
                patient_id=None,
                verification_attempts=0,
                verification_data={},
                pending_action=None,
                selected_appointment_id=None,
                appointments=[],
                error_count=0,
                last_error=None,
                session_metadata={}
            )
        
        # Add user message to conversation
        session_state["messages"].append(HumanMessage(content=user_message))
        
        # Process through state machine
        result = await self.graph.ainvoke(session_state)
        
        return result

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage(agent: HealthcareConversationAgent):
    """Example of how to use the healthcare conversation agent"""
    
    # Check if Groq API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Please set your GROQ_API_KEY environment variable")
        return
    
    # Test conversation
    print("🏥 Healthcare Agent Test")
    print("=" * 30)
    
    # Start conversation
    state = await agent.process_message("Hi, I need to check my appointments")
    print(f"Agent: {state['messages'][-1].content}")
    
    # Verification
    state = await agent.process_message("My name is John Doe, phone is 555-0123, and I was born on January 1st, 1990", state)
    print(f"Agent: {state['messages'][-1].content}")
    
    # List appointments
    state = await agent.process_message("I'd like to see my appointments", state)
    print(f"Agent: {state['messages'][-1].content}")

async def get_response(llm: ChatGroq, message: str):
    response = await llm.ainvoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message}
    ])
    return response

if __name__ == "__main__":
    import asyncio
    agent = HealthcareConversationAgent()
    asyncio.run(example_usage(agent))
