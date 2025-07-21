from constants import PII_FIELDS
from services import MockAppointmentService, MockPatientService
from typing import Dict, Any
from langchain_core.tools import tool

# ============================================================================
# LANGGRAPH TOOLS
# ============================================================================

@tool
async def extract_verification_info(field_name: str, value: str) -> Dict[str, Any]:
    """
    Extract verification information from a message
    
    Args:
        field_name: Name of the field to extract (full_name, phone_number, date_of_birth)
        value: Value of the field
    """
    if field_name not in PII_FIELDS:
        return {"success": False, "data": None}
    return {"success": True, "data": {field_name: value}}


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

# Non-tool extra utilities
async def verify_patient_identity(verification_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify patient identity using provided credentials
        
        Args:
            Dict of verification data containing:

            full_name: Patient's full name
            phone: Patient's phone number  
            date_of_birth: Patient's date of birth (YYYY-MM-DD format)
        """
        try:
            patient_id = await MockPatientService.verify_patient(**verification_data)
            return {
                "success": patient_id is not None,
                "patient_id": patient_id,
                "message": "Identity verified successfully" if patient_id else "Identity verification failed"
            }
        except Exception as e:
            return {"success": False, "patient_id": None, "message": f"Verification error: {str(e)}"}