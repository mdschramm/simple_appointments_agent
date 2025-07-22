from codecs import strict_errors
import datetime
import hashlib
import re
from typing import Optional, List, Dict, Any
import json


# ============================================================================
# MOCK DATABASE SERVICES
# ============================================================================

class MockPatientService:
    """Mock service for patient verification and data retrieval"""

    @staticmethod
    def load_patients() -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all patient data from JSON file, patient data pii keys are hashed from below patient data
        mock_patients = {
            ("john doe", "5550123", datetime.date(1990, 1, 1)): "patient_123",
            ("jane smith", "5550124", datetime.date(1985, 5, 15)): "patient_456",
        }
        """
        with open("patients.json", "r") as f:
            return json.load(f)
    
    @staticmethod
    def hash_pii(key: tuple[str, str, str]) -> str:
        """Hash PII for secure storage and comparison"""
        return hashlib.sha256(str(key).lower().strip().encode()).hexdigest()


    @staticmethod
    def parse_date(date_str: str) -> Optional[datetime.date]:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y"):
            try:
                return datetime.datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        return None
    
    @staticmethod
    async def verify_patient(full_name: str, phone: str, date_of_birth: str) -> Optional[str]:
        """Verify patient identity and return patient_id if successful"""
        # Normalize inputs
        name_normalized = full_name.lower().strip()
        phone_normalized = re.sub(r'\D', '', phone)[-10:] if phone else ""
        dob_normalized = MockPatientService.parse_date(date_of_birth) if date_of_birth else None
        
        normalized_key = (name_normalized, phone_normalized, dob_normalized)
        
        return MockPatientService.load_patients().get(MockPatientService.hash_pii(normalized_key))

class MockAppointmentService:
    """Mock service for appointment management"""

    @staticmethod
    def load_appointments() -> Dict[str, List[Dict[str, Any]]]:
        """Load all appointments from JSON file"""
        with open("patient_appointments.json", "r") as f:
            return json.load(f)

    @staticmethod
    def save_appointments(appointments: Dict[str, List[Dict[str, Any]]]):
        """Save all appointments to JSON file"""
        with open("patient_appointments.json", "w") as f:
            json.dump(appointments, f, indent=4)
    
    @staticmethod
    async def get_appointments(patient_id: str, show_cancelled: bool = False) -> List[Dict[str, Any]]:
        """Get all appointments for a patient"""
        appointments = MockAppointmentService.load_appointments().get(patient_id, [])
        if not show_cancelled:
            appointments = [a for a in appointments if a["status"] != "cancelled"]
        return appointments
    
    @staticmethod
    async def confirm_appointment(appointment_id: str, patient_id: str) -> bool:
        """Confirm an appointment"""
        all_appointments = MockAppointmentService.load_appointments()
        appointments = all_appointments.get(patient_id, [])
        for appointment in appointments:
            if appointment["id"] == appointment_id:
                appointment["status"] = "confirmed"
                MockAppointmentService.save_appointments(all_appointments)
                return True
        return False
    
    @staticmethod
    async def cancel_appointment(appointment_id: str, patient_id: str) -> bool:
        """Cancel an appointment"""
        all_appointments = MockAppointmentService.load_appointments()
        appointments = all_appointments.get(patient_id, [])
        for appointment in appointments:
            if appointment["id"] == appointment_id:
                appointment["status"] = "cancelled"
                MockAppointmentService.save_appointments(all_appointments)
                return True
        return False
