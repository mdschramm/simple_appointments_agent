import datetime
import hashlib
import re
from typing import Optional, List, Dict, Any
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
        # Mock verification - replace with actual database logic
        mock_patients = {
            ("john doe", "5550123", datetime.date(1990, 1, 1)): "patient_123",
            ("jane smith", "5550124", datetime.date(1985, 5, 15)): "patient_456",
        }
        
        # Normalize inputs
        name_normalized = full_name.lower().strip()
        phone_normalized = re.sub(r'\D', '', phone)[-10:] if phone else ""
        dob_normalized = MockPatientService.parse_date(date_of_birth) if date_of_birth else None
        
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
