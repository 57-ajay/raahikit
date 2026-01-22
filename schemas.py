from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


# =============================================================================
# STANDARD EVENT SCHEMA
# =============================================================================
# All client events follow this format:
# {
#     "name": "event_name",
#     "event_id": "SOME_ID",
#     "data": { ... }
# }
# =============================================================================

class ClientEvent(BaseModel):
    """Standard event schema from client."""
    name: str
    event_id: str
    data: Dict[str, Any] = Field(default_factory=dict)


class SessionStartEvent(ClientEvent):
    """Event fired when session starts - triggers pre-recorded audio."""
    name: str = "session_start"


class UIEventType(str, Enum):
    TRIP_UPDATE = "trip_update"


class UIEventPayload(BaseModel):
    event: UIEventType
    details: Dict[str, Any]
    timestamp: str


class UserProfile(BaseModel):
    """User profile from participant metadata."""
    name: str = "Cabswale Traveller"
    phone_number: str = "Unknown"
    extra_data: Dict[str, Any] = Field(default_factory=dict)


class TripDetails(BaseModel):
    pickup: Optional[str] = None
    destination: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    tripType: Optional[str] = None
    createTrip: bool = False
    show_vehicle_choices: bool = False
    preferencesAsked: bool = False

    preferences: Dict[str, Any] = Field(
        default_factory=lambda: {
            "vehicle_type": None,
            "gender": None,
            "dlDateOfIssue": None,
            "languages": None,
            "vehicleTypesList": None,
            "isPetAllowed": None,
            "allowHandicappedPersons": None,
            "married": None,
            "availableForCustomersPersonalCar": None,
            "availableForDrivingInEventWedding": None,
            "availableForPartTimeFullTime": None,
            "connections": None,
            "age": None,
            "withCarrier": None,
            "fuelType": None,
            "extraPreferences": None
        }
    )


class UserSelectionType(str, Enum):
    VEHICLE_TYPE = "vehicle_type"


class IncomingUserSelection(BaseModel):
    event: str
    type: UserSelectionType
    value: str
