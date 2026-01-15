from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class UIEventType(str, Enum):
    TRIP_UPDATE = "trip_update"


class UserProfile(BaseModel):
    """
    Modular user profile. Add fields here to extend user context.
    Client should send this as JSON in participant metadata.
    """
    name: str = "Cabswale Traveller"
    phone_number: str = "Unknown"
    extra_data: Dict[str, Any] = Field(default_factory=dict)


class TripDetails(BaseModel):
    pickup: Optional[str] = None
    destination: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    tripType: Optional[str] = None
    show_vehicle_choices: bool = False

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
            "fuelType": None
        }
    )


class UIEventPayload(BaseModel):
    event: UIEventType
    details: Dict[str, Any]
    timestamp: str


class UserSelectionType(str, Enum):
    VEHICLE_TYPE = "vehicle_type"


class IncomingUserSelection(BaseModel):
    event: str
    type: UserSelectionType
    value: str
