from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class UIEventType(str, Enum):
    TRIP_UPDATE = "trip_update"


class TripDetails(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    start_date: Optional[str] = None
    return_date: Optional[str] = None
    trip_type: Optional[str] = None
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
