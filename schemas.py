from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class UIEventType(str, Enum):
    TRIP_UPDATE = "trip_update"
    ASK_VEHICLE_TYPE = "ask_vehicle_type"


class TripDetails(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    start_date: Optional[str] = None
    return_date: Optional[str] = None
    trip_type: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"vehicle_type": None})


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
