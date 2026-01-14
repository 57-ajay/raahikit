import logging
from datetime import datetime
from livekit import rtc
from schemas import UIEventType, UIEventPayload, TripDetails
from typing import Dict, Any

logger = logging.getLogger("raahi-events")


class UIEventManager:
    def __init__(self, room: rtc.Room):
        self.room = room
        self.topic = "trip_events"

    async def emit(self, event_type: UIEventType, details: Dict[str, Any]):
        """Centralized method to send any UI event."""
        payload = UIEventPayload(
            event=event_type,
            details=details,
            timestamp=datetime.utcnow().isoformat()
        )

        logger.info(f"Emitting UI Event: {event_type.value}")

        await self.room.local_participant.publish_data(
            payload=payload.model_dump_json().encode("utf-8"),
            topic=self.topic,
            reliable=True
        )

    async def send_trip_update(self, trip_info: TripDetails):
        """Helper for standard trip updates."""
        if (
            trip_info.origin
            and trip_info.destination
            and trip_info.start_date
            and trip_info.return_date
            and trip_info.trip_type
            and not trip_info.preferences["vehicle_type"]
        ):
            await self.emit(
                UIEventType.ASK_VEHICLE_TYPE,
                trip_info.model_dump(),
            )
        else:
            await self.emit(
                UIEventType.TRIP_UPDATE,
                trip_info.model_dump(),
            )
