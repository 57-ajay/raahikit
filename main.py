import logging
import json
from typing import Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    AgentServer, AgentSession, Agent, room_io, function_tool, RunContext
)
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

try:
    from raahikit.prompt import PROMPT
except ImportError:
    from prompt import PROMPT

load_dotenv(".env")

logger = logging.getLogger("raahi-agent")


class Assistant(Agent):
    def __init__(self, room: rtc.Room) -> None:
        self.room = room
        formatted_prompt = PROMPT.format(
            current_date=datetime.now().strftime("%A, %Y-%m-%d %H:%M")
        )
        super().__init__(
            instructions=formatted_prompt,
        )

    @function_tool
    async def update_trip(
        self,
        ctx: RunContext,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        start_date: Optional[str] = None,
        trip_type: Optional[str] = None,
        preferences: Optional[dict] = None,
        return_date: Optional[str] = None,
    ):
        """
        Updates the trip details on the user's screen in real-time.
        Call this tool whenever the user provides new information.

        Args:
            origin: Pickup location.
            destination: Drop-off location.
            start_date: ISO 8601 string for the trip start.
            trip_type: 'one_way' or 'round_trip'.
            preferences: Dictionary containing vehicle_type and driver_language.
            return_date: ISO 8601 string for return trip.
        """

        final_return_date = return_date
        if trip_type == "one_way" and start_date and not final_return_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
                return_dt = start_dt + timedelta(hours=12)
                final_return_date = return_dt.isoformat()
            except ValueError:
                pass

        trip_data = {
            "event": "trip_update",
            "details": {
                "origin": origin,
                "destination": destination,
                "start_date": start_date,
                "return_date": final_return_date,
                "trip_type": trip_type,
                "preferences": preferences
            }
        }

        logger.info(f"Sending UI Update: {trip_data}")

        payload_json = json.dumps(trip_data)
        await self.room.local_participant.publish_data(
            payload=payload_json,
            topic="trip_events",
            reliable=True
        )

        return "User UI updated with current details."


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    await ctx.connect()

    logger.info(f"Agent connected to room: {ctx.room.name}")

    session = AgentSession(
        stt=google.STT(
            model="telephony",
            languages="hi-IN",
        ),
        llm=google.LLM(
            model="gemini-2.5-flash",
            vertexai=True,
            location="us-central1",
            project="cabswale-ai",
        ),
        tts=google.TTS(
            voice_name="hi-IN-Chirp3-HD-Aoede",
            language="hi-IN",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    agent_instance = Assistant(room=ctx.room)

    await session.start(
        room=ctx.room,
        agent=agent_instance,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )

    await session.generate_reply(
        instructions="""Greet the user in Hinglish, introduce yourself as
        Raahi, and ask how you can help with their travel plans today.""",
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
