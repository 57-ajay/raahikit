import logging
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    AgentServer, AgentSession, Agent, room_io, function_tool, RunContext
)
from livekit.plugins import google, noise_cancellation, silero
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
    async def create_trip(
        self,
        ctx: RunContext,
        origin: str,
        destination: str,
        start_date: str,
        trip_type: str,
        preferences: dict,
        return_date: str = None,
    ):
        """
        Finalizes the trip details and sends a booking event to the client.

        Args:
            origin: Pickup location.
            destination: Drop-off location.
            start_date: ISO 8601 string for the trip start.
            trip_type: 'one_way' or 'round_trip'.
            preferences: Dictionary containing vehicle_type and driver_language.
            return_date: ISO 8601 string for return trip (optional for one_way).
        """

        final_return_date = return_date

        if trip_type == "one_way" and not final_return_date:
            try:
                dt_format = "%Y-%m-%dT%H:%M:%S"
                if len(start_date.split(":")) == 2:
                    dt_format = "%Y-%m-%dT%H:%M"

                start_dt = datetime.fromisoformat(start_date)
                return_dt = start_dt + timedelta(hours=12)
                final_return_date = return_dt.isoformat()
                logger.info(
                    f"Auto-calculated return date: {final_return_date}")
            except ValueError as e:
                logger.error(f"Date parsing error: {e}")
                final_return_date = start_date

        trip_data = {
            "event": "create_trip",
            "details": {
                "origin": origin,
                "destination": destination,
                "start_date": start_date,
                "return_date": final_return_date,
                "trip_type": trip_type,
                "preferences": preferences
            }
        }

        payload_json = json.dumps(trip_data)
        await self.room.local_participant.publish_data(
            payload=payload_json,
            topic="trip_events",
            reliable=True
        )

        return f"Booking event sent! {origin} se {destination} ke liye request bhej di gayi hai."


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    # Connect to the room
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

    # Pass the room object to the Assistant so it can publish data
    agent_instance = Assistant(room=ctx.room)

    await session.start(
        room=ctx.room,
        agent=agent_instance,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony(
                ) if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user in Hinglish, introduce yourself as Raahi, and ask how you can help with their travel plans today.",
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
