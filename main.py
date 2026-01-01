import logging
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io, function_tool, RunContext
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
logger = logging.getLogger("cab-agent")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful cab booking assistant for LiveCab.
            Your goal is to book a trip for the user.
            You must collect the following 4 pieces of information:
            1. Origin (Pickup location)
            2. Destination (Dropoff location)
            3. Date/Time of trip
            4. Trip Type (One-way or Round-trip)

            Do not call the booking tool until you have ALL 4 pieces of information.
            If information is missing, ask the user specifically for that missing piece.
            Once you have all details, immediately call the 'create_trip' function.
            """,
        )

    @function_tool
    async def create_trip(self, ctx: RunContext, origin: str, destination: str, date: str, trip_type: str):
        """
        Creates a cab trip booking. Call this ONLY when you have collected all necessary details.

        Args:
            origin: The pickup location (city or address).
            destination: The dropoff location.
            date: The date and time of the trip (e.g., "tomorrow at 5pm").
            trip_type: The type of trip, either 'one-way' or 'round-trip'.
        """
        # In a real app, we would save this to your database here.
        logger.info(f"Creating trip: {origin} -> {destination} on {date} ({trip_type})")

        # The return value is spoken back to the user or used by the LLM to generate a confirmation
        return f"Success! I have booked a {trip_type} cab from {origin} to {destination} for {date}. Your driver will arrive shortly."

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and ask where they would like to go today."
    )

if __name__ == "__main__":
    agents.cli.run_app(server)
