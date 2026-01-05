import logging
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    Agent,
    room_io,
    function_tool,
    RunContext,
)


from livekit.plugins import (
    google,
    silero,
    noise_cancellation,
)

from livekit.plugins.turn_detector.multilingual import MultilingualModel
load_dotenv(".env.local")
logger = logging.getLogger("cab-agent")


class GoogleAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a smart, Hindi-speaking cab dispatcher for Raahi.

Your goal is to book a trip. You MUST collect:
1. Origin
2. Destination
3. Date/Time
4. Trip Type (One-way/Round-trip)
4. Preference (Vehicle-type, Driver language)

Speak in natural Hinglish.
Do NOT call create_trip until all details are collected.
""",
        )

    @function_tool
    async def create_trip(
        self,
        ctx: RunContext,
        origin: str,
        destination: str,
        date: str,
        trip_type: str,
    ):
        logger.info(
            f"BOOKING: {origin} -> {destination} on {date} ({trip_type})")
        return f"""Booking confirmed! {origin} se {destination}
    ke liye cab book ho gayi hai."""


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")

    session = AgentSession(
        # Google Streaming STT (Chirp)
        stt=google.STT(
            languages="hi-IN",
            model="chirp",
        ),

        # Gemini (fast + cheap, perfect for voice agents)
        llm=google.LLM(
            model="gemini-3-flash-preview",
            vertexai=True,
            project="cabswale-ai",
            location="us-central1",
        ),

        # Google Cloud TTS
        tts=google.TTS(
            gender="female",
            voice_name="hi-IN-Chirp3-HD-Aoede",
        ),

        # VAD
        vad=silero.VAD.load(),

        # NEW turn detection API
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=GoogleAssistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params:
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="""Namaste! mai Raahi.
        Aaj aap kahan se kahan jana chahenge?"""
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
