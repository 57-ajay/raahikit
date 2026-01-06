from dotenv import load_dotenv
import logging

from livekit import agents, rtc
from livekit.agents import (
    AgentServer, AgentSession, Agent, room_io, function_tool, RunContext)
from livekit.plugins import google, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env")

logger = logging.getLogger("test-agent")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are a smart, Hindi-speaking AI Agent named Raahi.
            You are a helpful female Assistant, whose taks is to help users and
            book a trip. You MUST collect:
            1. Origin
            2. Destination
            3. Date/Time
            4. Trip Type and Preferences ( VehicleType(SUV, SEDAN, HATCHBACK),
            DriverLanguage(Hindi, English, Gujrati etc.. (other indian languages))

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
        preferences: dict
    ):
        logger.info(
            f"BOOKING: {origin} -> {destination} on {date} ({trip_type}), Preferences: {preferences}")
        return f"Booking confirmed! {origin} se {destination} ke liye cab book ho gayi hai."


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt=google.STT(
            model="telephony",
            languages="hi-IN",
            # location="asia-south1",
        ),
        llm=google.LLM(
            model="gemini-2.5-flash",
            vertexai=True,
            location="us-central1",
            project="cabswale-ai",
        ),
        tts=google.TTS(
            # gender="female",
            voice_name="hi-IN-Chirp3-HD-Aoede",
            # voice_name="hi-IN-Neural2-A",
            language="hi-IN",
            # model_name="gemini-2.5-flash-preview-tts",
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony(
                ) if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance.",
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
