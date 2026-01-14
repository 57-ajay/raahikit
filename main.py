import logging
import json
from typing import Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io, function_tool, RunContext
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from prompt import PROMPT
from events import UIEventManager
from schemas import TripDetails, IncomingUserSelection

load_dotenv(".env")
logger = logging.getLogger("raahi-agent")


class Assistant(Agent):
    def __init__(self, room: rtc.Room) -> None:
        self.room = room
        self.trip_info = TripDetails()
        self.ui = UIEventManager(room)

        formatted_prompt = PROMPT.format(
            current_date=datetime.now().strftime("%A, %Y-%m-%d %H:%M")
        )
        super().__init__(instructions=formatted_prompt)

    async def handle_ui_selection(self, selection: IncomingUserSelection):
        """Processes selection events from the frontend UI."""
        if selection.type == "vehicle_type":
            self.trip_info.preferences["vehicle_type"] = selection.value
            logger.info(f"""Updated vehicle preference from UI: {
                        selection.value}""")

            await self.ui.send_trip_update(self.trip_info)

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
        Updates the trip details on the user's screen.
        Call this immediately when the user provides any piece of booking info.
        """
        if origin:
            self.trip_info.origin = origin
        if destination:
            self.trip_info.destination = destination
        if start_date:
            self.trip_info.start_date = start_date
        if trip_type:
            self.trip_info.trip_type = trip_type
        if preferences:
            self.trip_info.preferences.update(preferences)

        if trip_type == "one_way" and start_date and not return_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
                self.trip_info.return_date = (
                    start_dt + timedelta(hours=12)).isoformat()
            except ValueError:
                pass
        elif return_date:
            self.trip_info.return_date = return_date

        await self.ui.send_trip_update(self.trip_info)
        return "User UI updated with trip details."


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

    @ctx.room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        if data.participant is None:
            return

        try:
            payload = json.loads(data.data.decode("utf-8"))
            if payload.get("event") == "user_selection":
                selection = IncomingUserSelection(**payload)

                import asyncio
                asyncio.create_task(process_selection(selection))

        except Exception as e:
            logger.error(f"Failed to process data message: {e}")

    async def process_selection(selection: IncomingUserSelection):
        await agent_instance.handle_ui_selection(selection)

        await session.generate_reply(
            instructions=f"""User has selected {selection.value}
                as their vehicle type via the UI.
                Acknowledge this and reply accordingly."""
        )

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
