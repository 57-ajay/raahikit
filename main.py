import logging
import asyncio
import json
from typing import Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google.cloud import texttospeech
from livekit import agents, rtc
from livekit.agents.tts import StreamAdapter
from livekit.agents import (
    AgentServer, AgentSession, Agent, room_io, function_tool, RunContext, tokenize)
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from prompt import PROMPT
from events import UIEventManager
from schemas import TripDetails, IncomingUserSelection, UserProfile

load_dotenv(".env")
logger = logging.getLogger("raahi-agent")


class Assistant(Agent):
    def __init__(self, room: rtc.Room, user_profile: UserProfile) -> None:
        self.room = room
        self.trip_info = TripDetails()
        self.ui = UIEventManager(room)
        self.user_profile = user_profile

        formatted_prompt = PROMPT.format(
            current_date=datetime.now().strftime("%A, %Y-%m-%d %H:%M"),
            user_name=self.user_profile.name,
            user_phone=self.user_profile.phone_number,
            user_context_json=json.dumps(
                self.user_profile.extra_data, indent=2)
        )

        super().__init__(instructions=formatted_prompt)

    async def handle_ui_selection(self, selection: IncomingUserSelection):
        """Processes selection events from the frontend UI."""
        if selection.type == "vehicle_type":
            self.trip_info.preferences["vehicle_type"] = selection.value
            self.trip_info.show_vehicle_choices = False
            logger.info(f"""Updated vehicle preference from UI: {
                        selection.value}""")

            await self.ui.send_trip_update(self.trip_info)

    @function_tool
    async def update_trip(
        self,
        ctx: RunContext,
        pickup: Optional[str] = None,
        destination: Optional[str] = None,
        startDate: Optional[str] = None,
        tripType: Optional[str] = None,
        preferences: Optional[dict] = None,
        endDate: Optional[str] = None,
        createTrip: Optional[bool] = False,
    ):
        """
        Updates the trip details.

        Args:
            preferences: Dict containing 'vehicle_type' and optional silent preferences:
                         ['gender', 'languages', 'isPetAllowed', 'fuelType', 'married',
                          'withCarrier', 'age', 'connections', 'dlDateOfIssue',
                          'availableForDrivingInEventWedding', etc.]
            createTrip: Set to True ONLY when we have full trip information.
        """
        if pickup:
            self.trip_info.pickup = pickup
        if destination:
            self.trip_info.destination = destination
        if startDate:
            self.trip_info.startDate = startDate
        if tripType:
            self.trip_info.tripType = tripType
        if preferences:
            self.trip_info.preferences.update(preferences)
        if isinstance(preferences, dict) and preferences.get("vehicle_type") is not None:
            preferences["vehicleTypesList"] = [preferences["vehicle_type"]]
        if createTrip is not None:
            self.trip_info.createTrip = createTrip

        if tripType == "one-way" and startDate and not endDate:
            try:
                start_dt = datetime.fromisoformat(startDate)
                self.trip_info.endDate = (
                    start_dt + timedelta(hours=12)).isoformat()
            except ValueError:
                pass
        elif endDate:
            self.trip_info.endDate = endDate

        await self.ui.send_trip_update(self.trip_info)
        return "User UI updated with trip details."


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    await ctx.connect()

    logger.info(f"Agent connected to room: {ctx.room.name}")

    user = ctx.room.metadata
    logger.info(f"metadata: {user}")
    participant = next(iter(ctx.room.remote_participants.values()), None)
    user_profile = UserProfile()

    if participant and participant.name:
        try:
            logger.info(f"Parsing user metadata: {participant.metadata}")
            meta_data = json.loads(participant.name)
            user_profile = UserProfile(**meta_data)
        except Exception as e:
            logger.warning(
                f"Failed to parse user metadata, using defaults. Error: {e}")

    logger.info(f"Session started for user: {user_profile.name}")

    session = AgentSession(
        stt=google.STT(
            model="telephony",
            languages="hi-IN",
        ),
        llm=google.LLM(
            model="gemini-2.5-flash",
            vertexai=True,
            location="asia-south1",
            project="cabswale-ai",
        ),

        tts=StreamAdapter(
            tts=google.TTS(
                voice_name="hi-IN-Chirp3-HD-Aoede",
                language="hi-IN",
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            ),
            sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    agent_instance = Assistant(room=ctx.room, user_profile=user_profile)

    @ctx.room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        if data.participant is None:
            return

        try:
            payload = json.loads(data.data.decode("utf-8"))
            if payload.get("event") == "user_selection":
                selection = IncomingUserSelection(**payload)

                asyncio.create_task(process_selection(selection))

        except Exception as e:
            logger.error(f"Failed to process data message: {e}")

    async def process_selection(selection: IncomingUserSelection):
        await agent_instance.handle_ui_selection(selection)

        await session.generate_reply(
            instructions=f"""User has selected {selection.value} via UI.
            Confirm this briefly and proceed to completion if all fields are done."""
        )

    await session.start(
        room=ctx.room,
        agent=agent_instance,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )

    await session.generate_reply(
        instructions="""Greet the user warmly like ->
        'namaste mai Raahi Mai aap ki trip create karne me kaise madad kar sakti hu?'.
        Then, strictly ask: 'Aap Apna pickup aur Drop city bataiye.'
        """,
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
