"""
Raahi Voice Agent - Clean Implementation

Uses LiveKit's built-in features:
- user_away_timeout for idle detection
- user_state_changed event for handling user away/back
- close_on_disconnect=False for session persistence
- Built-in reconnection handling
"""

import logging
import json
import asyncio
from typing import Optional
from datetime import datetime, timedelta

from dotenv import load_dotenv
from google.cloud import texttospeech
from livekit import agents, rtc
from livekit.agents import (
    AgentServer, AgentSession, Agent, room_io, function_tool, RunContext,
    tokenize, UserStateChangedEvent
)
from livekit.agents.tts import StreamAdapter
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from prompt import PROMPT
from events import UIEventManager
from schemas import TripDetails, IncomingUserSelection, UserProfile, ClientEvent
from audio_responses import get_response, DEFAULT_EVENT_ID
from audio_player import stream_wav_file


# =============================================================================
# SESSION STATE EVENTS (Agent <-> Client)
# =============================================================================

class SessionEvent:
    """Events sent between agent and client."""
    # Agent -> Client
    PAUSED = "session_paused"
    RESUMED = "session_resumed"

    # Client -> Agent
    RESUME_REQUEST = "session_resume"


load_dotenv(".env")
logger = logging.getLogger("raahi-agent")


class Config:
    """Centralized configuration."""
    USER_AWAY_TIMEOUT = 30.0

    CLOSE_ON_DISCONNECT = False
    DELETE_ROOM_ON_CLOSE = False

    STT_MODEL = "telephony"
    STT_LANGUAGE = "hi-IN"
    TTS_VOICE = "hi-IN-Chirp3-HD-Aoede"
    TTS_LANGUAGE = "hi-IN"

    LLM_MODEL = "gemini-2.5-flash"
    LLM_LOCATION = "asia-south1"
    LLM_PROJECT = "cabswale-ai"

    VAD_MIN_SPEECH_DURATION = 0.25
    VAD_MIN_SILENCE_DURATION = 0.6
    VAD_ACTIVATION_THRESHOLD = 0.5


class RaahiAssistant(Agent):
    """
    Raahi voice assistant for cab booking.

    Clean implementation using LiveKit's Agent class.
    """

    def __init__(
        self,
        room: rtc.Room,
        user_profile: UserProfile,
        session_data: Optional[dict] = None,
    ) -> None:
        self.room = room
        self.trip_info = TripDetails()
        self.ui = UIEventManager(room)
        self.user_profile = user_profile
        self.session_data = session_data or {}

        self._is_paused = False
        self._pause_reason: Optional[str] = None

        formatted_prompt = PROMPT.format(
            current_date=datetime.now().strftime("%A, %Y-%m-%d %H:%M"),
            user_name=self.user_profile.name,
            user_phone=self.user_profile.phone_number,
            user_context_json=json.dumps({
                **self.user_profile.extra_data,
                **self.session_data
            }, indent=2)
        )

        super().__init__(instructions=formatted_prompt)

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    async def send_event(self, event_name: str, data: Optional[dict] = None):
        """Send event to client."""
        payload = {
            "event": event_name,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self.room.local_participant.publish_data(
            payload=json.dumps(payload).encode("utf-8"),
            topic="session_events",
            reliable=True,
        )
        logger.info(f"Sent event to client: {event_name}")

    async def pause_session(self, session: AgentSession, reason: str = "user_away"):
        """Pause the session and notify client."""
        if self._is_paused:
            return

        self._is_paused = True
        self._pause_reason = reason
        logger.info(f"Session paused: {reason}")

        await self.send_event(SessionEvent.PAUSED, {
            "reason": reason,
            "trip_info": self.trip_info.model_dump(),
            "message": "Session paused - send resume when ready",
        })

        await session.say(
            text="Maaf kijiye, Mai Samajh nahi paa rhi",
            allow_interruptions=True,
        )

    async def resume_session(self, session: AgentSession):
        """Resume the session and notify client."""
        if not self._is_paused:
            return

        self._is_paused = False
        previous_reason = self._pause_reason
        self._pause_reason = None
        logger.info(f"Session resumed from: {previous_reason}")

        await self.send_event(SessionEvent.RESUMED, {
            "trip_info": self.trip_info.model_dump(),
            "message": "Session resumed",
        })

        context = self._get_trip_context()
        await session.say(
            text=f"Welcome back! {context}",
            allow_interruptions=True,
        )

    def _get_trip_context(self) -> str:
        """Get a brief context of where we left off."""
        if self.trip_info.pickup and self.trip_info.destination:
            if not self.trip_info.tripType:
                return f"Aapka trip {self.trip_info.pickup} se {self.trip_info.destination} ke liye hai. One-way ya round-trip?"
            elif not self.trip_info.startDate:
                if self.trip_info.tripType == "round-trip":
                    return f"Aapka {self.trip_info.tripType} ready hai, Kya aap apni start aur end Date bata sakte hai?"
                return f"Aapka {self.trip_info.tripType} trip ready hai, Aapko Kab jana hai?"
            else:
                return "Aapki trip details ready hain."
        elif self.trip_info.pickup:
            return f"Aapne pickup {self.trip_info.pickup} bataya tha. Drop kahan hai?"
        return "Hum kahan the? Aap apna pickup aur drop city bataiye."

    async def handle_ui_selection(self, session: AgentSession, selection: IncomingUserSelection):
        """Process UI selection from client."""
        if selection.type == "vehicle_type":
            self.trip_info.preferences["vehicle_type"] = selection.value
            self.trip_info.show_vehicle_choices = False
            logger.info(f"Vehicle selected: {selection.value}")
            await self.ui.send_trip_update(self.trip_info)

            await session.generate_reply(
                instructions=f"""User selected {
                    selection.value}. Confirm briefly and continue."""
            )

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
        preferencesAsked: Optional[bool] = None,
        createTrip: Optional[bool] = False,
    ):
        """Updates trip details and syncs with UI."""
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
            if preferences.get("vehicle_type"):
                self.trip_info.preferences["vehicleTypesList"] = [
                    preferences["vehicle_type"]]
        if preferencesAsked is not None:
            self.trip_info.preferencesAsked = preferencesAsked
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
        return "Trip details updated."


async def play_greeting(session: AgentSession, event_id: str) -> bool:
    """
    Play pre-recorded greeting or use TTS fallback.

    Returns True if successful.
    """
    response = get_response(event_id)
    audio_path = response.audio_path if response.exists() else None

    logger.info(f"""Playing greeting: {
                event_id}, has_audio: {response.exists()}""")

    try:
        if audio_path and audio_path.exists():
            await session.say(
                text=response.transcript,
                audio=stream_wav_file(audio_path),
                allow_interruptions=False,
                add_to_chat_ctx=True,
            )
        else:
            await session.say(
                text=response.transcript,
                allow_interruptions=False,
                add_to_chat_ctx=True,
            )
        return True
    except Exception as e:
        logger.error(f"Error playing greeting: {e}")
        return False


server = AgentServer()


@server.rtc_session()
async def raahi_agent(ctx: agents.JobContext):
    """Main agent entrypoint."""
    await ctx.connect()
    logger.info(f"Agent connected to room: {ctx.room.name}")

    participant = next(iter(ctx.room.remote_participants.values()), None)
    user_profile = UserProfile()

    if participant and participant.name:
        try:
            meta = json.loads(participant.name)
            user_profile = UserProfile(**meta)
        except Exception as e:
            logger.warning(f"Failed to parse user metadata: {e}")

    logger.info(f"Session for: {user_profile.name}")

    event_id = DEFAULT_EVENT_ID
    session_data = {}
    session_start_received = asyncio.Event()

    @ctx.room.on("data_received")
    def on_session_start(data: rtc.DataPacket):
        nonlocal event_id, session_data

        if not data.participant:
            return

        try:
            payload = json.loads(data.data.decode("utf-8"))
            if payload.get("name") == "session_start":
                event = ClientEvent(**payload)
                event_id = event.event_id
                session_data = event.data
                session_start_received.set()
                logger.info(f"Received session_start: {event_id}")
        except Exception as e:
            logger.error(f"session_start parsing error: {e}")

    try:
        await asyncio.wait_for(session_start_received.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        logger.info(f"No session_start received, using default: {event_id}")

    assistant = RaahiAssistant(
        room=ctx.room,
        user_profile=user_profile,
        session_data=session_data,
    )

    session = AgentSession(
        stt=google.STT(model=Config.STT_MODEL, languages=Config.STT_LANGUAGE),
        llm=google.LLM(
            model=Config.LLM_MODEL,
            vertexai=True,
            location=Config.LLM_LOCATION,
            project=Config.LLM_PROJECT,
        ),
        tts=StreamAdapter(
            tts=google.TTS(
                voice_name=Config.TTS_VOICE,
                language=Config.TTS_LANGUAGE,
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            ),
            sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
        ),
        vad=silero.VAD.load(
            min_speech_duration=Config.VAD_MIN_SPEECH_DURATION,
            min_silence_duration=Config.VAD_MIN_SILENCE_DURATION,
            activation_threshold=Config.VAD_ACTIVATION_THRESHOLD,
            sample_rate=16000,
            force_cpu=True,
        ),
        turn_detection=MultilingualModel(),
        # Key setting: timeout before user is considered "away"
        user_away_timeout=Config.USER_AWAY_TIMEOUT,
    )

    @session.on("user_state_changed")
    def on_user_state_changed(event: UserStateChangedEvent):
        logger.info(f"User state: {event.old_state} -> {event.new_state}")

        if event.new_state == "away":
            asyncio.create_task(assistant.pause_session(session, "user_away"))
        elif event.old_state == "away" and event.new_state in ("speaking", "listening"):
            if assistant.is_paused:
                asyncio.create_task(assistant.resume_session(session))

    @ctx.room.on("data_received")
    def on_data_received_handler(data: rtc.DataPacket):
        if not data.participant:
            return

        try:
            payload = json.loads(data.data.decode("utf-8"))
            event_name = payload.get("name") or payload.get("event")

            if event_name == "session_start":
                return

            if event_name == SessionEvent.RESUME_REQUEST:
                logger.info("Client requested resume")
                if assistant.is_paused:
                    asyncio.create_task(assistant.resume_session(session))
                return

            if event_name == "user_selection":
                selection = IncomingUserSelection(**payload)
                asyncio.create_task(
                    assistant.handle_ui_selection(session, selection))
                return

        except Exception as e:
            logger.error(f"Data handling error: {e}")

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant disconnected: {participant.identity}")
        if not assistant.is_paused:
            asyncio.create_task(assistant.pause_session(
                session, "user_disconnected"))

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
            close_on_disconnect=Config.CLOSE_ON_DISCONNECT,
            delete_room_on_close=Config.DELETE_ROOM_ON_CLOSE,
        ),
    )

    if not await play_greeting(session, event_id):
        await session.generate_reply(
            instructions="Greet: 'Namaste, mai Raahi hoon. Aap apna pickup aur drop city bataiye.'",
            allow_interruptions=False
        )

    logger.info("Agent session running...")


if __name__ == "__main__":
    agents.cli.run_app(server)
