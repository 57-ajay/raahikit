import logging
import json
import asyncio
from typing import Optional
from datetime import datetime, timedelta
import numpy as np

from dotenv import load_dotenv
from google.cloud import texttospeech
from livekit import agents, rtc
from livekit.agents import (
    AgentServer, AgentSession, Agent, room_io, function_tool, RunContext,
    tokenize, UserStateChangedEvent, UserInputTranscribedEvent, ConversationItemAddedEvent
)
from livekit.agents.tts import StreamAdapter
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from prompt import PROMPT
from events import UIEventManager
from schemas import TripDetails, IncomingUserSelection, UserProfile, ClientEvent, ChatMessage
from audio_responses import get_response, DEFAULT_EVENT_ID
from audio_player import stream_wav_file
from session_monitor import SessionMonitor, SessionMonitorConfig, PauseReason
from audio_processor import VADEventBridge, AudioEnergyCalculator

# Import noise cancellation
from noise_cancellation import (
    NoiseCancellationManager,
    NoiseCancellationConfig,
)


class SessionEvent:
    """Events sent between agent and client."""
    PAUSED = "session_paused"
    RESUMED = "session_resumed"
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

    # Session monitor settings
    SILENCE_TIMEOUT = 30.0
    NOISE_WITHOUT_STT_TIMEOUT = 20.0
    CONTINUOUS_SPEECH_TIMEOUT = 30.0
    NOISE_ENERGY_THRESHOLD = 0.02
    POST_STT_GRACE_SECONDS = 15.0

    # Noise cancellation settings
    NC_ENABLED = True
    NC_STRENGTH = 0.7
    NC_SILENCE_TIMEOUT = 15.0
    NC_NOISE_FLOOD_TIMEOUT = 15.0
    NC_SPEECH_ENERGY_THRESHOLD = 0.01
    NC_AGENT_GRACE_SECONDS = 5.0


class RaahiAssistant(Agent):
    """
    Raahi voice assistant for cab booking.
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
        self._chat_history: list[ChatMessage] = []
        self._session: Optional[AgentSession] = None

        self._pending_create_trip = False
        self._create_trip_event_sent = False

        self._session_monitor: Optional[SessionMonitor] = None
        self._vad_bridge: Optional[VADEventBridge] = None
        self._energy_calculator = AudioEnergyCalculator()

        # Noise cancellation manager
        self._nc_manager: Optional[NoiseCancellationManager] = None
        self._audio_processing_task: Optional[asyncio.Task] = None

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

    def set_session(self, session: AgentSession):
        """Set the agent session reference."""
        self._session = session

    async def setup_noise_cancellation(self):
        """Initialize and start noise cancellation manager."""
        nc_config = NoiseCancellationConfig(
            enabled=Config.NC_ENABLED,
            noise_reduction_strength=Config.NC_STRENGTH,
            silence_timeout_seconds=Config.NC_SILENCE_TIMEOUT,
            noise_flood_timeout_seconds=Config.NC_NOISE_FLOOD_TIMEOUT,
            speech_energy_threshold=Config.NC_SPEECH_ENERGY_THRESHOLD,
            agent_speaking_grace_seconds=Config.NC_AGENT_GRACE_SECONDS,
        )

        self._nc_manager = NoiseCancellationManager(
            config=nc_config,
            on_timeout=self._on_nc_timeout,
        )

        await self._nc_manager.start()
        logger.info("Noise cancellation manager started")

    async def stop_noise_cancellation(self):
        """Stop noise cancellation manager."""
        if self._nc_manager:
            await self._nc_manager.stop()
            logger.info(f"NC stats: {self._nc_manager.get_stats()}")

        if self._audio_processing_task:
            self._audio_processing_task.cancel()
            try:
                await self._audio_processing_task
            except asyncio.CancelledError:
                pass

    async def _on_nc_timeout(self, reason: str):
        """Called when noise cancellation detects a timeout condition."""
        logger.warning(f"NC timeout triggered: {reason}")
        if self._session and not self._is_paused:
            await self.pause_session(self._session, reason)

    async def setup_session_monitor(self):
        """Initialize and start the session monitor with VAD bridge."""
        config = SessionMonitorConfig(
            silence_timeout_seconds=Config.SILENCE_TIMEOUT,
            noise_without_stt_timeout_seconds=Config.NOISE_WITHOUT_STT_TIMEOUT,
            continuous_speech_timeout_seconds=Config.CONTINUOUS_SPEECH_TIMEOUT,
            noise_energy_threshold=Config.NOISE_ENERGY_THRESHOLD,
            agent_speaking_grace_seconds=5.0,
            post_agent_grace_seconds=3.0,
            post_stt_grace_seconds=Config.POST_STT_GRACE_SECONDS,
            min_valid_utterance_length=2,
            max_fragmented_utterances=8,
            min_pause_interval_seconds=60.0,
        )

        self._session_monitor = SessionMonitor(
            config=config,
            on_pause_triggered=self._on_monitor_pause_triggered,
        )

        self._vad_bridge = VADEventBridge(
            on_speech_start=self._on_vad_speech_start,
            on_speech_end=self._on_vad_speech_end,
            on_audio_frame=self.on_audio_frame,
        )

        await self._session_monitor.start()
        logger.info("Session monitor and VAD bridge initialized")

    def _on_vad_speech_start(self):
        """Called when VAD detects speech start."""
        pass

    def _on_vad_speech_end(self):
        """Called when VAD detects speech end."""
        pass

    async def stop_session_monitor(self):
        """Stop the session monitor."""
        if self._session_monitor:
            await self._session_monitor.stop()

    async def _on_monitor_pause_triggered(self, reason: PauseReason, message: str):
        """Called when session monitor detects a pause condition."""
        logger.info(f"Monitor triggered pause: {reason.value} - {message}")

        if self._session and not self._is_paused:
            await self.pause_session(self._session, reason.value)

    def log_user(self, text: str):
        """Log user message to chat history."""
        if text and text.strip():
            msg = ChatMessage(role="user", text=text.strip())
            self._chat_history.append(msg)
            logger.debug(f"""Logged user: {text[:50]}... (total: {
                         len(self._chat_history)})""")

    def log_agent(self, text: str):
        """Log agent message to chat history."""
        if text and text.strip():
            msg = ChatMessage(role="agent", text=text.strip())
            self._chat_history.append(msg)
            logger.debug(f"""Logged agent: {
                         text[:50]}... (total: {len(self._chat_history)})""")

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

        logger.info(f"Session pausing: {reason}")

        if self._session_monitor:
            self._session_monitor.pause()

        try:
            session.interrupt()
            logger.debug("Interrupted ongoing agent activity")
        except Exception as e:
            logger.debug(f"No activity to interrupt: {e}")

        away_msg = "Maaf kijiyega me kuch samajh nahi paayi"

        pause_messages = {
            "user_away": away_msg,
            "noisy_environment": away_msg,
            "user_talking_to_others": away_msg,
            "noise_flood": away_msg,
            "silence_timeout": away_msg,
            "user_disconnected": away_msg,
            "noise_flood_timeout": away_msg,
        }

        message = pause_messages.get(
            reason, "Session paused. Resume when ready.")

        try:
            await session.say(
                text=message,
                allow_interruptions=False,
            )
        except Exception as e:
            logger.warning(f"Could not say pause message: {e}")

        self._is_paused = True
        self._pause_reason = reason

        await self.send_event(SessionEvent.PAUSED, {
            "reason": reason,
            "trip_info": self.trip_info.model_dump(),
            "message": message,
        })

        logger.info(
            f"Agent is now PAUSED - will not respond until client sends resume event")

    async def resume_session(self, session: AgentSession):
        """Resume the session and notify client."""
        logger.info(f"resume_session called, is_paused: {self._is_paused}")

        if not self._is_paused:
            logger.debug("Resume called but session is not paused")
            return

        logger.info(f"Session resuming from: {self._pause_reason}")

        self._is_paused = False
        self._pause_reason = None

        if self._session_monitor:
            self._session_monitor.resume()
            logger.debug("Session monitor resumed")

        await self.send_event(SessionEvent.RESUMED, {
            "trip_info": self.trip_info.model_dump(),
            "message": "Session resumed",
        })

        context = self._get_trip_context()
        try:
            await session.say(
                text=f"Welcome back! {context}",
                allow_interruptions=True,
            )
        except Exception as e:
            logger.error(f"Could not say resume message: {e}", exc_info=True)

        logger.info("Session RESUMED - agent is now active")

    def _get_trip_context(self) -> str:
        """Get a brief context of where we left off."""
        if self.trip_info.pickup and self.trip_info.destination:
            if not self.trip_info.tripType:
                return f"Aapka trip {self.trip_info.pickup} se {self.trip_info.destination} ke liye hai, One-way ya round-trip?"
            elif not self.trip_info.startDate:
                if self.trip_info.tripType == "round-trip":
                    return f"Aapka {self.trip_info.tripType} ready hai, Kya aap apni start aur end Date bata sakte hai?"
                return f"Aapka {self.trip_info.tripType} trip ready hai, Aapko Kab jana hai?"
            else:
                return "Aapki trip details ready hain."
        elif self.trip_info.pickup:
            return f"Aapne pickup {self.trip_info.pickup} bataya tha. Drop kahan hai?"
        return "Kya Aap apna pickup aur drop city bata sakte hai."

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

    def on_audio_frame(self, energy: float, is_speech: bool):
        """Forward audio frame data to session monitor."""
        if self._session_monitor:
            self._session_monitor.on_audio_frame(energy, is_speech)

    def on_stt_result(self, transcript: str, is_final: bool):
        """Forward STT result to session monitor and NC manager."""
        if self._session_monitor:
            self._session_monitor.on_stt_result(transcript, is_final)

        # Also notify NC manager of valid STT
        if self._nc_manager and is_final and transcript:
            self._nc_manager.on_stt_result(transcript)

    def on_agent_speech_start(self):
        """Notify session monitor that agent started speaking."""
        if self._session_monitor:
            self._session_monitor.on_agent_speech_start()

    def on_agent_speech_end(self):
        """Notify session monitor that agent finished speaking."""
        if self._session_monitor:
            self._session_monitor.on_agent_speech_end()

        # Also notify NC manager
        if self._nc_manager:
            self._nc_manager.on_agent_speech()

    async def handle_audio_track(self, track: rtc.Track, participant: rtc.RemoteParticipant):
        """
        Subscribe to audio track and process frames through noise cancellation.
        """
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        logger.info(f"Processing audio track from {participant.identity}")

        audio_stream = rtc.AudioStream(track)

        try:
            async for frame_event in audio_stream:
                if self._is_paused:
                    continue

                frame = frame_event.frame

                # Get raw audio data
                audio_bytes = frame.data.tobytes() if hasattr(
                    frame.data, 'tobytes') else bytes(frame.data)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

                # Process through noise cancellation
                if self._nc_manager:
                    processed_audio, energy = self._nc_manager.process_audio_frame(
                        audio_data,
                        frame.sample_rate
                    )

                    # Determine if this looks like speech
                    is_speech = energy > Config.NC_SPEECH_ENERGY_THRESHOLD

                    # Forward to session monitor
                    self.on_audio_frame(energy, is_speech)
                else:
                    # Fallback: just calculate energy
                    energy = self._energy_calculator.calculate_energy(
                        audio_bytes, sample_width=2)
                    is_speech = energy > Config.NOISE_ENERGY_THRESHOLD
                    self.on_audio_frame(energy, is_speech)

        except Exception as e:
            logger.error(f"Audio track processing error: {e}")
        finally:
            logger.info("Audio track processing ended")

    async def send_final_trip_event(self, completion_message: str):
        """Send the final createTrip event with complete chat history."""
        if self._create_trip_event_sent:
            logger.debug("Final trip event already sent, skipping")
            return

        if completion_message and completion_message.strip():
            if not self._chat_history or self._chat_history[-1].text != completion_message.strip():
                self.log_agent(completion_message)

        self.trip_info.chatHistory = self._chat_history.copy()

        logger.info(f"""Sending final trip event with {
                    len(self.trip_info.chatHistory)} messages""")

        await self.ui.send_trip_update(self.trip_info)
        self._create_trip_event_sent = True
        self._pending_create_trip = False

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

        if tripType == "one-way" and startDate and not endDate:
            try:
                start_dt = datetime.fromisoformat(startDate)
                self.trip_info.endDate = (
                    start_dt + timedelta(hours=12)).isoformat()
            except ValueError:
                pass
        elif endDate:
            self.trip_info.endDate = endDate

        if createTrip:
            self.trip_info.createTrip = True
            self._pending_create_trip = True
            logger.info(
                "createTrip=True received, waiting for completion message")
            return "Trip details updated. Completion message will trigger final event."
        else:
            await self.ui.send_trip_update(self.trip_info)
            return "Trip details updated."


async def play_greeting(session: AgentSession, event_id: str) -> bool:
    """Play pre-recorded greeting or use TTS fallback."""
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
        user_away_timeout=Config.USER_AWAY_TIMEOUT,
    )

    assistant.set_session(session)

    # Start noise cancellation BEFORE session monitor
    await assistant.setup_noise_cancellation()
    await assistant.setup_session_monitor()

    @session.on("user_state_changed")
    def on_user_state_changed(event: UserStateChangedEvent):
        logger.info(f"User state: {event.old_state} -> {event.new_state}")

        if event.new_state == "away" and not assistant.is_paused:
            asyncio.create_task(assistant.pause_session(session, "user_away"))

    @session.on("user_input_transcribed")
    def on_user_input(event: UserInputTranscribedEvent):
        if assistant.is_paused:
            logger.debug("Ignoring STT input - session is paused")
            return

        assistant.on_stt_result(event.transcript, event.is_final)

        if event.is_final and event.transcript:
            assistant.log_user(event.transcript)

    @session.on("conversation_item_added")
    def on_conversation_item(event: ConversationItemAddedEvent):
        if assistant.is_paused:
            return

        item = event.item
        role = "agent" if item.role == "assistant" else item.role
        text = item.text_content if hasattr(
            item, 'text_content') else str(item.content)

        if role == "agent" and text:
            assistant.log_agent(text)

            if assistant._pending_create_trip and not assistant._create_trip_event_sent:
                completion_phrases = [
                    "trip create kardi",
                    "drivers ki quotations",
                    "unse connect kar sakte"
                ]
                if any(phrase in text.lower() for phrase in completion_phrases):
                    logger.info(
                        "Completion message detected, sending final trip event")
                    asyncio.create_task(assistant.send_final_trip_event(text))

    @session.on("agent_speech_started")
    def on_agent_speech_started():
        if not assistant.is_paused:
            assistant.on_agent_speech_start()

    @session.on("agent_speech_stopped")
    def on_agent_speech_stopped():
        if not assistant.is_paused:
            assistant.on_agent_speech_end()

    @session.on("user_started_speaking")
    def on_user_started_speaking():
        if assistant.is_paused:
            return
        logger.debug("VAD: User started speaking")
        if assistant._vad_bridge:
            assistant._vad_bridge.handle_vad_start()

    @session.on("user_stopped_speaking")
    def on_user_stopped_speaking():
        if assistant.is_paused:
            return
        logger.debug("VAD: User stopped speaking")
        if assistant._vad_bridge:
            assistant._vad_bridge.handle_vad_end()

    @ctx.room.on("data_received")
    def on_data_received_handler(data: rtc.DataPacket):
        if not data.participant:
            return

        try:
            payload = json.loads(data.data.decode("utf-8"))
            event_name = payload.get("name") or payload.get("event")

            logger.info(f"""Received data event: {
                        event_name}, paused: {assistant.is_paused}""")

            if event_name == "session_start":
                return

            if event_name == SessionEvent.RESUME_REQUEST or event_name == "session_resume":
                logger.info(f"""Client requested resume, is_paused: {
                            assistant.is_paused}""")
                if assistant.is_paused:
                    asyncio.create_task(assistant.resume_session(session))
                return

            if event_name == "user_selection":
                if assistant.is_paused:
                    return
                selection = IncomingUserSelection(**payload)
                asyncio.create_task(
                    assistant.handle_ui_selection(session, selection))
                return

        except Exception as e:
            logger.error(f"Data handling error: {e}", exc_info=True)

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant disconnected: {participant.identity}")
        if not assistant.is_paused:
            asyncio.create_task(assistant.pause_session(
                session, "user_disconnected"))

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant
    ):
        """Handle track subscription - process audio through noise cancellation."""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Audio track subscribed from {participant.identity}")
            asyncio.create_task(
                assistant.handle_audio_track(track, participant))

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

    logger.info("Agent session running with noise cancellation enabled...")

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        await assistant.stop_noise_cancellation()
        await assistant.stop_session_monitor()
        raise


if __name__ == "__main__":
    agents.cli.run_app(server)
