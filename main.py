import logging
import asyncio
import json
import time
import numpy as np
from typing import Optional, AsyncIterable
from datetime import datetime, timedelta

from dotenv import load_dotenv
from google.cloud import texttospeech
from livekit import agents, rtc
from livekit.agents.tts import StreamAdapter
from livekit.agents import (
    AgentServer, AgentSession, Agent, room_io, function_tool, RunContext,
    tokenize, stt
)
from livekit.agents.voice import ModelSettings
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from prompt import PROMPT
from events import UIEventManager
from schemas import (
    TripDetails, IncomingUserSelection, UserProfile,
    ClientEvent
)
from noise_cancellation import (
    NoiseCancellationManager,
    NoiseCancellationConfig,
    NOISEREDUCE_AVAILABLE,
)
from audio_responses import get_response, DEFAULT_EVENT_ID
from audio_player import stream_wav_file

load_dotenv(".env")
logger = logging.getLogger("raahi-agent")

# =============================================================================
# CONFIGURATION
# =============================================================================

USER_AWAY_TIMEOUT = 30.0
SILENCE_TIMEOUT = 30.0
NOISE_FLOOD_TIMEOUT = 30.0
MAX_UTTERANCE_DURATION = 30.0

ENABLE_NOISE_CANCELLATION = True
NOISE_REDUCTION_STRENGTH = 0.7

SPEECH_ENERGY_THRESHOLD = 0.01
VAD_ACTIVATION_THRESHOLD = 0.5

SESSION_START_TIMEOUT = 2.0


class Assistant(Agent):
    """Raahi voice assistant for cab booking."""

    def __init__(
        self,
        room: rtc.Room,
        user_profile: UserProfile,
        nc_manager: Optional[NoiseCancellationManager] = None,
        session_data: Optional[dict] = None,
    ) -> None:
        self.room = room
        self.trip_info = TripDetails()
        self.ui = UIEventManager(room)
        self.user_profile = user_profile
        self.nc_manager = nc_manager
        self.session_data = session_data or {}

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

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        """STT node with optional noise cancellation."""
        if self.nc_manager and ENABLE_NOISE_CANCELLATION:
            async for event in Agent.default.stt_node(
                self,
                self._process_audio_with_nc(audio),
                model_settings
            ):
                yield event
        else:
            async for event in Agent.default.stt_node(self, audio, model_settings):
                yield event

    async def _process_audio_with_nc(
        self,
        audio_stream: AsyncIterable[rtc.AudioFrame]
    ) -> AsyncIterable[rtc.AudioFrame]:
        """Apply noise cancellation to audio frames."""
        async for frame in audio_stream:
            try:
                frame_bytes = bytes(frame.data.cast('b'))
                audio_data = np.frombuffer(frame_bytes, dtype=np.int16).copy()

                processed_data, energy = self.nc_manager.process_audio_frame(
                    audio_data,
                    sample_rate=frame.sample_rate
                )

                yield rtc.AudioFrame(
                    data=processed_data.tobytes(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    samples_per_channel=frame.samples_per_channel,
                )
            except Exception as e:
                logger.warning(f"NC error: {e}")
                yield frame

    async def handle_ui_selection(self, selection: IncomingUserSelection):
        """Process UI selection from client."""
        if selection.type == "vehicle_type":
            self.trip_info.preferences["vehicle_type"] = selection.value
            self.trip_info.show_vehicle_choices = False
            logger.info(f"Vehicle selected: {selection.value}")
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
        if isinstance(preferences, dict) and preferences.get("vehicle_type"):
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
        return "Trip details updated."


async def play_session_start_response(
    session: AgentSession,
    event_id: str,
) -> bool:
    """
    Play pre-recorded audio response for session_start event.

    Uses session.say() with:
    - text: transcript (for chat context + TTS fallback)
    - audio: WAV file stream (if available)

    Returns True if played successfully.
    """
    response = get_response(event_id)
    transcript = response.transcript
    audio_path = response.audio_path if response.exists() else None

    logger.info(f"""Playing response for: {
                event_id}, audio_exists: {response.exists()}""")

    try:
        if audio_path and audio_path.exists():
            logger.info(f"Streaming audio: {audio_path}")
            await session.say(
                text=transcript,
                audio=stream_wav_file(audio_path),
                allow_interruptions=False,
                add_to_chat_ctx=True,
            )
        else:
            logger.info(f"Using TTS for: {event_id}")
            await session.say(
                text=transcript,
                allow_interruptions=False,
                add_to_chat_ctx=True,
            )
        return True

    except Exception as e:
        logger.error(f"Error playing response: {e}")
        try:
            await session.say(
                text=transcript,
                allow_interruptions=False,
                add_to_chat_ctx=True,
            )
            return True
        except Exception as e2:
            logger.error(f"TTS fallback failed: {e2}")
            return False


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    await ctx.connect()
    logger.info(f"Agent connected: {ctx.room.name}")

    participant = next(iter(ctx.room.remote_participants.values()), None)
    user_profile = UserProfile()

    if participant and participant.name:
        try:
            meta = json.loads(participant.name)
            user_profile = UserProfile(**meta)
        except Exception as e:
            logger.warning(f"Failed to parse metadata: {e}")

    logger.info(f"Session started for: {user_profile.name}")

    # Session state
    session_ended = asyncio.Event()
    session_start_received = asyncio.Event()
    pending_event: Optional[ClientEvent] = None

    user_speech_start_time: Optional[float] = None
    long_utterance_task: Optional[asyncio.Task] = None
    accumulated_transcript: str = ""

    async def on_timeout(reason: str):
        logger.warning(f"Session timeout: {reason}")
        await ctx.room.local_participant.publish_data(
            json.dumps({"reason": reason, "topic": "timeout"}).encode(),
            reliable=True,
        )
        session_ended.set()

    async def handle_long_utterance():
        nonlocal accumulated_transcript
        await asyncio.sleep(MAX_UTTERANCE_DURATION)

        if session and not session_ended.is_set():
            logger.info(f"Long utterance detected ({MAX_UTTERANCE_DURATION}s)")
            transcript = accumulated_transcript.strip()

            if transcript:
                await session.generate_reply(
                    instructions=f"""User spoke for too long: "{transcript}"
                    Extract any trip info and continue. Keep response SHORT."""
                )
            else:
                await session.generate_reply(
                    instructions="User was speaking but unclear. Ask: 'Sorry, aap dobara bataiye?'"
                )

    nc_config = NoiseCancellationConfig(
        enabled=ENABLE_NOISE_CANCELLATION and NOISEREDUCE_AVAILABLE,
        noise_reduction_strength=NOISE_REDUCTION_STRENGTH,
        silence_timeout_seconds=SILENCE_TIMEOUT,
        noise_flood_timeout_seconds=NOISE_FLOOD_TIMEOUT,
        speech_energy_threshold=SPEECH_ENERGY_THRESHOLD,
        agent_speaking_grace_seconds=5.0,
    )

    nc_manager = NoiseCancellationManager(
        config=nc_config,
        on_timeout=on_timeout,
    )

    session = AgentSession(
        stt=google.STT(model="telephony", languages="hi-IN"),
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
        vad=silero.VAD.load(
            min_speech_duration=0.25,
            min_silence_duration=0.6,
            prefix_padding_duration=0.3,
            activation_threshold=VAD_ACTIVATION_THRESHOLD,
            sample_rate=16000,
            force_cpu=True,
        ),
        turn_detection=MultilingualModel(),
        user_away_timeout=300.0,
    )

    @session.on("user_input_transcribed")
    def on_transcription(ev):
        nonlocal accumulated_transcript
        transcript = getattr(ev, 'transcript', '') or ''
        if len(transcript.strip()) > 1:
            nc_manager.on_stt_result(transcript)
            accumulated_transcript += " " + transcript

    @session.on("user_speech_started")
    def on_user_speech_started(ev):
        nonlocal user_speech_start_time, long_utterance_task, accumulated_transcript
        user_speech_start_time = time.time()
        accumulated_transcript = ""
        if long_utterance_task and not long_utterance_task.done():
            long_utterance_task.cancel()
        long_utterance_task = asyncio.create_task(handle_long_utterance())

    @session.on("user_speech_committed")
    def on_user_speech_committed(ev):
        nonlocal long_utterance_task, accumulated_transcript
        if long_utterance_task and not long_utterance_task.done():
            long_utterance_task.cancel()
        accumulated_transcript = ""

    @session.on("agent_speech_started")
    def on_agent_speech_started(ev):
        nonlocal long_utterance_task
        nc_manager.on_agent_speech()
        if long_utterance_task and not long_utterance_task.done():
            long_utterance_task.cancel()

    @session.on("agent_speech_stopped")
    def on_agent_speech_stopped(ev):
        nc_manager.on_agent_speech()

    @ctx.room.on("data_received")
    def on_data(data: rtc.DataPacket):
        nonlocal pending_event

        if not data.participant:
            return

        try:
            payload = json.loads(data.data.decode("utf-8"))

            if payload.get("name") == "session_start":
                event = ClientEvent(**payload)
                pending_event = event
                session_start_received.set()
                logger.info(f"Received session_start: {event.event_id}")
                return

            if payload.get("event") == "user_selection":
                nc_manager.metrics.on_stt_result()
                selection = IncomingUserSelection(**payload)
                asyncio.create_task(handle_selection(selection))

        except Exception as e:
            logger.error(f"Data processing error: {e}")

    async def handle_selection(selection: IncomingUserSelection):
        await agent_instance.handle_ui_selection(selection)
        await session.generate_reply(
            instructions=f"""User selected {
                selection.value}. Confirm briefly and continue."""
        )

    await nc_manager.start()

    event_id = DEFAULT_EVENT_ID
    session_data = {}

    try:
        await asyncio.wait_for(
            session_start_received.wait(),
            timeout=SESSION_START_TIMEOUT
        )
        if pending_event:
            event_id = pending_event.event_id
            session_data = pending_event.data
            logger.info(f"Using event_id: {event_id}, data: {session_data}")
    except asyncio.TimeoutError:
        logger.info(f"No session_start received, using default: {event_id}")

    agent_instance = Assistant(
        room=ctx.room,
        user_profile=user_profile,
        nc_manager=nc_manager,
        session_data=session_data,
    )

    await session.start(
        room=ctx.room,
        agent=agent_instance,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )

    played = await play_session_start_response(session, event_id)

    if not played:
        await session.generate_reply(
            instructions="Greet: 'Namaste, mai Raahi hoon. Aap apna pickup aur drop city bataiye.'",
            allow_interruptions=False
        )

    try:
        await session_ended.wait()
    except asyncio.CancelledError:
        pass
    finally:
        if long_utterance_task and not long_utterance_task.done():
            long_utterance_task.cancel()

        await nc_manager.stop()
        logger.info(f"Session stats: {nc_manager.get_stats()}")

        try:
            await ctx.room.disconnect()
        except Exception:
            pass

        logger.info("Session ended")


if __name__ == "__main__":
    agents.cli.run_app(server)
