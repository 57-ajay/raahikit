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
    tokenize, stt)
from livekit.agents.voice import ModelSettings
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from prompt import PROMPT
from events import UIEventManager
from schemas import TripDetails, IncomingUserSelection, UserProfile
from noise_cancellation import (
    NoiseCancellationManager,
    NoiseCancellationConfig,
    NOISEREDUCE_AVAILABLE,
)

load_dotenv(".env")
logger = logging.getLogger("raahi-agent")

USER_AWAY_TIMEOUT = 30.0              # Built-in LiveKit timeout (backup)
SILENCE_TIMEOUT = 30.0                # No audio energy at all
NOISE_FLOOD_TIMEOUT = 30.0            # Audio but no valid STT
MAX_UTTERANCE_DURATION = 15.0

ENABLE_NOISE_CANCELLATION = True
NOISE_REDUCTION_STRENGTH = 0.7        # 0.0-1.0, higher = more aggressive

# Energy threshold for detecting speech (adjust if too sensitive/insensitive)
SPEECH_ENERGY_THRESHOLD = 0.01

# VAD tuning (higher = less sensitive to noise)
VAD_ACTIVATION_THRESHOLD = 0.5        # 0.4-0.7


class Assistant(Agent):
    """
    Raahi assistant with integrated noise cancellation.

    Overrides stt_node to apply RNNoise before sending audio to STT.
    """

    def __init__(
        self,
        room: rtc.Room,
        user_profile: UserProfile,
        nc_manager: Optional[NoiseCancellationManager] = None,
    ) -> None:
        self.room = room
        self.trip_info = TripDetails()
        self.ui = UIEventManager(room)
        self.user_profile = user_profile
        self.nc_manager = nc_manager

        formatted_prompt = PROMPT.format(
            current_date=datetime.now().strftime("%A, %Y-%m-%d %H:%M"),
            user_name=self.user_profile.name,
            user_phone=self.user_profile.phone_number,
            user_context_json=json.dumps(
                self.user_profile.extra_data, indent=2)
        )

        super().__init__(instructions=formatted_prompt)

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        """
        Custom STT node with noise cancellation.

        Processes each audio frame through RNNoise before passing to STT.
        This is an async generator that yields SpeechEvents.
        """
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
        """
        Generator that applies noise cancellation to each audio frame.
        """
        async for frame in audio_stream:
            try:
                frame_bytes = bytes(frame.data.cast('b'))
                audio_data = np.frombuffer(frame_bytes, dtype=np.int16).copy()

                processed_data, energy = self.nc_manager.process_audio_frame(
                    audio_data,
                    sample_rate=frame.sample_rate
                )

                processed_frame = rtc.AudioFrame(
                    data=processed_data.tobytes(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    samples_per_channel=frame.samples_per_channel,
                )

                yield processed_frame

            except Exception as e:
                logger.warning(f"NC frame processing error: {e}")
                # On error, pass through original frame
                yield frame

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
        """Updates the trip details."""
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

    if NOISEREDUCE_AVAILABLE and ENABLE_NOISE_CANCELLATION:
        logger.info("Noise cancellation ENABLED (noisereduce)")
    else:
        logger.warning("Noise cancellation DISABLED")

    participant = next(iter(ctx.room.remote_participants.values()), None)
    user_profile = UserProfile()

    if participant and participant.name:
        try:
            meta_data = json.loads(participant.name)
            user_profile = UserProfile(**meta_data)
        except Exception as e:
            logger.warning(f"Failed to parse user metadata: {e}")

    logger.info(f"Session started for user: {user_profile.name}")

    session_ended = asyncio.Event()
    session: Optional[AgentSession] = None

    user_speech_start_time: Optional[float] = None
    long_utterance_task: Optional[asyncio.Task] = None
    accumulated_transcript: str = ""

    async def on_timeout(reason: str):
        """Handle session timeout - silently end without any message."""
        logger.warning(f"Session timeout: {reason}")
        session_ended.set()

    async def handle_long_utterance():
        """Called when user speaks for too long - interrupt and respond."""
        nonlocal accumulated_transcript

        await asyncio.sleep(MAX_UTTERANCE_DURATION)

        if session and not session_ended.is_set():
            logger.info(f"""Long utterance detected ({
                        MAX_UTTERANCE_DURATION}s), interrupting...""")

            transcript_so_far = accumulated_transcript.strip()

            if transcript_so_far:
                await session.generate_reply(
                    instructions=f"""The user has been speaking for a long time. Their message so far: "{transcript_so_far}"

                    RULES:
                    1. If this contains ANY trip-related info (pickup, destination, date, trip type), extract it, call update_trip, and continue the booking flow naturally.
                    2. If this is completely unrelated to cab booking (chatting, asking random questions, etc.), say: "Main sirf cab booking mein madad kar sakti hoon. Aap apna pickup aur drop city bataiye." and continue.
                    3. Keep your response SHORT - one sentence max.
                    4. Do NOT end the session."""
                )
            else:
                await session.generate_reply(
                    instructions="User was speaking but no clear words detected. Ask them to repeat: 'Sorry, aap dobara bataiye?'"
                )

    nc_config = NoiseCancellationConfig(
        enabled=ENABLE_NOISE_CANCELLATION and NOISEREDUCE_AVAILABLE,
        noise_reduction_strength=NOISE_REDUCTION_STRENGTH,
        silence_timeout_seconds=SILENCE_TIMEOUT,
        noise_flood_timeout_seconds=NOISE_FLOOD_TIMEOUT,
        speech_energy_threshold=SPEECH_ENERGY_THRESHOLD,
        agent_speaking_grace_seconds=5.0,  # Don't timeout for 5s after agent speaks
    )

    nc_manager = NoiseCancellationManager(
        config=nc_config,
        on_timeout=on_timeout,
    )

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

    agent_instance = Assistant(
        room=ctx.room,
        user_profile=user_profile,
        nc_manager=nc_manager,
    )

    @session.on("user_input_transcribed")
    def on_transcription(ev):
        """Track valid STT results for noise flood detection and long utterance handling."""
        nonlocal accumulated_transcript

        transcript = getattr(ev, 'transcript', '') or ''
        if len(transcript.strip()) > 1:
            nc_manager.on_stt_result(transcript)
            accumulated_transcript += " " + transcript
            logger.debug(f"Valid STT: {transcript[:50]}...")

    @session.on("user_speech_started")
    def on_user_speech_started(ev):
        """Track when user starts speaking for long utterance detection."""
        nonlocal user_speech_start_time, long_utterance_task, accumulated_transcript

        user_speech_start_time = time.time()
        accumulated_transcript = ""

        if long_utterance_task and not long_utterance_task.done():
            long_utterance_task.cancel()

        long_utterance_task = asyncio.create_task(handle_long_utterance())
        logger.debug("User started speaking, monitoring for long utterance")

    @session.on("user_speech_committed")
    def on_user_speech_committed(ev):
        """User finished speaking normally - cancel long utterance monitor."""
        nonlocal long_utterance_task, accumulated_transcript

        if long_utterance_task and not long_utterance_task.done():
            long_utterance_task.cancel()
            logger.debug(
                "User speech committed, cancelled long utterance monitor")

        accumulated_transcript = ""

    @session.on("user_state_changed")
    def on_state_change(ev):
        """Handle user state changes."""
        state = getattr(ev, 'state', None)
        logger.debug(f"User state: {state}")

    @session.on("agent_speech_started")
    def on_agent_speech_started(ev):
        """Track when agent starts speaking."""
        nonlocal long_utterance_task

        nc_manager.on_agent_speech()

        if long_utterance_task and not long_utterance_task.done():
            long_utterance_task.cancel()

        logger.debug("Agent started speaking")

    @session.on("agent_speech_stopped")
    def on_agent_speech_stopped(ev):
        """Track when agent stops speaking (user should respond soon)."""
        nc_manager.on_agent_speech()  # Reset timeout when agent finishes
        logger.debug("Agent stopped speaking")

    @ctx.room.on("data_received")
    def on_data(data: rtc.DataPacket):
        """Handle UI selections."""
        if not data.participant:
            return

        try:
            payload = json.loads(data.data.decode("utf-8"))
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

    await session.start(
        room=ctx.room,
        agent=agent_instance,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )

    await session.generate_reply(
        instructions="""Greet warmly: 'Namaste, mai Raahi. Aap ki trip create karne me kaise madad kar sakti hu?'
        Then ask: 'Aap apna pickup aur drop city bataiye.'"""
    )

    try:
        await session_ended.wait()
    except asyncio.CancelledError:
        pass
    finally:
        if long_utterance_task and not long_utterance_task.done():
            long_utterance_task.cancel()

        await nc_manager.stop()
        stats = nc_manager.get_stats()
        logger.info(f"Session stats: {stats}")

        try:
            await ctx.room.disconnect()
        except Exception:
            pass
        logger.info("Session ended")


if __name__ == "__main__":
    agents.cli.run_app(server)
