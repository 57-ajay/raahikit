"""
Session Monitor - Intelligent Agent Hold Detection

Detects scenarios where the agent should pause:
1. Noisy environment: High audio energy but no valid STT for extended period
2. User talking to others: Continuous speech (30+ sec) without agent-directed patterns
3. Silence timeout: No audio activity for extended period

The agent pauses gracefully and resumes when client sends resume event.

IMPORTANT: The monitor should NOT pause when user is actively engaged in
trip-related conversation (receiving valid STT results).
"""

import logging
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Deque
from collections import deque
from enum import Enum

logger = logging.getLogger("session-monitor")


class PauseReason(str, Enum):
    """Reasons for pausing the session."""
    SILENCE_TIMEOUT = "silence_timeout"
    NOISY_ENVIRONMENT = "noisy_environment"
    USER_TALKING_TO_OTHERS = "user_talking_to_others"
    NOISE_FLOOD = "noise_flood"
    USER_AWAY = "user_away"
    USER_DISCONNECTED = "user_disconnected"


@dataclass
class SessionMonitorConfig:
    """Configuration for session monitoring."""

    silence_timeout_seconds: float = 30.0

    noise_without_stt_timeout_seconds: float = 15.0
    noise_energy_threshold: float = 0.02

    continuous_speech_timeout_seconds: float = 30.0
    utterance_gap_seconds: float = 2.0

    min_valid_utterance_length: int = 2
    max_fragmented_utterances: int = 8

    agent_speaking_grace_seconds: float = 5.0
    post_agent_grace_seconds: float = 3.0

    post_stt_grace_seconds: float = 10.0

    energy_window_size: int = 50

    min_pause_interval_seconds: float = 60.0


@dataclass
class SpeechMetrics:
    """Tracks speech patterns to detect if user is talking to agent or others."""

    last_speech_start: float = 0.0
    last_speech_end: float = 0.0
    last_valid_stt_time: float = field(default_factory=time.time)
    last_agent_speech_end: float = field(default_factory=time.time)
    last_any_audio_time: float = field(default_factory=time.time)

    continuous_speech_start: Optional[float] = None
    is_currently_speaking: bool = False

    energy_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=50))
    high_energy_start: Optional[float] = None

    recent_utterances: Deque[str] = field(
        default_factory=lambda: deque(maxlen=10))
    fragmented_utterance_count: int = 0

    total_speech_frames: int = 0
    total_frames: int = 0

    valid_stt_count: int = 0
    last_pause_time: float = 0.0

    def get_average_energy(self) -> float:
        if not self.energy_history:
            return 0.0
        return sum(self.energy_history) / len(self.energy_history)

    def get_speech_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.total_speech_frames / self.total_frames


class SessionMonitor:
    """
    Monitors session state and detects when agent should pause.

    Detection strategies:
    1. Silence: No audio energy for extended period
    2. Noisy environment: High energy but no valid STT
    3. User talking to others: Continuous speech without conversation patterns

    IMPORTANT: Does NOT pause when user is actively engaged (recent valid STT).
    """

    def __init__(
        self,
        config: Optional[SessionMonitorConfig] = None,
        on_pause_triggered: Optional[Callable[[
            PauseReason, str], Awaitable[None]]] = None,
    ):
        self.config = config or SessionMonitorConfig()
        self.on_pause_triggered = on_pause_triggered
        self.metrics = SpeechMetrics()

        self._running = False
        self._paused = False
        self._monitor_task: Optional[asyncio.Task] = None

        logger.info(f"SessionMonitor initialized with config: "
                    f"silence={self.config.silence_timeout_seconds}s, "
                    f"""noise_no_stt={
                        self.config.noise_without_stt_timeout_seconds}s, """
                    f"""continuous_speech={
                        self.config.continuous_speech_timeout_seconds}s, """
                    f"post_stt_grace={self.config.post_stt_grace_seconds}s")

    async def start(self):
        """Start the session monitor."""
        self._running = True
        self._paused = False
        self.metrics = SpeechMetrics()  # Reset metrics
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Session monitor started")

    async def stop(self):
        """Stop the session monitor."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Session monitor stopped. Stats: "
                    f"frames={self.metrics.total_frames}, "
                    f"speech_ratio={self.metrics.get_speech_ratio():.1%}, "
                    f"avg_energy={self.metrics.get_average_energy():.4f}, "
                    f"valid_stt_count={self.metrics.valid_stt_count}")

    def pause(self):
        """Externally pause the monitor (prevents triggering while already paused)."""
        self._paused = True
        self.metrics.last_pause_time = time.time()
        logger.debug("Monitor paused externally")

    def resume(self):
        """Resume monitoring after external pause."""
        self._paused = False
        now = time.time()
        self.metrics.last_valid_stt_time = now
        self.metrics.last_any_audio_time = now
        self.metrics.last_agent_speech_end = now
        self.metrics.continuous_speech_start = None
        self.metrics.high_energy_start = None
        self.metrics.fragmented_utterance_count = 0
        logger.debug("Monitor resumed")

    def on_audio_frame(self, energy: float, is_speech: bool):
        """
        Called for each audio frame.

        Args:
            energy: RMS energy of the frame (0.0 - 1.0 normalized)
            is_speech: Whether VAD detected speech
        """
        now = time.time()
        self.metrics.total_frames += 1
        self.metrics.energy_history.append(energy)

        if energy > 0.001:
            self.metrics.last_any_audio_time = now

        if is_speech:
            self.metrics.total_speech_frames += 1

            if not self.metrics.is_currently_speaking:
                self.metrics.is_currently_speaking = True
                self.metrics.last_speech_start = now

                gap = now - self.metrics.last_speech_end
                if gap > self.config.utterance_gap_seconds:
                    self.metrics.continuous_speech_start = now
                elif self.metrics.continuous_speech_start is None:
                    self.metrics.continuous_speech_start = now
        else:
            if self.metrics.is_currently_speaking:
                self.metrics.is_currently_speaking = False
                self.metrics.last_speech_end = now

        if energy > self.config.noise_energy_threshold:
            if self.metrics.high_energy_start is None:
                self.metrics.high_energy_start = now
        else:
            self.metrics.high_energy_start = None

    def on_stt_result(self, transcript: str, is_final: bool):
        """
        Called when STT produces a result.

        Args:
            transcript: The transcribed text
            is_final: Whether this is a final (vs interim) result
        """
        if not is_final or not transcript:
            return

        transcript = transcript.strip()
        if not transcript:
            return

        now = time.time()
        self.metrics.last_valid_stt_time = now
        self.metrics.recent_utterances.append(transcript)
        self.metrics.valid_stt_count += 1

        self.metrics.high_energy_start = None

        word_count = len(transcript.split())

        if word_count < self.config.min_valid_utterance_length:
            self.metrics.fragmented_utterance_count += 1
            logger.debug(f"Short utterance ({word_count} words): '{transcript[:30]}...' "
                         f"(fragmented count: {self.metrics.fragmented_utterance_count})")
        else:
            self.metrics.fragmented_utterance_count = 0
            self.metrics.continuous_speech_start = None
            logger.debug(f"""Valid utterance ({word_count} words): '{
                         transcript[:50]}...'""")

    def on_agent_speech_start(self):
        """Called when agent starts speaking."""
        pass

    def on_agent_speech_end(self):
        """Called when agent finishes speaking."""
        self.metrics.last_agent_speech_end = time.time()
        self.metrics.continuous_speech_start = None
        self.metrics.fragmented_utterance_count = 0

    async def _monitor_loop(self):
        """Main monitoring loop - checks for pause conditions."""
        while self._running:
            try:
                await asyncio.sleep(2.0)

                if self._paused:
                    continue

                pause_reason = self._check_pause_conditions()
                if pause_reason:
                    reason, message = pause_reason
                    logger.warning(f"""Pause triggered: {
                                   reason.value} - {message}""")
                    self._paused = True  # Prevent re-triggering
                    self.metrics.last_pause_time = time.time()

                    if self.on_pause_triggered:
                        await self.on_pause_triggered(reason, message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

    def _check_pause_conditions(self) -> Optional[tuple[PauseReason, str]]:
        """
        Check all pause conditions.

        Returns:
            Tuple of (PauseReason, message) if should pause, None otherwise
        """
        now = time.time()

        time_since_agent = now - self.metrics.last_agent_speech_end
        if time_since_agent < self.config.agent_speaking_grace_seconds:
            return None

        time_since_valid_stt = now - self.metrics.last_valid_stt_time
        if time_since_valid_stt < self.config.post_stt_grace_seconds:
            logger.debug(f"""Within STT grace period ({time_since_valid_stt:.1f}s < {
                         self.config.post_stt_grace_seconds}s)""")
            return None

        time_since_last_pause = now - self.metrics.last_pause_time
        if self.metrics.last_pause_time > 0 and time_since_last_pause < self.config.min_pause_interval_seconds:
            return None

        silence_duration = now - self.metrics.last_any_audio_time
        if silence_duration >= self.config.silence_timeout_seconds:
            return (
                PauseReason.SILENCE_TIMEOUT,
                f"No audio detected for {silence_duration:.1f}s"
            )

        if self.metrics.high_energy_start is not None:
            high_energy_duration = now - self.metrics.high_energy_start
            stt_silence = now - self.metrics.last_valid_stt_time

            if (high_energy_duration >= self.config.noise_without_stt_timeout_seconds and
                    stt_silence >= self.config.noise_without_stt_timeout_seconds):
                avg_energy = self.metrics.get_average_energy()
                return (
                    PauseReason.NOISY_ENVIRONMENT,
                    f"""High noise (energy={avg_energy:.3f}) for {
                        high_energy_duration:.1f}s """
                    f"with no valid speech detected"
                )

        if self.metrics.continuous_speech_start is not None:
            continuous_duration = now - self.metrics.continuous_speech_start
            stt_silence = now - self.metrics.last_valid_stt_time

            if (continuous_duration >= self.config.continuous_speech_timeout_seconds and
                    stt_silence >= self.config.continuous_speech_timeout_seconds):
                return (
                    PauseReason.USER_TALKING_TO_OTHERS,
                    f"""Continuous speech for {
                        continuous_duration:.1f}s without valid transcription - """
                    f"user may be talking to someone else"
                )

        if self.metrics.fragmented_utterance_count >= self.config.max_fragmented_utterances:
            stt_gap = now - self.metrics.last_valid_stt_time
            if stt_gap > self.config.post_stt_grace_seconds:
                return (
                    PauseReason.USER_TALKING_TO_OTHERS,
                    f"""Detected {
                        self.metrics.fragmented_utterance_count} fragmented """
                    f"utterances - possible background conversation"
                )

        return None

    def get_stats(self) -> dict:
        """Get current monitoring statistics."""
        now = time.time()
        return {
            "total_frames": self.metrics.total_frames,
            "speech_ratio": self.metrics.get_speech_ratio(),
            "avg_energy": self.metrics.get_average_energy(),
            "silence_duration": now - self.metrics.last_any_audio_time,
            "stt_silence_duration": now - self.metrics.last_valid_stt_time,
            "continuous_speech_duration": (
                now - self.metrics.continuous_speech_start
                if self.metrics.continuous_speech_start else 0.0
            ),
            "fragmented_utterances": self.metrics.fragmented_utterance_count,
            "valid_stt_count": self.metrics.valid_stt_count,
            "is_paused": self._paused,
        }
