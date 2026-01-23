"""
Noise Cancellation for LiveKit Agents (Self-Hosted)

Uses noisereduce (spectral gating) which works at any sample rate.
Falls back gracefully if no library available.

Features:
- Real-time noise suppression
- Voice Activity Detection based on audio energy
- Session timeout protection
"""

import logging
import asyncio
import numpy as np
from typing import AsyncIterable, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from collections import deque
import time

logger = logging.getLogger("noise-cancellation")

NOISEREDUCE_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
    logger.info("noisereduce noise cancellation available")
except ImportError:
    logger.warning("noisereduce not installed - no noise cancellation")


@dataclass
class NoiseCancellationConfig:
    """Configuration for noise cancellation and session management."""

    enabled: bool = True

    # Noise reduction strength (0.0-1.0, higher = more aggressive)
    noise_reduction_strength: float = 0.7

    silence_timeout_seconds: float = 15.0
    noise_flood_timeout_seconds: float = 15.0

    # VAD settings (energy-based)
    speech_energy_threshold: float = 0.01  # RMS energy threshold

    agent_speaking_grace_seconds: float = 5.0


@dataclass
class AudioMetrics:
    """Tracks audio quality metrics."""

    last_speech_time: float = field(default_factory=time.time)
    last_valid_stt_time: float = field(default_factory=time.time)
    last_agent_speech_time: float = field(default_factory=time.time)
    energy_history: deque = field(default_factory=lambda: deque(maxlen=50))
    frames_processed: int = 0
    speech_frames: int = 0

    def update_energy(self, energy: float, threshold: float):
        """Update with new frame energy."""
        self.energy_history.append(energy)
        self.frames_processed += 1

        if energy > threshold:
            self.last_speech_time = time.time()
            self.speech_frames += 1

    def get_average_energy(self) -> float:
        if not self.energy_history:
            return 0.0
        return sum(self.energy_history) / len(self.energy_history)

    def on_stt_result(self):
        """Called when STT produces a valid result."""
        self.last_valid_stt_time = time.time()
        self.last_speech_time = time.time()

    def on_agent_speech(self):
        """Called when agent starts/continues speaking."""
        self.last_agent_speech_time = time.time()


class NoiseReduceProcessor:
    """
    Noise reduction using spectral gating.
    Works at any sample rate.
    """

    def __init__(self, strength: float = 0.7):
        self.strength = strength
        self._noise_profile: Optional[np.ndarray] = None
        self._frame_buffer: list = []
        self._buffer_size = 5

    def process_frame(self, audio_data: np.ndarray, sample_rate: int = 16000) -> tuple[np.ndarray, float]:
        """
        Process audio with spectral gating noise reduction.

        Returns: (processed_audio, speech_energy)
        """
        if not NOISEREDUCE_AVAILABLE:
            energy = np.sqrt(
                np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0
            return audio_data, energy

        try:
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)

            energy = np.sqrt(np.mean(audio_float ** 2))

            if energy < 0.001:
                return audio_data, energy

            reduced = nr.reduce_noise(
                y=audio_float,
                sr=sample_rate,
                stationary=False,  # Better for varying noise
                prop_decrease=self.strength,
                time_constant_s=0.5,  # Faster adaptation
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50,
            )

            output = (reduced * 32768.0).clip(-32768, 32767).astype(np.int16)
            return output, energy

        except Exception as e:
            logger.warning(f"Noisereduce error: {e}")
            energy = np.sqrt(
                np.mean(audio_data.astype(np.float32) ** 2)) / 32768.0
            return audio_data, energy


class NoiseCancellationManager:
    """
    Manages noise cancellation and session timeout.
    """

    def __init__(
        self,
        config: Optional[NoiseCancellationConfig] = None,
        on_timeout: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.config = config or NoiseCancellationConfig()
        self.on_timeout = on_timeout
        self.metrics = AudioMetrics()

        if NOISEREDUCE_AVAILABLE and self.config.enabled:
            self._processor = NoiseReduceProcessor(
                self.config.noise_reduction_strength)
            logger.info(f"""Noise reduction enabled (strength={
                        self.config.noise_reduction_strength})""")
        else:
            self._processor = None
            logger.warning("No noise cancellation processor available")

        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._sample_rate = 16000  # Will be updated from first frame

    async def start(self):
        """Start the session monitor."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Noise cancellation manager started")

    async def stop(self):
        """Stop the session monitor."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        if self.metrics.frames_processed > 0:
            speech_ratio = self.metrics.speech_frames / self.metrics.frames_processed
            logger.info(f"Session stats: {self.metrics.frames_processed} frames, "
                        f"{speech_ratio:.1%} speech, avg_energy={self.metrics.get_average_energy():.4f}")

    def process_audio_frame(self, frame_data: np.ndarray, sample_rate: int = 16000) -> tuple[np.ndarray, float]:
        """
        Process a single audio frame.

        Returns: (processed_audio, energy)
        """
        self._sample_rate = sample_rate

        if not self.config.enabled or self._processor is None:
            energy = np.sqrt(
                np.mean(frame_data.astype(np.float32) ** 2)) / 32768.0
            self.metrics.update_energy(
                energy, self.config.speech_energy_threshold)
            return frame_data, energy

        processed, energy = self._processor.process_frame(
            frame_data, sample_rate)
        self.metrics.update_energy(energy, self.config.speech_energy_threshold)

        return processed, energy

    def on_stt_result(self, transcript: str):
        """Called when STT produces a result."""
        if transcript and len(transcript.strip()) > 1:
            self.metrics.on_stt_result()
            logger.debug(f"Valid STT result received")

    def on_agent_speech(self):
        """Called when agent speaks (resets timeout)."""
        self.metrics.on_agent_speech()

    async def _monitor_loop(self):
        """Monitor for session timeouts."""
        while self._running:
            try:
                await asyncio.sleep(2.0)

                now = time.time()

                silence_duration = now - self.metrics.last_speech_time
                stt_silence_duration = now - self.metrics.last_valid_stt_time
                agent_silence = now - self.metrics.last_agent_speech_time

                if agent_silence < self.config.agent_speaking_grace_seconds:
                    logger.debug(f"""Agent spoke {
                                 agent_silence:.1f}s ago, skipping timeout check""")
                    continue

                if silence_duration >= self.config.silence_timeout_seconds:
                    logger.warning(f"""Silence timeout: no audio energy for {
                                   silence_duration:.1f}s""")
                    if self.on_timeout:
                        await self.on_timeout("silence_timeout")
                    break

                avg_energy = self.metrics.get_average_energy()
                if (avg_energy > self.config.speech_energy_threshold and
                        stt_silence_duration >= self.config.noise_flood_timeout_seconds):
                    logger.warning(f"""Noise flood: energy={avg_energy:.4f}, no STT for {
                                   stt_silence_duration:.1f}s""")
                    if self.on_timeout:
                        await self.on_timeout("noise_flood_timeout")
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")

    def get_stats(self) -> dict:
        """Get current session statistics."""
        return {
            "frames_processed": self.metrics.frames_processed,
            "speech_frames": self.metrics.speech_frames,
            "avg_energy": self.metrics.get_average_energy(),
            "silence_duration": time.time() - self.metrics.last_speech_time,
            "stt_silence_duration": time.time() - self.metrics.last_valid_stt_time,
        }
