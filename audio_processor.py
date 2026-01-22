"""
Audio Stream Processor

Hooks into LiveKit's audio pipeline to monitor:
- Audio energy levels
- VAD (Voice Activity Detection) events
- Speech patterns

Provides data to SessionMonitor for intelligent pause detection.
"""

import logging
import numpy as np
from typing import Optional, Callable
from livekit import rtc

logger = logging.getLogger("audio-processor")


class AudioEnergyCalculator:
    """
    Calculates audio energy from raw audio frames.

    Uses RMS (Root Mean Square) for energy calculation.
    """

    def __init__(self, smoothing_factor: float = 0.3):
        """
        Args:
            smoothing_factor: Exponential smoothing for energy (0-1).
                             Higher = more responsive, lower = more stable.
        """
        self.smoothing_factor = smoothing_factor
        self._smoothed_energy = 0.0
        self._frame_count = 0

    def calculate_energy(self, audio_data: bytes, sample_width: int = 2) -> float:
        """
        Calculate normalized RMS energy from audio data.

        Args:
            audio_data: Raw audio bytes
            sample_width: Bytes per sample (2 for 16-bit)

        Returns:
            Normalized energy (0.0 to 1.0)
        """
        try:
            if not audio_data:
                return 0.0

            if sample_width == 2:
                samples = np.frombuffer(audio_data, dtype=np.int16)
            else:
                samples = np.frombuffer(audio_data, dtype=np.int8)

            if len(samples) == 0:
                return 0.0

            # Calculate RMS energy, normalized to 0-1
            samples_float = samples.astype(np.float32)
            max_val = 32768.0 if sample_width == 2 else 128.0
            normalized = samples_float / max_val

            rms = np.sqrt(np.mean(normalized ** 2))

            # Apply exponential smoothing
            self._smoothed_energy = (
                self.smoothing_factor * rms +
                (1 - self.smoothing_factor) * self._smoothed_energy
            )
            self._frame_count += 1

            return self._smoothed_energy

        except Exception as e:
            logger.warning(f"Energy calculation error: {e}")
            return 0.0

    def reset(self):
        """Reset the energy calculator state."""
        self._smoothed_energy = 0.0
        self._frame_count = 0


class AudioStreamProcessor:
    """
    Processes audio streams and provides callbacks for energy and VAD events.

    Integrates with LiveKit's audio track to monitor incoming audio.
    """

    def __init__(
        self,
        on_audio_frame: Optional[Callable[[float, bool], None]] = None,
        energy_threshold: float = 0.02,
    ):
        """
        Args:
            on_audio_frame: Callback(energy, is_speech) for each frame
            energy_threshold: Threshold for considering audio as "activity"
        """
        self.on_audio_frame = on_audio_frame
        self.energy_threshold = energy_threshold

        self._energy_calculator = AudioEnergyCalculator()
        self._is_speech = False
        self._running = False

    def process_frame(self, frame: rtc.AudioFrame) -> tuple[float, bool]:
        """
        Process a single audio frame.

        Args:
            frame: LiveKit AudioFrame

        Returns:
            Tuple of (energy, is_likely_speech)
        """
        energy = self._energy_calculator.calculate_energy(
            frame.data.tobytes() if hasattr(frame.data, 'tobytes') else bytes(frame.data),
            sample_width=2  # 16-bit audio
        )

        is_likely_speech = energy > self.energy_threshold

        if self.on_audio_frame:
            self.on_audio_frame(energy, is_likely_speech)

        return energy, is_likely_speech

    def on_vad_event(self, is_speech: bool):
        """
        Called when VAD detects speech start/end.

        Args:
            is_speech: True if speech started, False if ended
        """
        self._is_speech = is_speech
        logger.debug(f"""VAD event: speech={
                     'started' if is_speech else 'ended'}""")

    def reset(self):
        """Reset processor state."""
        self._energy_calculator.reset()
        self._is_speech = False


class VADEventBridge:
    """
    Bridges VAD events from LiveKit to the session monitor.

    Tracks speech patterns and provides callbacks.
    """

    def __init__(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        on_audio_frame: Optional[Callable[[float, bool], None]] = None,
    ):
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_audio_frame = on_audio_frame

        self._is_speaking = False
        self._energy_calculator = AudioEnergyCalculator()

    def handle_vad_start(self):
        """Called when VAD detects speech start."""
        if not self._is_speaking:
            self._is_speaking = True
            logger.debug("Speech started")
            if self.on_speech_start:
                self.on_speech_start()

    def handle_vad_end(self):
        """Called when VAD detects speech end."""
        if self._is_speaking:
            self._is_speaking = False
            logger.debug("Speech ended")
            if self.on_speech_end:
                self.on_speech_end()

    def process_audio_frame(self, frame: rtc.AudioFrame):
        """Process audio frame for energy calculation."""
        energy = self._energy_calculator.calculate_energy(
            frame.data.tobytes() if hasattr(frame.data, 'tobytes') else bytes(frame.data),
            sample_width=2
        )

        if self.on_audio_frame:
            self.on_audio_frame(energy, self._is_speaking)

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
