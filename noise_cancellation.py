"""
Noise Cancellation for Self-Hosted LiveKit

IMPORTANT: LiveKit's official noise cancellation plugin (livekit-plugins-noise-cancellation)
requires LiveKit Cloud and will NOT work with self-hosted setups.

This module provides:
1. noisereduce-based spectral gating (works locally)
2. Proper integration with LiveKit's audio pipeline via rtc.FrameProcessor
3. Energy-based activity detection for session monitoring

For self-hosted setups, this is your best option for noise reduction.
"""

import logging
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
from collections import deque
import asyncio

from livekit import rtc

logger = logging.getLogger("noise-cancellation")

NOISEREDUCE_AVAILABLE = False
SCIPY_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
    logger.info("✓ noisereduce available for noise cancellation")
except ImportError:
    logger.warning("✗ noisereduce not installed - limited noise cancellation")

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("✗ scipy not installed - no high-pass filtering")


@dataclass
class NoiseConfig:
    """Configuration for noise cancellation."""

    enabled: bool = True

    # Strength: 0.0 (no reduction) to 1.0 (aggressive)
    # Recommended: 0.5-0.7 for voice, 0.3-0.5 for noisy environments
    noise_reduction_strength: float = 0.6

    # True = assumes stationary background noise (fans, AC)
    # False = adapts to varying noise (better for real-world)
    stationary_noise: bool = False

    # Time constant for noise estimation (seconds)
    # Lower = faster adaptation, Higher = more stable
    time_constant_s: float = 0.4

    # Frequency smoothing for noise mask (Hz)
    freq_mask_smooth_hz: int = 500

    # Time smoothing for noise mask (ms)
    time_mask_smooth_ms: int = 50

    # Skip processing for very quiet frames (saves CPU)
    min_energy_threshold: float = 0.0005

    # High-pass filter to remove low-frequency rumble (Hz)
    # Set to 0 to disable
    highpass_cutoff_hz: int = 80

    # Apply soft noise gate
    noise_gate_threshold: float = 0.01
    noise_gate_attack_ms: float = 5.0
    noise_gate_release_ms: float = 50.0


class NoiseReduceProcessor:
    """
    Real-time noise reduction using spectral gating.

    This processor uses noisereduce library for effective
    background noise suppression.
    """

    def __init__(self, config: Optional[NoiseConfig] = None):
        self.config = config or NoiseConfig()
        self._sample_rate: int = 16000
        self._frames_processed: int = 0
        self._noise_frames_skipped: int = 0

        # High-pass filter state
        self._highpass_zi: Optional[np.ndarray] = None
        self._highpass_b: Optional[np.ndarray] = None
        self._highpass_a: Optional[np.ndarray] = None

        # Noise gate state
        self._gate_gain: float = 1.0

        # Energy tracking
        self._energy_history: deque = deque(maxlen=50)

    def _init_highpass_filter(self, sample_rate: int):
        """Initialize high-pass filter coefficients."""
        if not SCIPY_AVAILABLE or self.config.highpass_cutoff_hz <= 0:
            return

        try:
            nyquist = sample_rate / 2
            normalized_cutoff = self.config.highpass_cutoff_hz / nyquist
            if normalized_cutoff < 1.0:
                self._highpass_b, self._highpass_a = signal.butter(
                    2, normalized_cutoff, btype='high'
                )
                self._highpass_zi = signal.lfilter_zi(
                    self._highpass_b, self._highpass_a
                )
        except Exception as e:
            logger.warning(f"Failed to initialize high-pass filter: {e}")

    def process_frame(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> tuple[np.ndarray, float, bool]:
        """
        Process a single audio frame.

        Args:
            audio_data: Audio samples as int16 numpy array
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (processed_audio, energy, is_speech_likely)
        """
        if self._sample_rate != sample_rate:
            self._sample_rate = sample_rate
            self._init_highpass_filter(sample_rate)

        self._frames_processed += 1

        audio_float = audio_data.astype(np.float32) / 32768.0

        # Calculate energy
        energy = float(np.sqrt(np.mean(audio_float ** 2)))
        self._energy_history.append(energy)

        # Skip near-silent frames
        if energy < self.config.min_energy_threshold:
            self._noise_frames_skipped += 1
            return audio_data, energy, False

        if not self.config.enabled:
            return audio_data, energy, energy > self.config.noise_gate_threshold

        processed = audio_float.copy()

        # 1. Apply high-pass filter (remove rumble)
        if self._highpass_b is not None and self._highpass_zi is not None:
            try:
                processed, self._highpass_zi = signal.lfilter(
                    self._highpass_b, self._highpass_a,
                    processed,
                    zi=self._highpass_zi *
                    processed[0] if len(processed) > 0 else self._highpass_zi
                )
            except Exception:
                pass  # Skip filter on error

        # 2. Apply noisereduce spectral gating
        if NOISEREDUCE_AVAILABLE:
            try:
                processed = nr.reduce_noise(
                    y=processed,
                    sr=sample_rate,
                    stationary=self.config.stationary_noise,
                    prop_decrease=self.config.noise_reduction_strength,
                    time_constant_s=self.config.time_constant_s,
                    freq_mask_smooth_hz=self.config.freq_mask_smooth_hz,
                    time_mask_smooth_ms=self.config.time_mask_smooth_ms,
                )
            except Exception as e:
                logger.debug(f"noisereduce error (using original): {e}")

        # 3. Apply soft noise gate
        processed_energy = float(np.sqrt(np.mean(processed ** 2)))
        is_speech = processed_energy > self.config.noise_gate_threshold

        if not is_speech:
            # Gradually reduce gain
            attack_coef = 1.0 - \
                np.exp(-1.0 / (sample_rate * self.config.noise_gate_attack_ms / 1000))
            self._gate_gain = max(0.0, self._gate_gain - attack_coef)
        else:
            # Gradually restore gain
            release_coef = 1.0 - \
                np.exp(-1.0 / (sample_rate * self.config.noise_gate_release_ms / 1000))
            self._gate_gain = min(1.0, self._gate_gain + release_coef)

        processed = processed * self._gate_gain

        # Convert back to int16
        output = (processed * 32768.0).clip(-32768, 32767).astype(np.int16)

        return output, energy, is_speech

    def get_average_energy(self) -> float:
        """Get average energy over recent frames."""
        if not self._energy_history:
            return 0.0
        return sum(self._energy_history) / len(self._energy_history)

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "frames_processed": self._frames_processed,
            "frames_skipped": self._noise_frames_skipped,
            "skip_ratio": self._noise_frames_skipped / max(1, self._frames_processed),
            "average_energy": self.get_average_energy(),
            "sample_rate": self._sample_rate,
        }


class LiveKitFrameProcessor:
    """
    Implements rtc.FrameProcessor interface for LiveKit integration.

    This can be passed to RoomInputOptions.noise_cancellation parameter
    as a custom frame processor.

    Usage:
        processor = LiveKitFrameProcessor(NoiseConfig())

        await session.start(
            room=room,
            agent=agent,
            room_input_options=room_io.RoomInputOptions(
                noise_cancellation=processor,
            ),
        )
    """

    def __init__(
        self,
        config: Optional[NoiseConfig] = None,
        on_audio_metrics: Optional[Callable[[float, bool], None]] = None,
    ):
        """
        Args:
            config: Noise cancellation configuration
            on_audio_metrics: Callback(energy, is_speech) for each frame
        """
        self.config = config or NoiseConfig()
        self.on_audio_metrics = on_audio_metrics
        self._processor = NoiseReduceProcessor(self.config)

    async def process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        """
        Process an audio frame (implements FrameProcessor interface).

        This method is called by LiveKit for each incoming audio frame.
        """
        try:
            # Extract audio data
            audio_data = np.frombuffer(
                frame.data.tobytes() if hasattr(frame.data, 'tobytes') else bytes(frame.data),
                dtype=np.int16
            )

            # Process the audio
            processed, energy, is_speech = self._processor.process_frame(
                audio_data,
                frame.sample_rate
            )

            # Notify listener
            if self.on_audio_metrics:
                self.on_audio_metrics(energy, is_speech)

            # Create new frame with processed audio
            return rtc.AudioFrame(
                data=processed.tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                samples_per_channel=frame.samples_per_channel,
            )

        except Exception as e:
            logger.warning(f"Frame processing error: {e}")
            return frame  # Return original on error

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return self._processor.get_stats()


class AudioMonitoringProcessor:
    """
    Audio processor that provides monitoring without heavy noise reduction.

    Use this when you want energy/speech detection for session monitoring
    but don't want to apply aggressive noise reduction (which can affect
    STT quality in some cases).
    """

    def __init__(
        self,
        energy_threshold: float = 0.02,
        smoothing_factor: float = 0.3,
        on_audio_metrics: Optional[Callable[[float, bool], None]] = None,
    ):
        self.energy_threshold = energy_threshold
        self.smoothing_factor = smoothing_factor
        self.on_audio_metrics = on_audio_metrics

        self._smoothed_energy = 0.0
        self._frames_processed = 0
        self._speech_frames = 0

    def process_frame(self, audio_data: bytes, sample_width: int = 2) -> tuple[float, bool]:
        """
        Calculate energy and detect speech from raw audio.

        Args:
            audio_data: Raw audio bytes
            sample_width: Bytes per sample (2 for 16-bit)

        Returns:
            Tuple of (smoothed_energy, is_speech)
        """
        try:
            if not audio_data:
                return 0.0, False

            # Convert to numpy
            if sample_width == 2:
                samples = np.frombuffer(audio_data, dtype=np.int16)
            else:
                samples = np.frombuffer(audio_data, dtype=np.int8)

            if len(samples) == 0:
                return 0.0, False

            # Normalize and calculate RMS
            max_val = 32768.0 if sample_width == 2 else 128.0
            normalized = samples.astype(np.float32) / max_val
            rms = float(np.sqrt(np.mean(normalized ** 2)))

            # Apply smoothing
            self._smoothed_energy = (
                self.smoothing_factor * rms +
                (1 - self.smoothing_factor) * self._smoothed_energy
            )

            self._frames_processed += 1

            is_speech = self._smoothed_energy > self.energy_threshold
            if is_speech:
                self._speech_frames += 1

            # Notify listener
            if self.on_audio_metrics:
                self.on_audio_metrics(self._smoothed_energy, is_speech)

            return self._smoothed_energy, is_speech

        except Exception as e:
            logger.warning(f"Energy calculation error: {e}")
            return 0.0, False

    def reset(self):
        """Reset processor state."""
        self._smoothed_energy = 0.0
        self._frames_processed = 0
        self._speech_frames = 0

    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        return {
            "frames_processed": self._frames_processed,
            "speech_frames": self._speech_frames,
            "speech_ratio": self._speech_frames / max(1, self._frames_processed),
            "current_energy": self._smoothed_energy,
        }


# =============================================================================
# Integration functions for main.py
# =============================================================================

def create_noise_cancellation_processor(
    config: Optional[NoiseConfig] = None,
    on_audio_metrics: Optional[Callable[[float, bool], None]] = None,
) -> Optional[LiveKitFrameProcessor]:
    """
    Create a noise cancellation processor for LiveKit.

    Returns None if noise reduction is not available.

    Usage in main.py:
        from noise_cancellation_fixed import create_noise_cancellation_processor

        nc_processor = create_noise_cancellation_processor(
            on_audio_metrics=assistant.on_audio_frame
        )

        await session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=room_io.RoomInputOptions(
                noise_cancellation=nc_processor,  # Pass as frame processor
            ),
        )
    """
    if not NOISEREDUCE_AVAILABLE:
        logger.warning(
            "noisereduce not available. Install with: pip install noisereduce"
        )
        return None

    return LiveKitFrameProcessor(config, on_audio_metrics)


def diagnose_setup() -> dict:
    """
    Diagnose the noise cancellation setup.

    Call this at startup to verify everything is working.
    """
    status = {
        "noisereduce_available": NOISEREDUCE_AVAILABLE,
        "scipy_available": SCIPY_AVAILABLE,
        "can_do_noise_reduction": NOISEREDUCE_AVAILABLE,
        "can_do_highpass_filter": SCIPY_AVAILABLE,
        "recommendations": [],
    }

    if not NOISEREDUCE_AVAILABLE:
        status["recommendations"].append(
            "Install noisereduce for noise cancellation: pip install noisereduce"
        )

    if not SCIPY_AVAILABLE:
        status["recommendations"].append(
            "Install scipy for high-pass filtering: pip install scipy"
        )

    if NOISEREDUCE_AVAILABLE:
        status["recommendations"].append(
            "✓ Noise cancellation is ready to use"
        )

    # Note about LiveKit Cloud
    status["notes"] = [
        "LiveKit's official noise cancellation plugin requires LiveKit Cloud.",
        "This implementation uses noisereduce for self-hosted setups.",
        "For best results, also enable noise/echo cancellation in frontend WebRTC settings.",
    ]

    return status
