import logging
import asyncio
import numpy as np
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, field
from collections import deque
import time

logger = logging.getLogger("noise-cancellation")

# Check for noisereduce availability
NOISEREDUCE_AVAILABLE = False
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
    logger.info("noisereduce library available for noise analysis")
except ImportError:
    logger.warning("noisereduce not installed - using energy-only analysis")


@dataclass
class NoiseCancellationConfig:
    """
    Configuration for noise cancellation and session management.

    Tuning guide for noisy environments:
    - Increase timeouts (silence_timeout, noise_flood_timeout) for more tolerance
    - Decrease speech_energy_threshold if missing soft speech
    - Increase min_snr_for_speech if getting false positives from noise
    """

    enabled: bool = True

    # === Noise Reduction Parameters ===
    # Strength (0.0-1.0): Higher = more aggressive noise removal
    # 0.4-0.6 recommended for voice to avoid distortion
    noise_reduction_strength: float = 0.5

    # Stationary noise model: True for consistent noise (AC, fans)
    # False for varying noise (traffic, crowd) - generally False is better
    stationary_noise: bool = False

    # Time constant for noise estimation (seconds)
    # Lower = faster adaptation to changing noise
    time_constant_s: float = 0.4

    # Frequency smoothing (Hz): Higher preserves more speech harmonics
    freq_mask_smooth_hz: int = 500

    # Time smoothing (ms): Higher = smoother transitions, fewer artifacts
    time_mask_smooth_ms: int = 50

    # === Session Timeout Settings ===
    # These are GENEROUS to avoid false pauses
    silence_timeout_seconds: float = 45.0   # Complete silence before pause
    noise_flood_timeout_seconds: float = 30.0  # High noise, no STT before pause

    # === VAD / Speech Detection ===
    # Energy threshold for considering audio as potential speech
    speech_energy_threshold: float = 0.02

    # Minimum energy to register as any audio activity
    min_audio_energy: float = 0.003

    # Minimum SNR to consider signal as speech (not noise)
    # Higher = more strict, fewer false positives from noise
    min_snr_for_speech: float = 2.5

    # === Grace Periods ===
    # Don't trigger pause during these periods
    agent_speaking_grace_seconds: float = 8.0   # After agent finishes speaking
    post_stt_grace_seconds: float = 15.0        # After receiving valid STT

    # === Audio Processing ===
    buffer_frames: int = 8  # Frames to buffer before processing
    sample_rate: int = 16000


@dataclass
class AudioMetrics:
    """
    Tracks audio quality metrics with adaptive noise floor estimation.

    Uses percentile-based noise floor estimation which is more robust
    than simple averages in varying noise conditions.
    """

    # Timestamps for timeout tracking
    last_speech_time: float = field(default_factory=time.time)
    last_valid_stt_time: float = field(default_factory=time.time)
    last_agent_speech_time: float = field(default_factory=time.time)

    # Energy history for analysis
    energy_history: deque = field(default_factory=lambda: deque(maxlen=100))
    noise_floor_samples: deque = field(
        default_factory=lambda: deque(maxlen=300))

    # Adaptive noise floor
    noise_floor: float = 0.01

    # Exponentially smoothed energy for stable VAD
    smoothed_energy: float = 0.0
    smoothing_alpha: float = 0.25  # Lower = more stable, higher = more responsive

    # Counters
    frames_processed: int = 0
    speech_frames: int = 0
    noise_reduced_frames: int = 0

    def update_energy(self, energy: float, config: NoiseCancellationConfig) -> bool:
        """
        Update metrics with new energy reading.

        Uses adaptive threshold based on estimated noise floor.
        Returns True if this frame looks like speech.
        """
        self.energy_history.append(energy)
        self.noise_floor_samples.append(energy)
        self.frames_processed += 1

        # Exponential smoothing for stability
        self.smoothed_energy = (
            self.smoothing_alpha * energy +
            (1 - self.smoothing_alpha) * self.smoothed_energy
        )

        # Update noise floor estimate using 15th percentile
        # This is robust to speech spikes while tracking actual noise level
        if len(self.noise_floor_samples) >= 50:
            sorted_samples = sorted(self.noise_floor_samples)
            percentile_idx = int(len(sorted_samples) * 0.15)
            self.noise_floor = max(sorted_samples[percentile_idx], 0.002)

        # Determine if this is speech using SNR-based adaptive threshold
        snr = self.get_snr()
        is_speech = (
            snr >= config.min_snr_for_speech and
            self.smoothed_energy > config.speech_energy_threshold
        )

        if is_speech:
            self.last_speech_time = time.time()
            self.speech_frames += 1

        return is_speech

    def get_snr(self) -> float:
        """Get current signal-to-noise ratio estimate."""
        if self.noise_floor < 0.001:
            return 0.0
        return self.smoothed_energy / self.noise_floor

    def get_average_energy(self) -> float:
        """Get average energy over recent history."""
        if not self.energy_history:
            return 0.0
        return sum(self.energy_history) / len(self.energy_history)

    def on_stt_result(self):
        """Called when STT produces a valid result - resets timers."""
        now = time.time()
        self.last_valid_stt_time = now
        self.last_speech_time = now

    def on_agent_speech(self):
        """Called when agent speaks - extends grace period."""
        self.last_agent_speech_time = time.time()


class NoiseReduceProcessor:
    """
    Spectral gating noise reduction with audio buffering.

    Buffers audio frames for better spectral analysis and
    learns noise profile from quiet periods.
    """

    def __init__(
        self,
        strength: float = 0.5,
        stationary: bool = False,
        time_constant: float = 0.4,
        freq_smooth: int = 500,
        time_smooth: int = 50,
        buffer_frames: int = 8,
    ):
        self.strength = strength
        self.stationary = stationary
        self.time_constant = time_constant
        self.freq_smooth = freq_smooth
        self.time_smooth = time_smooth
        self.buffer_frames = buffer_frames

        self._audio_buffer: list = []
        self._sample_rate = 16000

        # Noise profile learning
        self._noise_profile: Optional[np.ndarray] = None
        self._noise_samples: list = []
        self._noise_samples_max = 50
        self._is_learning_noise = True
        self._frames_since_speech = 0
        self._speech_threshold = 0.02

    def process_frame(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> tuple[np.ndarray, float]:
        """
        Process audio frame with noise reduction.

        Returns: (processed_audio, energy)
        """
        self._sample_rate = sample_rate

        if not NOISEREDUCE_AVAILABLE:
            return audio_data, self._calculate_energy(audio_data)

        try:
            # Convert to float32
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)

            energy = float(np.sqrt(np.mean(audio_float ** 2)))

            # Skip very quiet audio
            if energy < 0.001:
                return audio_data, energy

            # Buffer audio for better processing
            self._audio_buffer.append(audio_float)

            # Learn noise profile from quiet periods
            if self._is_learning_noise and energy < self._speech_threshold:
                self._noise_samples.append(audio_float)
                if len(self._noise_samples) >= self._noise_samples_max:
                    self._build_noise_profile()

            # Track frames since speech for profile updates
            if energy > self._speech_threshold:
                self._frames_since_speech = 0
            else:
                self._frames_since_speech += 1
                if self._frames_since_speech > 100 and not self._is_learning_noise:
                    self._noise_samples = []
                    self._is_learning_noise = True

            # Process when buffer is ready
            if len(self._audio_buffer) >= self.buffer_frames:
                buffered = np.concatenate(self._audio_buffer)

                reduced = nr.reduce_noise(
                    y=buffered,
                    sr=sample_rate,
                    stationary=self.stationary,
                    prop_decrease=self.strength,
                    time_constant_s=self.time_constant,
                    freq_mask_smooth_hz=self.freq_smooth,
                    time_mask_smooth_ms=self.time_smooth,
                    y_noise=self._noise_profile,
                    n_fft=512,
                    hop_length=128,
                )

                # Get most recent frame from processed buffer
                frame_size = len(audio_float)
                processed = reduced[-frame_size:]

                # Keep some overlap for continuity
                self._audio_buffer = self._audio_buffer[-(
                    self.buffer_frames // 2):]

                # Convert back to int16
                output = (processed * 32768.0).clip(-32768,
                                                    32767).astype(np.int16)
                post_energy = float(np.sqrt(np.mean(processed ** 2)))

                return output, post_energy
            else:
                return audio_data, energy

        except Exception as e:
            logger.warning(f"Noise reduction error: {e}")
            return audio_data, self._calculate_energy(audio_data)

    def _build_noise_profile(self):
        """Build noise profile from collected quiet samples."""
        if not self._noise_samples:
            return
        try:
            self._noise_profile = np.concatenate(self._noise_samples)
            self._is_learning_noise = False
            logger.info(f"""Built noise profile from {
                        len(self._noise_samples)} frames""")
        except Exception as e:
            logger.warning(f"Failed to build noise profile: {e}")

    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy."""
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        return float(np.sqrt(np.mean(audio_float ** 2)))

    def reset_noise_profile(self):
        """Reset to re-learn noise profile."""
        self._noise_profile = None
        self._noise_samples = []
        self._is_learning_noise = True
        logger.info("Noise profile reset")


class NoiseCancellationManager:
    """
    Main manager for noise analysis and session timeout detection.

    Key features:
    - Processes audio for energy/SNR analysis
    - Monitors for timeout conditions
    - Respects grace periods to avoid false pauses
    - Uses SNR to distinguish speech from noise
    """

    def __init__(
        self,
        config: Optional[NoiseCancellationConfig] = None,
        on_timeout: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.config = config or NoiseCancellationConfig()
        self.on_timeout = on_timeout
        self.metrics = AudioMetrics()

        # Initialize processor if available
        if NOISEREDUCE_AVAILABLE and self.config.enabled:
            self._processor = NoiseReduceProcessor(
                strength=self.config.noise_reduction_strength,
                stationary=self.config.stationary_noise,
                time_constant=self.config.time_constant_s,
                freq_smooth=self.config.freq_mask_smooth_hz,
                time_smooth=self.config.time_mask_smooth_ms,
                buffer_frames=self.config.buffer_frames,
            )
            logger.info(
                f"""NC Manager initialized: strength={
                    self.config.noise_reduction_strength}, """
                f"silence_timeout={self.config.silence_timeout_seconds}s, "
                f"""noise_flood_timeout={
                    self.config.noise_flood_timeout_seconds}s"""
            )
        else:
            self._processor = None
            logger.info("NC Manager initialized (energy-only mode)")

        self._running = False
        self._paused = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the session monitor."""
        self._running = True
        self._paused = False
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
        self._log_stats()

    def _log_stats(self):
        """Log session statistics."""
        m = self.metrics
        if m.frames_processed > 0:
            speech_pct = (m.speech_frames / m.frames_processed) * 100
            logger.info(
                f"NC Session complete: {m.frames_processed} frames, "
                f"""{speech_pct:.1f}% speech, avg_energy={
                    m.get_average_energy():.4f}, """
                f"noise_floor={m.noise_floor:.4f}, final_snr={m.get_snr():.2f}"
            )

    def pause(self):
        """Pause monitoring (called externally when session pauses)."""
        self._paused = True

    def resume(self):
        """Resume monitoring (called externally when session resumes)."""
        self._paused = False
        now = time.time()
        self.metrics.last_valid_stt_time = now
        self.metrics.last_speech_time = now
        self.metrics.last_agent_speech_time = now

    def process_audio_frame(
        self,
        frame_data: np.ndarray,
        sample_rate: int = 16000
    ) -> tuple[np.ndarray, float]:
        """
        Process audio frame and return (processed_audio, energy).

        The processed audio can be used for analysis/logging.
        Energy is used for VAD and timeout detection.
        """
        if not self.config.enabled or self._processor is None:
            energy = self._calculate_energy(frame_data)
            self.metrics.update_energy(energy, self.config)
            return frame_data, energy

        processed, energy = self._processor.process_frame(
            frame_data, sample_rate)
        self.metrics.update_energy(energy, self.config)
        self.metrics.noise_reduced_frames += 1

        return processed, energy

    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy from audio data."""
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        return float(np.sqrt(np.mean(audio_float ** 2)))

    def on_stt_result(self, transcript: str):
        """Called when STT produces a result."""
        if transcript and len(transcript.strip()) > 1:
            self.metrics.on_stt_result()
            logger.debug(f"Valid STT received: '{transcript[:40]}...'")

    def on_agent_speech(self):
        """Called when agent speaks."""
        self.metrics.on_agent_speech()

    def reset_noise_profile(self):
        """Force re-learning of noise profile."""
        if self._processor:
            self._processor.reset_noise_profile()

    async def _monitor_loop(self):
        """Monitor for timeout conditions."""
        while self._running:
            try:
                await asyncio.sleep(2.0)

                if self._paused:
                    continue

                timeout = self._check_timeout()
                if timeout:
                    reason, message = timeout
                    logger.warning(f"NC timeout: {reason} - {message}")
                    if self.on_timeout:
                        await self.on_timeout(reason)
                    # Don't break - let external handler manage pause state

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"NC monitor error: {e}")

    def _check_timeout(self) -> Optional[tuple[str, str]]:
        """
        Check for timeout conditions.

        Returns (reason, message) if should pause, None otherwise.
        """
        now = time.time()
        m = self.metrics
        cfg = self.config

        # === Grace Period Checks ===
        # Don't pause right after agent speaks
        agent_silence = now - m.last_agent_speech_time
        if agent_silence < cfg.agent_speaking_grace_seconds:
            return None

        # Don't pause right after valid STT (conversation is active)
        stt_silence = now - m.last_valid_stt_time
        if stt_silence < cfg.post_stt_grace_seconds:
            return None

        # === Timeout Checks ===

        # 1. Complete silence (no audio energy at all)
        audio_silence = now - m.last_speech_time
        if audio_silence >= cfg.silence_timeout_seconds:
            return (
                "silence_timeout",
                f"No audio activity for {audio_silence:.1f}s"
            )

        # 2. Noise flood (high energy but no valid STT = too noisy)
        avg_energy = m.get_average_energy()
        snr = m.get_snr()

        # Only trigger if:
        # - Significant energy (something is happening)
        # - Low SNR (energy is noise, not speech)
        # - No STT success for extended period
        if (avg_energy > cfg.speech_energy_threshold and
            snr < cfg.min_snr_for_speech and
                stt_silence >= cfg.noise_flood_timeout_seconds):
            return (
                "noise_flood",
                f"High noise (energy={avg_energy:.3f}, SNR={snr:.1f}) "
                f"with no valid STT for {stt_silence:.1f}s"
            )

        return None

    def get_stats(self) -> dict:
        """Get current statistics for debugging/monitoring."""
        now = time.time()
        m = self.metrics
        return {
            "frames_processed": m.frames_processed,
            "speech_frames": m.speech_frames,
            "noise_reduced_frames": m.noise_reduced_frames,
            "smoothed_energy": m.smoothed_energy,
            "noise_floor": m.noise_floor,
            "snr": m.get_snr(),
            "avg_energy": m.get_average_energy(),
            "silence_duration": now - m.last_speech_time,
            "stt_silence_duration": now - m.last_valid_stt_time,
            "agent_silence_duration": now - m.last_agent_speech_time,
            "is_paused": self._paused,
        }


# =============================================================================
# RECOMMENDED: CLIENT-SIDE NOISE SUPPRESSION
# =============================================================================
"""
The most effective solution for noisy environments is client-side noise
suppression. Here's how to enable it in LiveKit's JavaScript SDK:

```javascript
// Option 1: When creating local audio track
import { createLocalAudioTrack } from 'livekit-client';

const audioTrack = await createLocalAudioTrack({
    noiseSuppression: true,      // WebRTC noise suppression
    echoCancellation: true,      // Echo cancellation
    autoGainControl: true,       // Auto gain control
    // Optional: Use Krisp noise suppression (better quality, requires LiveKit Cloud)
    // processor: { name: 'krisp-noise-filter' }
});

// Option 2: When enabling microphone
await room.localParticipant.setMicrophoneEnabled(true, {
    noiseSuppression: true,
    echoCancellation: true,
    autoGainControl: true,
});

// Option 3: Update existing track
const audioTrack = room.localParticipant.getTrack(Track.Source.Microphone);
if (audioTrack) {
    await audioTrack.setAudioPreferences({
        noiseSuppression: true,
        echoCancellation: true,
    });
}
```

Benefits of client-side noise suppression:
1. Clean audio enters the pipeline from the start
2. STT receives clean audio directly
3. No server-side processing overhead
4. WebRTC's noise suppression is well-optimized
5. Works with any STT provider
"""
