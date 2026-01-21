"""
Audio Player - Simple WAV file streaming

Reads WAV files and yields rtc.AudioFrame for LiveKit.

Requirements for WAV files:
- Format: LINEAR16 (PCM signed 16-bit)
- Sample rate: 24000 Hz recommended (or 16000, 48000)
- Channels: Mono (1 channel)
"""

import wave
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional
from livekit import rtc

logger = logging.getLogger("audio-player")

FRAME_DURATION_MS = 10


async def stream_wav_file(file_path: Path) -> AsyncGenerator[rtc.AudioFrame, None]:
    """
    Stream a WAV file as AudioFrames.

    Args:
        file_path: Path to WAV file

    Yields:
        rtc.AudioFrame objects
    """
    try:
        with wave.open(str(file_path), 'rb') as wav:
            sample_rate = wav.getframerate()
            num_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()

            if sample_width != 2:
                logger.error(f"WAV must be 16-bit, got {sample_width * 8}-bit")
                return

            samples_per_frame = (sample_rate * FRAME_DURATION_MS) // 1000

            logger.debug(f"""Streaming WAV: {sample_rate}Hz, {
                         num_channels}ch, {samples_per_frame} samples/frame""")

            while True:
                data = wav.readframes(samples_per_frame)
                if not data:
                    break

                actual_samples = len(data) // (num_channels * sample_width)
                if actual_samples == 0:
                    break

                yield rtc.AudioFrame(
                    data=data,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                    samples_per_channel=actual_samples,
                )

    except FileNotFoundError:
        logger.error(f"WAV file not found: {file_path}")
    except wave.Error as e:
        logger.error(f"Invalid WAV file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error streaming WAV {file_path}: {e}")


def validate_wav_file(file_path: Path) -> Optional[dict]:
    """
    Validate a WAV file and return its properties.

    Returns:
        Dict with file properties or None if invalid
    """
    try:
        with wave.open(str(file_path), 'rb') as wav:
            return {
                "sample_rate": wav.getframerate(),
                "channels": wav.getnchannels(),
                "sample_width": wav.getsampwidth(),
                "frames": wav.getnframes(),
                "duration_seconds": wav.getnframes() / wav.getframerate(),
            }
    except Exception as e:
        logger.error(f"Invalid WAV file {file_path}: {e}")
        return None


def get_recommended_format() -> str:
    """Return recommended WAV format string."""
    return "LINEAR16, 24000Hz, Mono (1 channel), 16-bit"
