import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, AsyncIterator
from pathlib import Path

logger = logging.getLogger("raahi-greetings")

# Directory where audio files are stored
AUDIO_DIR = Path(
    "/app/audio") if os.path.exists("/app/audio") else Path("./audio")


@dataclass
class Greeting:
    """Represents a single greeting with audio file and transcript."""
    id: str
    file_name: str
    transcript: str
    description: str = ""

    @property
    def audio_path(self) -> Path:
        """Get the full path to the audio file."""
        return AUDIO_DIR / self.file_name

    def audio_exists(self) -> bool:
        """Check if the audio file exists."""
        return self.audio_path.exists()


# ============================================================================
# GREETINGS REGISTRY - Add new greetings here
# ============================================================================
# Format: "GREETING_ID": Greeting(id, file_name, transcript, description)
#
# To add a new greeting:
# 1. Record/generate the audio file
# 2. Place it in the AUDIO_DIR directory
# 3. Add an entry below with unique ID
# ============================================================================

GREETINGS: Dict[str, Greeting] = {
    "HOMEPAGE": Greeting(
        id="HOMEPAGE",
        file_name="greeting_homepage.wav",
        transcript="""Namaste! Main Raahi hoon, Cabswale ki taraf se.
        Main aapki trip create karne mein madad kar sakti hoon.
        Aap apna pickup aur drop city bataiye.""",
        description="Default homepage greeting"
    ),

}


class GreetingsManager:
    """
    Manages greeting playback with audio file loading and TTS fallback.
    """

    def __init__(self, audio_dir: Optional[Path] = None):
        """
        Initialize the greetings manager.

        Args:
            audio_dir: Custom directory for audio files (optional)
        """
        self.audio_dir = audio_dir or AUDIO_DIR
        self._ensure_audio_dir()
        self._log_available_greetings()

    def _ensure_audio_dir(self):
        """Create audio directory if it doesn't exist."""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Audio directory: {self.audio_dir}")

    def _log_available_greetings(self):
        """Log which greetings have audio files available."""
        available = []
        missing = []

        for greeting_id, greeting in GREETINGS.items():
            if greeting.audio_exists():
                available.append(greeting_id)
            else:
                missing.append(greeting_id)

        logger.info(f"Greetings with audio: {available}")
        if missing:
            logger.warning(
                f"Greetings missing audio (will use TTS): {missing}")

    def get_greeting(self, greeting_id: str) -> Optional[Greeting]:
        """
        Get a greeting by its ID.

        Args:
            greeting_id: The unique identifier for the greeting

        Returns:
            Greeting object or None if not found
        """
        greeting = GREETINGS.get(greeting_id.upper())
        if not greeting:
            logger.warning(f"Unknown greeting ID: {greeting_id}")
        return greeting

    def get_transcript(self, greeting_id: str) -> Optional[str]:
        """
        Get just the transcript for a greeting.

        Args:
            greeting_id: The unique identifier for the greeting

        Returns:
            Transcript string or None if not found
        """
        greeting = self.get_greeting(greeting_id)
        return greeting.transcript if greeting else None

    def get_audio_path(self, greeting_id: str) -> Optional[Path]:
        """
        Get the audio file path for a greeting if it exists.

        Args:
            greeting_id: The unique identifier for the greeting

        Returns:
            Path to audio file or None if not available
        """
        greeting = self.get_greeting(greeting_id)
        if greeting and greeting.audio_exists():
            return greeting.audio_path
        return None

    def list_greetings(self) -> Dict[str, dict]:
        """
        List all available greetings with their status.

        Returns:
            Dictionary of greeting info with availability status
        """
        result = {}
        for greeting_id, greeting in GREETINGS.items():
            result[greeting_id] = {
                "description": greeting.description,
                "transcript": greeting.transcript,
                "audio_available": greeting.audio_exists(),
                "file_name": greeting.file_name
            }
        return result

    @staticmethod
    def get_default_greeting_id() -> str:
        """Get the default greeting ID (HOMEPAGE)."""
        return "HOMEPAGE"


_manager: Optional[GreetingsManager] = None


def get_greetings_manager() -> GreetingsManager:
    """Get the singleton GreetingsManager instance."""
    global _manager
    if _manager is None:
        _manager = GreetingsManager()
    return _manager


def get_greeting_for_event(greeting_id: str) -> tuple[Optional[str], Optional[Path]]:
    """
    Convenience function to get transcript and audio path for a greeting.

    Args:
        greeting_id: The greeting identifier from client event

    Returns:
        Tuple of (transcript, audio_path) - audio_path may be None
    """
    manager = get_greetings_manager()
    greeting = manager.get_greeting(greeting_id)

    if not greeting:
        # Return default greeting if unknown ID
        greeting = manager.get_greeting("HOMEPAGE")

    if greeting:
        audio_path = greeting.audio_path if greeting.audio_exists() else None
        return greeting.transcript, audio_path

    return None, None


async def load_audio_frames(audio_path: Path) -> Optional[AsyncIterator]:
    """
    Load audio file as AsyncIterator of AudioFrames for LiveKit.

    This uses wave module for WAV files (fast) or falls back to
    reading raw bytes for other formats.

    Args:
        audio_path: Path to the audio file

    Returns:
        AsyncIterator of rtc.AudioFrame or None if loading fails
    """

    try:
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return None

        if audio_path.suffix.lower() == '.wav':
            return _load_wav_frames(audio_path)
        else:
            logger.info(f"Loading non-WAV audio: {audio_path}")
            return str(audio_path)

    except Exception as e:
        logger.error(f"Failed to load audio file {audio_path}: {e}")
        return None


async def _load_wav_frames(audio_path: Path):
    """
    Generator that yields AudioFrames from a WAV file.

    Args:
        audio_path: Path to WAV file

    Yields:
        rtc.AudioFrame objects
    """
    import wave
    from livekit import rtc

    try:
        with wave.open(str(audio_path), 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()

            if sample_width != 2:
                logger.warning(
                    f"Expected 16-bit audio, got {sample_width * 8}-bit")

            samples_per_frame = sample_rate // 100  # 10ms

            while True:
                data = wav_file.readframes(samples_per_frame)
                if not data:
                    break

                yield rtc.AudioFrame(
                    data=data,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                    samples_per_channel=len(
                        data) // (num_channels * sample_width)
                )

    except Exception as e:
        logger.error(f"Error reading WAV file {audio_path}: {e}")


class GreetingEvent:
    """Schema for greeting events from client."""

    EVENT_NAME = "initial_greeting"

    def __init__(self, greeting_id: str, metadata: Optional[dict] = None):
        self.greeting_id = greeting_id.upper()
        self.metadata = metadata or {}

    @classmethod
    def from_dict(cls, data: dict) -> Optional["GreetingEvent"]:
        """Parse a greeting event from dictionary data."""
        if data.get("event") != cls.EVENT_NAME:
            return None

        greeting_id = data.get("greeting_id") or data.get("id")
        if not greeting_id:
            return None

        return cls(
            greeting_id=greeting_id,
            metadata=data.get("metadata", {})
        )
