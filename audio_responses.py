"""
Audio Responses Registry

Simple mapping of event_id -> (audio_file, transcript)

To add a new response:
1. Add entry to RESPONSES dict
2. Place WAV file in audio/ directory

Audio requirements:
- Format: WAV (LINEAR16)
- Sample rate: 24000 Hz (recommended) or 16000 Hz
- Channels: Mono (1 channel)
- Bit depth: 16-bit
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

AUDIO_DIR = Path("./AUDIO_DIR")


@dataclass(frozen=True)
class AudioResponse:
    """Single audio response with file and transcript."""
    file_name: str
    transcript: str

    @property
    def audio_path(self) -> Path:
        return AUDIO_DIR / self.file_name

    def exists(self) -> bool:
        return self.audio_path.exists()


# =============================================================================
# RESPONSES REGISTRY
# =============================================================================
# Add new responses here. Keep it simple.
#
# Format:
#   "EVENT_ID": AudioResponse("filename.wav", "transcript text")
# =============================================================================

RESPONSES: Dict[str, AudioResponse] = {
    "HOMEPAGE": AudioResponse(
        file_name="output.wav",
        transcript="""Namaste me Raahi, mai aapki trip create
        karne me madad kar sakti hu, aap apna pickup or drop city bataiye"""
    ),

}

DEFAULT_EVENT_ID = "HOMEPAGE"


def get_response(event_id: str) -> AudioResponse:
    """Get response for event_id, falls back to default."""
    return RESPONSES.get(event_id.upper(), RESPONSES[DEFAULT_EVENT_ID])


def get_transcript(event_id: str) -> str:
    """Get just the transcript for an event_id."""
    return get_response(event_id).transcript


def get_audio_path(event_id: str) -> Optional[Path]:
    """Get audio path if file exists, None otherwise."""
    response = get_response(event_id)
    return response.audio_path if response.exists() else None


def list_responses() -> Dict[str, dict]:
    """List all responses with availability status."""
    return {
        event_id: {
            "file": r.file_name,
            "exists": r.exists(),
            "transcript": r.transcript[:50] + "..."
        }
        for event_id, r in RESPONSES.items()
    }
