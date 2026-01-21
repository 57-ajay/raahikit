"""
Audio Responses Registry

Simple mapping of event_id -> (audio_file, transcript)

Audio requirements:
- Format: WAV (LINEAR16)
- Sample rate: 24000 Hz (recommended)
- Channels: Mono
- Bit depth: 16-bit
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

AUDIO_DIR = Path("./AUDIO_DIR")


@dataclass(frozen=True)
class AudioResponse:
    """Audio response with file and transcript."""
    file_name: str
    transcript: str

    @property
    def audio_path(self) -> Path:
        return AUDIO_DIR / self.file_name

    def exists(self) -> bool:
        return self.audio_path.exists()


RESPONSES: Dict[str, AudioResponse] = {
    "HOMEPAGE": AudioResponse(
        file_name="output.wav",
        transcript="Namaste me Raahi, mai aapki trip create karne me madad kar sakti hu, aap apna pickup or drop city bataiye"
    ),
}

DEFAULT_EVENT_ID = "HOMEPAGE"


def get_response(event_id: str) -> AudioResponse:
    """Get response for event_id, falls back to default."""
    return RESPONSES.get(event_id.upper(), RESPONSES[DEFAULT_EVENT_ID])


def get_transcript(event_id: str) -> str:
    """Get transcript for an event_id."""
    return get_response(event_id).transcript


def get_audio_path(event_id: str) -> Optional[Path]:
    """Get audio path if file exists."""
    response = get_response(event_id)
    return response.audio_path if response.exists() else None
