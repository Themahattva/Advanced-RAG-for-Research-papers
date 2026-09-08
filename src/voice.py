"""
src/voice.py — Audio transcription service for MedAssist using Groq Whisper API.

Provides fast, accurate multilingual speech-to-text (English, Hindi, and Hinglish)
using Groq's hosted Whisper models (whisper-large-v3-turbo / whisper-large-v3).
"""

import io
import os
from typing import Optional
import groq
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_WHISPER_MODEL = "whisper-large-v3-turbo"
AVAILABLE_WHISPER_MODELS = [
    "whisper-large-v3-turbo",
    "whisper-large-v3",
]

# Medical & consumer-health prompt guide for Whisper to optimize accuracy
# on symptoms, remedies, and colloquial English/Hinglish phrasing.
HEALTH_TRANSCRIPTION_PROMPT = (
    "MedAssist consumer health consultation in English, Hindi, or Hinglish: "
    "symptoms, illness, medical conditions, remedies, doctor consultation, "
    "bukhar, dard, khansi, blood pressure, diabetes, asthma, infection."
)


def get_groq_client() -> Optional[groq.Groq]:
    """Initialize and return a Groq client if GROQ_API_KEY is available."""
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return groq.Groq(api_key=api_key)


def is_voice_transcription_available() -> bool:
    """Check if the Groq API key is configured for transcription."""
    load_dotenv(override=True)
    return bool(os.getenv("GROQ_API_KEY"))


def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "recording.wav",
    model: str = DEFAULT_WHISPER_MODEL,
    prompt: Optional[str] = HEALTH_TRANSCRIPTION_PROMPT,
) -> str:
    """
    Transcribe audio bytes using Groq Whisper API.

    Args:
        audio_bytes: Raw audio bytes (typically WAV, MP3, M4A, or WEBM).
        filename: Optional filename indicator for the audio format.
        model: Whisper model name (default: whisper-large-v3-turbo).
        prompt: Contextual prompt to guide Whisper on vocabulary and spelling.

    Returns:
        Transcribed text as a clean string.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set or audio is empty.
        Exception: If Groq Whisper API call fails.
    """
    if not audio_bytes or len(audio_bytes) == 0:
        raise ValueError("Audio input is empty.")

    client = get_groq_client()
    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your .env file."
        )

    # Groq accepts file tuples in the format (filename, bytes_data)
    file_payload = (filename, audio_bytes)

    transcription = client.audio.transcriptions.create(
        file=file_payload,
        model=model,
        prompt=prompt,
        response_format="text",
    )

    result_text = str(transcription).strip()
    return result_text
