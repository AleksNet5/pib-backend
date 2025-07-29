"""Gemini chat provider using the async Google Generative AI API."""

from __future__ import annotations

from typing import AsyncIterable, Iterable, List, Optional

from public_api_client.public_voice_client import PublicApiChatMessage

from ..gemini_chat_async import GeminiLiveSession
from .base import ChatProvider


_sessions: dict[str, GeminiLiveSession] = {}


class GeminiChat(ChatProvider):
    """Chat provider that maintains a live session per chat id."""

    async def chat_completion(
        self,
        *,
        text: str,
        description: str,
        message_history: List[PublicApiChatMessage],
        public_api_token: str,  # unused but kept for API compatibility
        image_base64: Optional[str] = None,
        model: str,
        chat_id: str,
        audio_stream: Optional[Iterable[bytes]] = None,
    ) -> AsyncIterable[str]:
        session = _sessions.get(chat_id)
        if session is None:
            session = GeminiLiveSession(
                description=description,
                message_history=message_history,
                image_base64=image_base64,
                model=model,
            )
            _sessions[chat_id] = session
        async for token in session.ask(text, audio_stream=audio_stream):
            yield token
