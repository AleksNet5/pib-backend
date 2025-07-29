from __future__ import annotations
import os
import asyncio
from typing import AsyncIterable, Iterable, List, Optional

import google as genai
from google import ChatMessage, ChatRole, Blob
from public_api_client.public_voice_client import PublicApiChatMessage

GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"


def _configure_api_key() -> None:
    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{GOOGLE_API_KEY_ENV} not set")
    genai.configure(api_key=api_key)


def _build_history(messages: List[PublicApiChatMessage]) -> List[ChatMessage]:
    return [
        ChatMessage(
            role=ChatRole.USER if m.is_user else ChatRole.MODEL,
            content=m.content
        )
        for m in messages
    ]


class GeminiLiveSession:
    def __init__(
        self,
        *,
        description: str,
        message_history: List[PublicApiChatMessage],
        model: str,
    ) -> None:
        self.description = description
        self.history = message_history
        self.model = model
        self._session_cm = None
        self._session = None
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> None:
        if self._session is not None:
            return

        _configure_api_key()

        self._session_cm = genai.Client().aio.live.connect(
            model=self.model,
            config={
                "system_instruction": self.description,
                "response_modalities": ["TEXT", "AUDIO"]
            }
        )
        self._session = await self._session_cm.__aenter__()

        # Send system instruction and chat history
        await self._session.send(ChatMessage(role=ChatRole.SYSTEM, content=self.description))
        for msg in _build_history(self.history):
            await self._session.send(msg)

    async def ask(
        self,
        text: Optional[str] = None,
        *,
        audio_stream: Optional[Iterable[bytes]] = None
    ) -> AsyncIterable[str]:
        async with self._lock:
            await self._ensure_session()

            if text:
                await self._session.send(ChatMessage(role=ChatRole.USER, content=text))
                if not audio_stream:
                    await self._session.end_turn()

            if audio_stream:
                for chunk in audio_stream:
                    await self._session.send_realtime_input(
                        audio=Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )
                await self._session.send_realtime_input(audio_stream_end=True)

            async for event in self._session.receive():
                if getattr(event, "interrupted", False):
                    break
                if event.text:
                    yield event.text

    async def close(self) -> None:
        if self._session_cm:
            await self._session_cm.__aexit__(None, None, None)
        self._session_cm = None
        self._session = None


async def gemini_chat_completion(
    *,
    text: str,
    description: str,
    message_history: List[PublicApiChatMessage],
    image_base64: Optional[str],
    model: str,
    public_api_token: str,
    audio_stream: Optional[Iterable[bytes]] = None,
) -> AsyncIterable[str]:
    session = GeminiLiveSession(
        description=description,
        message_history=message_history,
        model=model,
    )
    try:
        async for token in session.ask(text=text, audio_stream=audio_stream):
            yield token
    finally:
        await session.close()
