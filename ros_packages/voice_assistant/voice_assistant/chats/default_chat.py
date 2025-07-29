"""Default chat provider using public API."""

from __future__ import annotations

from typing import AsyncIterable, Iterable, List, Optional

from public_api_client import public_voice_client
from public_api_client.public_voice_client import PublicApiChatMessage

from .base import ChatProvider


class DefaultChat(ChatProvider):
    """Chat provider that proxies to ``public_voice_client.chat_completion``."""

    async def chat_completion(
        self,
        *,
        text: str,
        description: str,
        message_history: List[PublicApiChatMessage],
        public_api_token: str,
        image_base64: Optional[str] = None,
        model: str,
        chat_id: str,
        audio_stream: Optional[Iterable[bytes]] = None,
    ) -> AsyncIterable[str]:
        del audio_stream  # not supported
        tokens = public_voice_client.chat_completion(
            text=text,
            description=description,
            message_history=message_history,
            image_base64=image_base64,
            model=model,
            public_api_token=public_api_token,
        )
        for token in tokens:
            yield token
