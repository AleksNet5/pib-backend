"""Base classes for chat providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterable, Iterable, List, Optional

from public_api_client.public_voice_client import PublicApiChatMessage


class ChatProvider(ABC):
    """Abstract chat provider interface."""

    @abstractmethod
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
        """Yield tokens for the chat response."""
        raise NotImplementedError
