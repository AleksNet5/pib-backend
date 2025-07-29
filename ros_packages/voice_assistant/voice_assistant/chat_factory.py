"""Factory to route chat requests to the correct provider."""
from __future__ import annotations

from typing import Iterable, AsyncIterable, Callable, Dict, List, Optional, Type

from public_api_client.public_voice_client import PublicApiChatMessage

from .chats import ChatProvider, DefaultChat, GeminiChat


_PROVIDER_MAP: Dict[str, Type[ChatProvider]] = {
    "default": DefaultChat,
    "gemini": GeminiChat,
}


def get_provider(model: str) -> ChatProvider:
    """Return chat provider instance for given model name."""
    if "gemini" in model.lower():
        provider_cls = _PROVIDER_MAP.get("gemini", DefaultChat)
    else:
        provider_cls = _PROVIDER_MAP.get("default", DefaultChat)
    return provider_cls()


async def chat_completion(
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
    """Return tokens from the selected LLM model."""
    provider = get_provider(model)
    async for token in provider.chat_completion(
        text=text,
        description=description,
        message_history=message_history,
        public_api_token=public_api_token,
        image_base64=image_base64,
        model=model,
        chat_id=chat_id,
        audio_stream=audio_stream,
    ):
        yield token
