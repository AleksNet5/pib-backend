import asyncio
import datetime
import os
import re
from threading import Lock
from typing import AsyncIterable, Iterable, List, Optional

import google.genai as genai
from google.genai import types
import rclpy
from datatypes.action import Chat
from datatypes.msg import ChatMessage
from datatypes.srv import GetCameraImage
from pib_api_client import voice_assistant_client
from public_api_client.public_voice_client import PublicApiChatMessage
from rclpy.action import ActionServer, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher

GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"


class GeminiLiveSession:
    """Persistent Gemini Live API session with interrupt support."""

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
        self._session_cm: Optional[types.LiveConnect] = None
        self._session: Optional[types.LiveSession] = None
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> None:
        if self._session is not None:
            return
        key = os.getenv(GOOGLE_API_KEY_ENV)
        if not key:
            raise RuntimeError(f"{GOOGLE_API_KEY_ENV} not set")
        genai.configure(api_key=key)
        self._session_cm = genai.Client().aio.live.connect(
            model=self.model,
            config={
                "system_instruction": self.description,
                "response_modalities": ["AUDIO", "TEXT"],
            },
        )
        self._session = await self._session_cm.__aenter__()
        await self._session.send(
            types.ChatMessage(role=types.ChatRole.SYSTEM, content=self.description)
        )
        for msg in self._build_history(self.history):
            await self._session.send(msg)

    @staticmethod
    def _build_history(messages: List[PublicApiChatMessage]) -> List[types.ChatMessage]:
        return [
            types.ChatMessage(
                role=types.ChatRole.USER if m.is_user else types.ChatRole.ASSISTANT,
                content=m.content,
            )
            for m in messages
        ]

    async def ask(
        self,
        text: Optional[str] = None,
        *,
        audio_stream: Optional[Iterable[bytes]] = None,
    ) -> AsyncIterable[str]:
        async with self._lock:
            await self._ensure_session()
            if text is not None:
                await self._session.send(
                    types.ChatMessage(role=types.ChatRole.USER, content=text)
                )
            if audio_stream is not None:
                for chunk in audio_stream:
                    await self._session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )
                await self._session.send_realtime_input(audio_stream_end=True)
            async for event in self._session.receive():
                if getattr(event, "interrupted", False):
                    break
                if event.text:
                    yield event.text

    async def close(self) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
            self._session = None


class GeminiChatNode(Node):
    def __init__(self) -> None:
        super().__init__("gemini_chat")
        self.last_pib_message_id: Optional[str] = None
        self.message_content: Optional[str] = None
        self.history_length: int = 10

        self.chat_server = ActionServer(
            self,
            Chat,
            "chat",
            execute_callback=self.chat,
            cancel_callback=(lambda _: CancelResponse.ACCEPT),
            callback_group=ReentrantCallbackGroup(),
        )

        self.chat_message_publisher: Publisher = self.create_publisher(
            ChatMessage, "chat_messages", 10
        )
        self.get_camera_image_client = self.create_client(
            GetCameraImage, "get_camera_image"
        )
        self.voice_assistant_client_lock = Lock()
        self.sessions: dict[str, GeminiLiveSession] = {}
        self.get_logger().info("Now running Gemini Chat")

    def create_chat_message(
        self,
        chat_id: str,
        text: str,
        is_user: bool,
        update_message: bool,
        update_database: bool,
    ) -> None:
        if text == "":
            return
        with self.voice_assistant_client_lock:
            if update_message:
                if update_database:
                    self.message_content = f"{self.message_content} {text}"
                    successful, chat_message = (
                        voice_assistant_client.update_chat_message(
                            chat_id,
                            self.message_content,
                            is_user,
                            self.last_pib_message_id,
                        )
                    )
                    if not successful:
                        self.get_logger().error(
                            f"unable to create chat message: {(chat_id, text, is_user, update_message, update_database)}"
                        )
                        return
                else:
                    self.message_content = f"{self.message_content} {text}"
            else:
                successful, chat_message = voice_assistant_client.create_chat_message(
                    chat_id, text, is_user
                )
                self.last_pib_message_id = chat_message.message_id
                self.message_content = text
            if not successful:
                self.get_logger().error(
                    f"unable to create chat message: {(chat_id, text, is_user, update_message, update_database)}"
                )
                return
        chat_message_ros = ChatMessage()
        chat_message_ros.chat_id = chat_id
        chat_message_ros.content = self.message_content
        chat_message_ros.is_user = is_user
        chat_message_ros.message_id = self.last_pib_message_id
        chat_message_ros.timestamp = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.chat_message_publisher.publish(chat_message_ros)

    async def _get_session(
        self,
        *,
        chat_id: str,
        description: str,
        message_history: List[PublicApiChatMessage],
        model: str,
    ) -> GeminiLiveSession:
        session = self.sessions.get(chat_id)
        if session is None:
            session = GeminiLiveSession(
                description=description,
                message_history=message_history,
                model=model,
            )
            self.sessions[chat_id] = session
        return session

    async def chat(self, goal_handle: ServerGoalHandle):
        request: Chat.Goal = goal_handle.request
        chat_id = request.chat_id
        content = request.text
        generate_code = request.generate_code

        self.executor.create_task(
            self.create_chat_message, chat_id, content, True, False, True
        )

        with self.voice_assistant_client_lock:
            successful, personality = voice_assistant_client.get_personality_from_chat(
                chat_id
            )
            self.history_length = personality.message_history
        if not successful:
            self.get_logger().error(f"no personality found for id {chat_id}")
            goal_handle.abort()
            return Chat.Result()
        description = (
            personality.description
            if personality.description is not None
            else "Du bist pib, ein humanoider Roboter."
        )
        if generate_code:
            description = description

        with self.voice_assistant_client_lock:
            successful, chat_messages = voice_assistant_client.get_chat_history(
                chat_id, self.history_length
            )
        if not successful:
            self.get_logger().error(f"chat with id'{chat_id}' does not exist...")
            goal_handle.abort()
            return Chat.Result()
        message_history = [
            PublicApiChatMessage(message.content, message.is_user)
            for message in chat_messages
        ]

        image_base64 = None
        if personality.assistant_model.has_image_support:
            response = await self.get_camera_image_client.call_async(
                GetCameraImage.Request()
            )
            image_base64 = response.image_base64

        try:
            session = await self._get_session(
                chat_id=chat_id,
                description=description,
                message_history=message_history,
                model=personality.assistant_model.api_name,
            )
            tokens = session.ask(text=content)
            sentence_pattern = re.compile(
                r"^(?!<pib-program>)(.*?)([^\d | ^A-Z][\.|!|\?|:])", re.DOTALL
            )
            code_visual_pattern = re.compile(
                r"^<pib-program>(.*?)</pib-program>", re.DOTALL
            )
            curr_text = ""
            prev_text: Optional[str] = None
            bool_update_chat_message = False
            async for token in tokens:
                if prev_text is not None:
                    feedback = Chat.Feedback()
                    feedback.text = prev_text
                    feedback.text_type = prev_text_type
                    goal_handle.publish_feedback(feedback)
                    prev_text = None
                    prev_text_type = None
                curr_text = curr_text + (
                    token if len(curr_text) > 0 else token.lstrip()
                )
                while True:
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        return Chat.Result()
                    code_visual_match = code_visual_pattern.search(curr_text)
                    if code_visual_match is not None:
                        code_visual = code_visual_match.group(1)
                        prev_text = code_visual
                        prev_text_type = Chat.Goal.TEXT_TYPE_CODE_VISUAL
                        chat_message_text = code_visual_match.group(0)
                        self.executor.create_task(
                            self.create_chat_message,
                            chat_id,
                            chat_message_text,
                            False,
                            bool_update_chat_message,
                            True,
                        )
                        bool_update_chat_message = True
                        curr_text = curr_text[code_visual_match.end() :].rstrip()
                        continue
                    sentence_match = sentence_pattern.search(curr_text)
                    if sentence_match is not None:
                        sentence = sentence_match.group(1) + (
                            sentence_match.group(2)
                            if sentence_match.group(2) is not None
                            else ""
                        )
                        prev_text = sentence
                        prev_text_type = Chat.Goal.TEXT_TYPE_SENTENCE
                        chat_message_text = sentence
                        self.executor.create_task(
                            self.create_chat_message,
                            chat_id,
                            chat_message_text,
                            False,
                            bool_update_chat_message,
                            True,
                        )
                        bool_update_chat_message = True
                        curr_text = curr_text[
                            sentence_match.end(
                                2 if sentence_match.group(2) is not None else 1
                            ) :
                        ].rstrip()
                        continue
                    break
        except Exception as e:
            self.get_logger().error(f"failed to send request to Gemini: {e}")
            goal_handle.abort()
            return Chat.Result()

        goal_handle.succeed()
        result = Chat.Result()
        if prev_text is None:
            result.text = curr_text
            result.text_type = Chat.Goal.TEXT_TYPE_SENTENCE
        else:
            result.text = prev_text
            result.text_type = prev_text_type
        return result


def main(args=None):
    rclpy.init()
    node = GeminiChatNode()
    executor = MultiThreadedExecutor(8)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
