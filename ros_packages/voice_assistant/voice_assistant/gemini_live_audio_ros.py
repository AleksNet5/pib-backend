"""ROS node streaming Gemini Live API audio via topics."""

import argparse
import asyncio
import threading
from typing import Optional
from threading import Lock

import numpy as np
from google import generativeai as genai

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
from datatypes.msg import ChatMessage, VoiceAssistantState
from pib_api_client import voice_assistant_client

MODEL = "gemini-2.5-flash-preview-native-audio-dialog"
CONFIG = {"response_modalities": ["AUDIO"]}


class GeminiAudioRosNode(Node):
    """Bridge Gemini Live API native audio to ROS topics."""

    def __init__(self, chat_id: str) -> None:
        super().__init__("gemini_live_audio")
        # running loop will be set once `run` executes
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.chat_id = chat_id
        self.voice_assistant_client_lock = Lock()

        self._va_on = False
        self._sent_end = False

        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._audio_out_queue: asyncio.Queue = asyncio.Queue()
        self._msg_counter = 0

        self.create_subscription(
            Int16MultiArray, "audio_stream", self._audio_callback, 10
        )
        self.create_subscription(
            VoiceAssistantState, "voice_assistant_state", self._va_state_cb, 10
        )
        self.audio_pub = self.create_publisher(
            Int16MultiArray, "audio_playback", 10
        )
        self.chat_pub = self.create_publisher(ChatMessage, "chat_messages", 10)

        self.session: Optional[genai.types.LiveSession] = None

    def _audio_callback(self, msg: Int16MultiArray) -> None:
        if not self._va_on:
            return
        chunk = np.array(msg.data, dtype=np.int16).tobytes()
        try:
            if self.loop:
                self.loop.call_soon_threadsafe(
                    self._audio_queue.put_nowait,
                    {"data": chunk, "mime_type": "audio/pcm"},
                )
        except asyncio.QueueFull:
            pass

    def _va_state_cb(self, msg: VoiceAssistantState) -> None:
        prev = self._va_on
        self._va_on = msg.turned_on
        if not self._va_on and prev and self.loop:
            asyncio.run_coroutine_threadsafe(
                self._audio_queue.put(None), self.loop
            )

    async def _send_realtime(self) -> None:
        while True:
            msg = await self._audio_queue.get()
            if msg is None:
                await self.session.send_realtime_input(audio_stream_end=True)
                self._sent_end = True
                continue
            self._sent_end = False
            await self.session.send_realtime_input(audio=msg)

    async def _publish_audio(self) -> None:
        while True:
            data = await self._audio_out_queue.get()
            arr = np.frombuffer(data, dtype=np.int16)
            msg = Int16MultiArray()
            msg.data = arr.tolist()
            self.audio_pub.publish(msg)

    async def _receive_audio(self) -> None:
        while True:
            turn = self.session.receive()
            text_accum = ""
            async for response in turn:
                if response.data:
                    self._audio_out_queue.put_nowait(response.data)
                if response.text:
                    text_accum += response.text
            if text_accum:
                with self.voice_assistant_client_lock:
                    successful, chat_dto = voice_assistant_client.create_chat_message(
                        self.chat_id, text_accum, False
                    )
                if successful:
                    chat_msg = ChatMessage()
                    chat_msg.chat_id = self.chat_id
                    chat_msg.message_id = chat_dto.message_id
                    chat_msg.timestamp = chat_dto.timestamp
                    chat_msg.is_user = False
                    chat_msg.content = chat_dto.content
                    self.chat_pub.publish(chat_msg)

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        client = genai.Client()
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            self.session = session
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._send_realtime())
                tg.create_task(self._receive_audio())
                tg.create_task(self._publish_audio())


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-id", required=True, help="target chat ID")
    parsed_args, remaining = parser.parse_known_args(args=args)

    rclpy.init(args=remaining)
    node = GeminiAudioRosNode(parsed_args.chat_id)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        asyncio.run(node.run())
    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
