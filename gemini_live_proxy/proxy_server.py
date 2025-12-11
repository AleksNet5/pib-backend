import asyncio
import base64
import json
import logging
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from google import genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GeminiLiveProxy")

app = FastAPI()


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


async def _client_to_gemini(
    websocket: WebSocket, session, stop_event: asyncio.Event
) -> None:
    """
    Forward binary audio frames from the websocket to Gemini live session.
    Text frames after the initial start message are ignored (reserved for future control).
    """
    while not stop_event.is_set():
        try:
            msg = await websocket.receive()
        except WebSocketDisconnect:
            stop_event.set()
            break
        except Exception:
            logger.exception("Error receiving from client websocket")
            stop_event.set()
            break

        msg_type = msg.get("type")
        if msg_type == "websocket.disconnect":
            stop_event.set()
            break

        if msg.get("bytes") is not None:
            try:
                await session.send_realtime_input(
                    audio={"data": msg["bytes"], "mime_type": "audio/pcm"}
                )
            except Exception:
                logger.exception("Failed to forward audio to Gemini")
                stop_event.set()
                break


async def _gemini_to_client(
    websocket: WebSocket, session, stop_event: asyncio.Event
) -> None:
    """
    Receive events from Gemini live session and push them to the websocket client.
    Sends a JSON serialization of the original response so the client can process
    it similarly to a direct Gemini connection.
    """
    try:
        while not stop_event.is_set():
            pending_assistant_text: Optional[str] = None
            async for resp in session.receive():
                if stop_event.is_set():
                    break

                payload = {}

                # server_content transcripts
                sc = getattr(resp, "server_content", None)
                if sc:
                    sc_payload = {}
                    input_transcription = getattr(sc, "input_transcription", None)
                    if input_transcription and getattr(input_transcription, "text", None):
                        sc_payload["input_transcription"] = {
                            "text": input_transcription.text
                        }
                    output_transcription = getattr(sc, "output_transcription", None)
                    if output_transcription and getattr(output_transcription, "text", None):
                        pending_assistant_text = output_transcription.text
                        sc_payload["output_transcription"] = {
                            "text": output_transcription.text
                        }
                    if sc_payload:
                        payload["server_content"] = sc_payload

                # audio data
                if data := getattr(resp, "data", None):
                    payload["data_b64"] = base64.b64encode(data).decode("ascii")
                    if pending_assistant_text:
                        payload.setdefault("server_content", {})
                        payload["server_content"]["output_transcription"] = {
                            "text": pending_assistant_text
                        }
                        pending_assistant_text = None

                if payload:
                    await websocket.send_text(json.dumps(payload))
    except Exception:
        if not stop_event.is_set():
            logger.exception("Error streaming from Gemini to client")
    finally:
        stop_event.set()


@app.websocket("/ws")
async def websocket_proxy(websocket: WebSocket):
    await websocket.accept()

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        await websocket.close(code=4401, reason="missing GOOGLE_API_KEY")
        return

    # Expect first message as JSON with model + config (system_instruction, modalities, etc.)
    try:
        start_msg = await websocket.receive_text()
        start = json.loads(start_msg)
    except Exception:
        await websocket.close(code=4000)
        return

    model = start.get("model") or os.getenv(
        "GEMINI_DEFAULT_MODEL", "gemini-2.5-flash-native-audio-preview-09-2025"
    )
    config = start.get("config", {})

    client = genai.Client(api_key=api_key)
    stop_event = asyncio.Event()
    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            tasks = [
                asyncio.create_task(_client_to_gemini(websocket, session, stop_event)),
                asyncio.create_task(_gemini_to_client(websocket, session, stop_event)),
            ]

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for t in done:
                if t.exception():
                    raise t.exception()
    except Exception as e:
        logger.exception("Proxy websocket session failed")
        await websocket.close(code=1011, reason=str(e))
    finally:
        stop_event.set()
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("proxy_server:app", host="0.0.0.0", port=8081, reload=False)
