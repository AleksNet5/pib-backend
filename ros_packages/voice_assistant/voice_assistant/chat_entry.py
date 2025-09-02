"""Dispatch chat processing to Gemini or generic chat node."""
import argparse
from pib_api_client import voice_assistant_client


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-id", required=True, help="target chat ID")
    parsed_args, remaining = parser.parse_known_args(args=args)

    success, personality = voice_assistant_client.get_personality_from_chat(parsed_args.chat_id)
    if not success:
        raise RuntimeError(f"could not get personality for chat {parsed_args.chat_id}")

    model = personality.assistant_model.api_name.lower()

    if "gemini" in model and "native-audio" in model:
        from .gemini_live_audio_ros import main as gemini_main
        gemini_main(["--chat-id", parsed_args.chat_id, *remaining])
    else:
        from .chat import main as chat_main
        chat_main(remaining)


if __name__ == "__main__":
    main()
