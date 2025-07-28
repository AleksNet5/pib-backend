import argparse

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .chat import ChatNode
from .gemini_chat_node import GeminiChatNode


def main(args=None):
    parser = argparse.ArgumentParser(description="Start chat node")
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Use Gemini Live API instead of standard public API",
    )
    parsed_args, remaining = parser.parse_known_args(args=args)

    rclpy.init(args=remaining)
    node_cls = GeminiChatNode if parsed_args.gemini else ChatNode
    node = node_cls()

    executor = MultiThreadedExecutor(8)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
