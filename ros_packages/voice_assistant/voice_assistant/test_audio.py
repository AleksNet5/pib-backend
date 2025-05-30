#!/usr/bin/env python3
import os
import time
import wave

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray


class Ros2AudioRecorder(Node):
    def __init__(self,
                 topic: str = 'audio_stream',
                 output_dir: str = '/app/ros2_ws/voice_assistant/audiofiles/',
                 sample_rate: int = 16000,
                 channels: int = 1,
                 clip_duration: float = 4.0):
        super().__init__('test_audio')

        # Recording parameters
        self.sample_rate = sample_rate
        self.channels = channels
        self.clip_duration = clip_duration
        self.frames_per_clip = int(sample_rate * clip_duration)
        self.output_dir = output_dir

        # Internal buffer of raw PCM bytes
        self._buffer = bytearray()
        self._frames_collected = 0  # number of samples collected

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Subscribe to the ROS audio topic
        self.subscription = self.create_subscription(
            Int16MultiArray,
            topic,
            self.audio_callback,
            10
        )

        # Timer to save every clip_duration seconds
        self.timer = self.create_timer(clip_duration, self.save_clip)

        self.get_logger().info(
            f"Subscribed to '{topic}', saving {clip_duration}s clips "
            f"({self.frames_per_clip} frames) at {output_dir}"
        )

    def audio_callback(self, msg: Int16MultiArray):
        # Convert incoming list of int16 samples to raw bytes
        pcm = np.array(msg.data, dtype=np.int16).tobytes()
        self._buffer.extend(pcm)
        # Update how many samples we've got so far
        self._frames_collected += len(msg.data)

        # If we have more than enough, truncate extra
        if self._frames_collected >= self.frames_per_clip:
            # Nothing here; we’ll cut back when saving

            # (Optionally, you could save multiple back-to-back clips if backlog,
            # but usually the timer will align well.)

            pass

    def save_clip(self):
        if self._frames_collected < self.frames_per_clip:
            self.get_logger().warn(
                f"Only {self._frames_collected} samples received; "
                f"skipping save this interval."
            )
            return

        # Trim or pad buffer to exactly the number of bytes we want
        bytes_per_sample = 2  # int16
        desired_bytes = self.frames_per_clip * bytes_per_sample
        clip_bytes = self._buffer[:desired_bytes]

        # Remove used bytes from buffer
        self._buffer = self._buffer[desired_bytes:]
        self._frames_collected -= self.frames_per_clip

        # Build filename
        ts = int(time.time())
        filename = os.path.join(self.output_dir, f"clip_{ts}.wav")
        self.get_logger().info(f"Saving {self.clip_duration}s clip to: {filename}")

        # Write WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(bytes_per_sample)
            wf.setframerate(self.sample_rate)
            wf.writeframes(clip_bytes)

        self.get_logger().info(f"Saved {filename}")

    def destroy_node(self):
        # Nothing special to clean up here
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Ros2AudioRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
