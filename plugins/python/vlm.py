# VLM (Vision-Language Model)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301, USA.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from video_transform import VideoTransform
    from engine.vlm_engine import VlmEngine
    from engine.engine_factory import EngineFactory
    from backend import frameio, GObject
    from tasks.vlm import VlmTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'vlm' element will not be available. Error {e}")

# Header prefix for VLM response buffer metadata
VLM_META_HEADER = b"GST-VLM:"


class VlmTransform(VideoTransform, VlmTask):
    """
    GStreamer element for Vision-Language Model inference on video frames.

    For each processed frame, runs a VLM and appends the generated text as a
    GST-VLM: JSON memory chunk on the buffer.

    Downstream elements can read the generated text:
      for i in range(buf.n_memory()):
          data = bytes(buf.peek_memory(i).map(Gst.MapFlags.READ).data)
          if data.startswith(b"GST-VLM:"):
              result = json.loads(data[8:])
              text = result["text"]

    Use frame-stride to control processing frequency:
      pyml_vlm model-name=llava-hf/llava-1.5-7b-hf frame-stride=30 \\
        prompt="What activity is happening in this scene?"
    """

    __gstmetadata__ = (
        "Vision-Language Model",
        "Transform",
        "Vision-Language Model inference for video frame understanding",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    prompt = GObject.Property(
        type=str,
        default="Describe this image in detail.",
        nick="Prompt",
        blurb="User prompt for the VLM",
        flags=GObject.ParamFlags.READWRITE,
    )

    max_tokens = GObject.Property(
        type=int,
        default=256,
        minimum=1,
        maximum=4096,
        nick="Max Tokens",
        blurb="Maximum number of tokens to generate",
        flags=GObject.ParamFlags.READWRITE,
    )

    system_prompt = GObject.Property(
        type=str,
        default=None,
        nick="System Prompt",
        blurb="Optional system prompt for the VLM",
        flags=GObject.ParamFlags.READWRITE,
    )

    temperature = GObject.Property(
        type=float,
        default=0.7,
        minimum=0.0,
        maximum=2.0,
        nick="Temperature",
        blurb="Sampling temperature for generation (0 = greedy)",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_vlm_engine"
        EngineFactory.register(self.mgr.engine_name, VlmEngine)
        self._frame_count = 0

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_vlm")

    def process_frames(self, frames, num_sources, fmt, target):
        """Caption the frame and append the payload, skipping strided-out frames."""
        self._frame_count += 1
        if self.frame_stride > 1 and (self._frame_count % self.frame_stride) != 1:
            return

        if self.engine is None:
            return

        frame = frames[0] if num_sources > 1 else frames

        text = self.forward(frame)

        if text is None:
            return

        # Portable task: serialize the VLM response payload.
        _, payload = self.decode(text)
        # Append VLM response as a JSON memory chunk.
        frameio.write_result(target, None, payload, VLM_META_HEADER)

        self.logger.debug(
            f"VLM response ({len(text)} chars): {text[:80]}..."
            if len(text) > 80
            else f"VLM response: {text}"
        )


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("pyml_vlm", VlmTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_vlm' element will not be registered because required modules are missing."
    )
