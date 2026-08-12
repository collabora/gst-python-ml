# Action Recognition
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
    from collections import deque

    from video_transform import VideoTransform
    from utils.format_converter import FormatConverter
    from engine.action_engine import ActionEngine
    from engine.engine_factory import EngineFactory
    from backend import frameio, GObject
    from tasks.action import ActionTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'action' element will not be available. Error {e}")

# Header prefix for action classification buffer metadata
ACTION_META_HEADER = b"GST-ACTION:"


class ActionTransform(VideoTransform, ActionTask):
    """
    GStreamer element for video action/activity recognition using VideoMAE.

    Buffers a window of frames internally and runs classification when the
    buffer is full. The predicted action label is attached as a GST-ACTION:
    memory chunk (JSON) and optionally drawn on the frame.

    Set model-name to a HuggingFace model ID, e.g.:
      MCG-NJU/videomae-base-finetuned-kinetics
    """

    META_HEADER = ACTION_META_HEADER

    __gstmetadata__ = (
        "Action Recognition",
        "Transform",
        "Video action/activity recognition using VideoMAE",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    num_frames = GObject.Property(
        type=int,
        default=16,
        minimum=2,
        maximum=128,
        nick="Num Frames",
        blurb="Number of frames to buffer before running classification",
        flags=GObject.ParamFlags.READWRITE,
    )

    draw_label = GObject.Property(
        type=bool,
        default=True,
        nick="Draw Label",
        blurb="Draw predicted action label on the video frame",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_action_engine"
        EngineFactory.register(self.mgr.engine_name, ActionEngine)
        self.format_converter = FormatConverter()
        self._frame_buffer = deque(maxlen=16)
        self._last_result = None

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_action")

    def process_frames(self, frames, num_sources, fmt, target):
        """Accumulate a frame window, classify when it is full, decode the
        latest result on every frame."""
        frame = frames[0] if frames.ndim == 4 else frames

        if self._frame_buffer.maxlen != self.num_frames:
            self._frame_buffer = deque(self._frame_buffer, maxlen=self.num_frames)
        self._frame_buffer.append(frame.copy())

        if len(self._frame_buffer) == self.num_frames:
            result = self.forward(list(self._frame_buffer))
            if result is not None:
                self._last_result = result

        if self._last_result is None:
            return
        output, blob = self.decode(frame, self._last_result, fmt)
        frameio.write_result(target, output, blob, self.META_HEADER)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("pyml_action", ActionTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_action' element will not be registered because required modules are missing."
    )
