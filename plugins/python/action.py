# Action Recognition
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from collections import deque

    from video_transform import VideoTransform
    from utils.format_converter import FormatConverter
    from engine.action_engine import ActionEngine, ExportedAction
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
        self.task_engine_name = self.mgr.engine_name = "mp_action_engine"
        EngineFactory.register(self.mgr.engine_name, ActionEngine)
        self.format_converter = FormatConverter()
        self._frame_buffer = deque(maxlen=16)
        self._last_result = None

    def export_model(self, model_name):
        return ExportedAction(model_name, self.mgr.engine_name)

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
    __gstelementfactory__ = backend.register_gst_element("mp_action", ActionTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_action' element will not be registered because required modules are missing."
    )
