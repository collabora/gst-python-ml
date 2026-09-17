# AlertRecorder
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
    import threading
    import time
    from collections import deque

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

    from log.logger_factory import LoggerFactory  # noqa: E402
    from backend import GObject  # noqa: E402
    from utils.blobs import read_blobs  # noqa: E402

    if backend.BACKEND == "gst":
        VIDEO_CAPS = Gst.Caps.from_string("video/x-raw")
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_alertrecorder' element will not be available. Error: {e}"
    )

# the header pyml_alert attaches, lowercased by read_blobs
ALERT_BLOB = "alert"
DEFAULT_LOCATION = "alert-%s.webm"
DEFAULT_ENCODER = "vp8enc deadline=1 ! webmmux"
CLIP_TIME_FORMAT = "%Y%m%d-%H%M%S"
CLIP_DRAIN_SECONDS = 10


class Clip:
    def __init__(self, path, encoder, caps, start_pts, logger):
        self.path = path
        self.start_pts = start_pts
        self.end_pts = start_pts
        self.logger = logger
        self.pipeline = Gst.parse_launch(
            f"appsrc name=source format=time max-bytes=0 ! videoconvert ! {encoder} "
            f"! filesink location={path}"
        )
        self.source = self.pipeline.get_by_name("source")
        self.source.set_property("caps", caps)
        self.pipeline.set_state(Gst.State.PLAYING)

    def push(self, buffer):
        # buffer.copy() is not writable in pygobject
        frame = Gst.Buffer.new()
        frame.append_memory(buffer.peek_memory(0))
        frame.pts = buffer.pts - self.start_pts
        frame.duration = buffer.duration
        self.source.emit("push-buffer", frame)

    def finish(self):
        self.source.emit("end-of-stream")
        drain = threading.Thread(target=self._drain, daemon=True)
        drain.start()
        return drain

    def _drain(self):
        message = self.pipeline.get_bus().timed_pop_filtered(
            CLIP_DRAIN_SECONDS * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
        )
        self.pipeline.set_state(Gst.State.NULL)
        if message is None or message.type == Gst.MessageType.ERROR:
            self.logger.error(f"clip {self.path} did not finish cleanly")
        else:
            self.logger.info(f"wrote clip {self.path}")


class AlertRecorder(GstBase.BaseTransform):
    __gstmetadata__ = (
        "Alert Recorder",
        "Transform",
        "Records a clip of the frames around each alert an upstream pyml_alert attached",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new(
                "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, VIDEO_CAPS.copy()
            ),
            Gst.PadTemplate.new(
                "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, VIDEO_CAPS.copy()
            ),
        )

    location = GObject.Property(
        type=str,
        default=DEFAULT_LOCATION,
        nick="Location",
        blurb="File each clip is written to, %s stands for the time the alert fired",
    )

    encoder = GObject.Property(
        type=str,
        default=DEFAULT_ENCODER,
        nick="Encoder",
        blurb="Encoder and muxer a clip goes through, must match the location's extension",
    )

    seconds_before = GObject.Property(
        type=float,
        default=3.0,
        minimum=0.0,
        maximum=600.0,
        nick="Seconds Before",
        blurb="Seconds of video kept from before the alert",
    )

    seconds_after = GObject.Property(
        type=float,
        default=5.0,
        minimum=0.0,
        maximum=3600.0,
        nick="Seconds After",
        blurb="Seconds of video recorded after the last alert",
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_passthrough(True)
        self._caps = None
        self._recent = deque()
        self._clip = None
        self._drains = []

    def do_set_caps(self, incaps, outcaps):
        self._caps = incaps.copy()
        return True

    def do_transform_ip(self, buffer):
        if buffer.pts == Gst.CLOCK_TIME_NONE:
            return Gst.FlowReturn.OK
        self._recent.append(buffer)
        before_ns = int(self.seconds_before * Gst.SECOND)
        while buffer.pts - self._recent[0].pts > before_ns:
            self._recent.popleft()
        alerted = ALERT_BLOB in read_blobs(buffer)
        if self._clip is None and alerted:
            self._clip = Clip(
                self.location % time.strftime(CLIP_TIME_FORMAT),
                self.encoder,
                self._caps,
                self._recent[0].pts,
                self.logger,
            )
            for held in self._recent:
                self._clip.push(held)
        elif self._clip is not None:
            self._clip.push(buffer)
        if self._clip is None:
            return Gst.FlowReturn.OK
        if alerted:
            self._clip.end_pts = buffer.pts + int(self.seconds_after * Gst.SECOND)
        if buffer.pts >= self._clip.end_pts:
            self._drains.append(self._clip.finish())
            self._clip = None
        return Gst.FlowReturn.OK

    def do_stop(self):
        if self._clip is not None:
            self._drains.append(self._clip.finish())
            self._clip = None
        for drain in self._drains:
            drain.join()
        self._drains.clear()
        self._recent.clear()
        return True


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_alertrecorder", AlertRecorder
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_alertrecorder' element will not be registered because required modules are missing."
    )
