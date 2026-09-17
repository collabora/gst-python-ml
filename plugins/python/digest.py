# Incident Digest
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
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

    from backend import GObject  # noqa: E402

    # Building a Gst object needs Gst.init, which only the gst backend calls.
    if backend.BACKEND == "gst":
        TEXT_CAPS = Gst.Caps.from_string("text/x-raw")

    # pygobject does not expose GST_BASE_TRANSFORM_FLOW_DROPPED
    BASE_TRANSFORM_FLOW_DROPPED = Gst.FlowReturn.CUSTOM_SUCCESS

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_digest' element will not be available. Error: {e}"
    )

DEFAULT_WINDOW_SECONDS = 60.0
DIGEST_LINE = "At {seconds:.1f}s: {text}"
DIGEST_SEPARATOR = "\n"


class Digest(GstBase.BaseTransform):
    GST_PLUGIN_NAME = "pyml_digest"

    __gstmetadata__ = (
        "Incident Digest",
        "Text/Transform",
        "Joins the text buffers of a time window into one text buffer per window",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new(
                "src",
                Gst.PadDirection.SRC,
                Gst.PadPresence.ALWAYS,
                TEXT_CAPS.copy(),
            ),
            Gst.PadTemplate.new(
                "sink",
                Gst.PadDirection.SINK,
                Gst.PadPresence.ALWAYS,
                TEXT_CAPS.copy(),
            ),
        )

    window_seconds = GObject.Property(
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        minimum=0.0,
        nick="Window Seconds",
        blurb="Length of the time window each digest covers",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        # every input is consumed, the digest is pushed on the source pad instead
        self.set_in_place(True)
        self._lines = []
        self._window_start = None

    def _push_window(self):
        if not self._lines:
            return Gst.FlowReturn.OK
        digest = Gst.Buffer.new_wrapped(
            DIGEST_SEPARATOR.join(self._lines).encode("utf-8")
        )
        digest.pts = int(self._window_start * Gst.SECOND)
        digest.duration = int(self.window_seconds * Gst.SECOND)
        self._lines = []
        return self.srcpad.push(digest)

    def do_transform_ip(self, buffer):
        with buffer.map(Gst.MapFlags.READ) as info:
            text = bytes(info.data).decode("utf-8", errors="replace")
        seconds = 0.0 if buffer.pts == Gst.CLOCK_TIME_NONE else buffer.pts / Gst.SECOND
        if self._window_start is None:
            self._window_start = seconds
        if seconds >= self._window_start + self.window_seconds:
            result = self._push_window()
            if result != Gst.FlowReturn.OK:
                return result
            self._window_start = seconds
        self._lines.append(DIGEST_LINE.format(seconds=seconds, text=text.strip()))
        return BASE_TRANSFORM_FLOW_DROPPED

    def do_sink_event(self, event):
        if event.type == Gst.EventType.EOS:
            self._push_window()
        elif event.type == Gst.EventType.FLUSH_STOP:
            self._lines = []
            self._window_start = None
        return GstBase.BaseTransform.do_sink_event(self, event)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(Digest.GST_PLUGIN_NAME, Digest)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_digest' element will not be registered because required modules are missing."
    )
