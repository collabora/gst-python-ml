# MetaSink
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
    import json
    import sys

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

    from log.logger_factory import LoggerFactory  # noqa: E402
    from backend import analytics, GObject  # noqa: E402
    from utils.blobs import read_blobs  # noqa: E402
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_metasink' element will not be available. Error: {e}"
    )

BUS_MESSAGE_NAME = "pyml-metadata"
BUS_MESSAGE_FIELD = "json"

TEXT_MEDIA_PREFIX = "text/"


def buffer_record(buffer, media_type):
    record = {}
    if buffer.pts != Gst.CLOCK_TIME_NONE:
        record["pts"] = buffer.pts / Gst.SECOND
    if media_type and media_type.startswith(TEXT_MEDIA_PREFIX):
        with buffer.map(Gst.MapFlags.READ) as info:
            record["text"] = bytes(info.data).decode("utf-8", errors="replace")
        return record
    meta = analytics.get_relation_meta(buffer)
    if meta:
        record["detections"] = analytics.read_objects(meta)
    for name, payload in read_blobs(buffer).items():
        # embedding blobs are binary, not json
        try:
            record[name] = json.loads(payload)
        except ValueError:
            continue
    if len(record) <= 1:
        return None
    return record


class MetaSink(GstBase.BaseSink):
    GST_PLUGIN_NAME = "pyml_metasink"

    __gstmetadata__ = (
        "Metadata Sink",
        "Sink",
        "Writes the analytics metadata, blobs and text of each buffer as one JSON line",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        __gsttemplates__ = Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
        )

    location = GObject.Property(
        type=str,
        default="",
        nick="Location",
        blurb="File to append the JSON lines to, stdout when empty",
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_sync(False)
        self._output = None
        self._media_type = None

    def do_start(self):
        self._output = open(self.location, "a") if self.location else sys.stdout
        return True

    def do_stop(self):
        if self._output is not sys.stdout:
            self._output.close()
        self._output = None
        return True

    def do_set_caps(self, caps):
        self._media_type = caps.get_structure(0).get_name()
        return True

    def do_render(self, buffer):
        record = buffer_record(buffer, self._media_type)
        if record is None:
            return Gst.FlowReturn.OK
        line = json.dumps(record)
        self._output.write(line + "\n")
        self._output.flush()
        structure = Gst.Structure.new_empty(BUS_MESSAGE_NAME)
        structure.set_value(BUS_MESSAGE_FIELD, line)
        self.post_message(Gst.Message.new_application(self, structure))
        return Gst.FlowReturn.OK


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        MetaSink.GST_PLUGIN_NAME, MetaSink
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_metasink' element will not be registered because required modules are missing."
    )
