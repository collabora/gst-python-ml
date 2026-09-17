# MetaReplay
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
    import bisect
    import json

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

    from log.logger_factory import LoggerFactory  # noqa: E402
    from backend import analytics, frameio, GObject  # noqa: E402

    if backend.BACKEND == "gst":
        VIDEO_CAPS = Gst.Caps.from_string("video/x-raw")
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_metareplay' element will not be available. Error: {e}"
    )

MILLISECONDS_PER_SECOND = 1000
NANOSECONDS_PER_MILLISECOND = 1000000
UNKNOWN_DURATION_TOLERANCE_MILLISECONDS = 20
NON_BLOB_KEYS = ("pts", "text", "detections")


def read_records(path):
    records = {}
    with open(path) as lines:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "pts" not in record:
                continue
            records[round(record["pts"] * MILLISECONDS_PER_SECOND)] = record
    return records


class MetaReplay(GstBase.BaseTransform):
    GST_PLUGIN_NAME = "pyml_metareplay"

    __gstmetadata__ = (
        "Metadata Replay",
        "Transform",
        "Reattaches the detections and blobs a pyml_metasink wrote to the frames at the same timestamps",
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
        default="",
        nick="Location",
        blurb="JSON lines file a pyml_metasink wrote",
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        # not passthrough: appending blobs needs a writable buffer
        self.set_passthrough(False)
        self.set_in_place(True)
        self._records = {}
        self._times = []

    def do_start(self):
        self._records = read_records(self.location)
        self._times = sorted(self._records)
        self.logger.info(f"replaying {len(self._records)} records from {self.location}")
        return True

    def do_stop(self):
        self._records = {}
        self._times = []
        return True

    def _record_for(self, buffer):
        if not self._times:
            return None
        milliseconds = round(buffer.pts / NANOSECONDS_PER_MILLISECOND)
        if buffer.duration == Gst.CLOCK_TIME_NONE:
            tolerance = UNKNOWN_DURATION_TOLERANCE_MILLISECONDS
        else:
            tolerance = buffer.duration / NANOSECONDS_PER_MILLISECOND / 2
        index = bisect.bisect_left(self._times, milliseconds)
        candidates = self._times[max(index - 1, 0) : index + 1]
        closest = min(candidates, key=lambda time: abs(time - milliseconds))
        if abs(closest - milliseconds) > tolerance:
            return None
        return self._records[closest]

    def do_transform_ip(self, buffer):
        if buffer.pts == Gst.CLOCK_TIME_NONE:
            return Gst.FlowReturn.OK
        record = self._record_for(buffer)
        if record is None:
            return Gst.FlowReturn.OK
        detections = record.get("detections", [])
        if detections:
            meta = analytics.add_relation_meta(buffer)
            for detection in detections:
                analytics.add_object(
                    meta,
                    detection["label"],
                    detection["x"],
                    detection["y"],
                    detection["w"],
                    detection["h"],
                    detection["score"],
                )
        for name, value in record.items():
            if name in NON_BLOB_KEYS:
                continue
            frameio.append_blob(
                buffer,
                f"GST-{name.upper()}:".encode(),
                json.dumps(value).encode(),
            )
        return Gst.FlowReturn.OK


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        MetaReplay.GST_PLUGIN_NAME, MetaReplay
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_metareplay' element will not be registered because required modules are missing."
    )
