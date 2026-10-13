# CoalesceHistory
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


import collections

import backend
from backend import GObject
from base_aggregator import BaseAggregator

if backend.BACKEND == "gst":
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

DEFAULT_HISTORY_LENGTH = 10


class CoalesceHistory(BaseAggregator):
    __gstmetadata__ = (
        "Coalesce History",
        "Text/Transform",
        "Coalesce many text string into a history of the recent past",
        "Olivier Crête <olivier.crete@collabora.com>",
    )

    INPUT_CAPS = "text/x-raw,format=utf8"
    OUTPUT_CAPS = "text/x-raw,format=utf8"

    # Building a Gst object needs Gst.init, which only the gst backend calls.
    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new_with_gtype(
                "sink",
                Gst.PadDirection.SINK,
                Gst.PadPresence.REQUEST,
                Gst.Caps.from_string(INPUT_CAPS),
                GstBase.AggregatorPad.__gtype__,
            ),
            Gst.PadTemplate.new_with_gtype(
                "src",
                Gst.PadDirection.SRC,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.from_string(OUTPUT_CAPS),
                GstBase.AggregatorPad.__gtype__,
            ),
        )

    @GObject.Property(type=int, default=DEFAULT_HISTORY_LENGTH, minimum=1)
    def history_length(self):
        """How many messages to coalesce"""
        return self.history.maxlen

    @history_length.setter
    def history_length(self, value):
        self.history = collections.deque(self.history, maxlen=value)

    def __init__(self):
        super().__init__()
        # this element loads no model
        self.engine_name = None
        self.history = collections.deque(maxlen=DEFAULT_HISTORY_LENGTH)

    def do_load_model(self):
        pass

    def process_payload(self, payload: bytes) -> list[bytes]:
        self.history.append(payload.decode("utf-8") + "\n\n")
        return ["".join(self.history).encode("utf-8")]


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "coalescehistory", CoalesceHistory
    )
