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


import backend

CAN_REGISTER_ELEMENT = True
try:
    import collections

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GObject", "2.0")

    from gi.repository import GObject, Gst  # noqa: E402

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    print(f"The 'coalescehistory' element will not be available. Error {e}")


class CoalesceHistory(Gst.Element):
    __gstmetadata__ = (
        "Coalesce History",
        "Text/Transform",
        "Coalesce many text string into a history of the recent past",
        "Olivier Crête <olivier.crete@collabora.com>",
    )

    # Building a Gst object needs Gst.init, which only the gst backend calls.
    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new(
                "src",
                Gst.PadDirection.SRC,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.from_string("text/x-raw,format=utf8"),
            ),
            Gst.PadTemplate.new(
                "sink",
                Gst.PadDirection.SINK,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.from_string("text/x-raw,format=utf8"),
            ),
        )

    @GObject.Property(type=GObject.TYPE_UINT)
    def history_length(self):
        """How many messages to coalesce"""
        return self.history_maxlen

    @history_length.setter
    def history_length(self, value):
        self.history_maxlen = value

    def __init__(self):
        super().__init__()

        self.srcpad = Gst.Pad.new_from_template(self.__gsttemplates__[0], "src")
        self.add_pad(self.srcpad)

        self.sinkpad = Gst.Pad.new_from_template(self.__gsttemplates__[1], "sink")
        self.sinkpad.set_chain_function(self.sink_chain)
        self.sinkpad.set_event_function(self.sink_event)
        self.add_pad(self.sinkpad)

        self.history_maxlen = 10

    def sink_chain(self, _pad, buf):
        print("got buffer : " + str(buf))
        success, map_info_in = buf.map(Gst.MapFlags.READ)
        if not success:
            # printerr("Could not map buffer")
            return Gst.FlowReturn.ERROR

        self.history.append(
            "At time %s:\n%s\n\n"
            % (Gst.TIME_ARGS(buf.pts), bytes(map_info_in.data).decode("utf-8"))
        )

        coalesced_history = "".join(self.history)
        coalesced_buf = Gst.Buffer.new_wrapped(coalesced_history.encode("utf-8"))
        # buf.copy_into(coalesced_buf, Gst.BufferCopyFlags.TIMESTAMPS | Gst.BufferCopyFlags.META, 0, GLib.MAXUINT64)
        coalesced_buf.pts = buf.pts
        coalesced_buf.dts = buf.dts
        coalesced_buf.duration = 60 * Gst.SECOND

        print("pushing")
        ret = self.srcpad.push(coalesced_buf)
        print("pushed")
        return ret

    def sink_event(self, pad, parent, event):
        if event.type in [Gst.EventType.FLUSH_STOP, Gst.EventType.EOS]:
            self.history.clear()
        return pad.event_default(parent, event)

    def do_change_state(self, state_change):
        if state_change == Gst.StateChange.READY_TO_PAUSED:
            self.history = collections.deque(maxlen=self.history_maxlen)

        ret = Gst.Element.do_change_state(self, state_change)

        if state_change == Gst.StateChange.PAUSED_TO_READY:
            del self.history

        return ret


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "coalescehistory", CoalesceHistory
    )
