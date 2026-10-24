# InPlaceAudioTransform (GStreamer backend)
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

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GstBase  # noqa: E402

from backend.gst.errors import post_error  # noqa: E402
from backend.gst.frameio import map_video_memory  # noqa: E402
from backend.gst.video_transform import pad_templates  # noqa: E402
from log.logger_factory import LoggerFactory  # noqa: E402


class InPlaceAudioTransform(GstBase.BaseTransform):
    # the element's one caps description, which mp-launch also hands g2g
    INPUT_CAPS = "audio/x-raw"

    # runs before pygobject registers the subclass and reads its templates
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__gsttemplates__ = pad_templates(Gst.Caps.from_string(cls.INPUT_CAPS))

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_in_place(True)

    def on_start(self):
        pass

    def do_start(self):
        try:
            self.on_start()
        except Exception as exception:
            post_error(self, "failed to start", exception)
            return False
        return True

    def do_transform_ip(self, buf):
        import numpy as np

        ok, map_info = map_video_memory(buf, Gst.MapFlags.READ | Gst.MapFlags.WRITE)
        if not ok:
            self.logger.error("failed to map the audio buffer")
            return Gst.FlowReturn.ERROR
        try:
            self.process_in_place(np.frombuffer(map_info.data, dtype=np.uint8), buf)
            return Gst.FlowReturn.OK
        except Exception as exception:
            post_error(self, "transform error", exception)
            return Gst.FlowReturn.ERROR
        finally:
            buf.unmap(map_info)
