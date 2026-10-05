# InPlaceVideoTransform (g2g backend)
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


from backend.g2g.analytics import analytics
from backend.g2g.frameio import frameio
from backend.g2g.shims import declared_property_names
from log.logger_factory import LoggerFactory


class InPlaceVideoTransform:
    # an element that only reads and writes metadata leaves this False
    READS_PIXELS = True

    def __init__(self):
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.width = 0
        self.height = 0
        self.format = None

    def g2g_properties(self):
        return declared_property_names(type(self))

    def g2g_process(self, buf, width, height, fmt, sink):
        self.width = width
        self.height = height
        self.format = (fmt or "RGB").upper()
        frameio.bind(sink, fmt)
        analytics.bind(sink)
        frame = None
        if self.READS_PIXELS:
            frame = frameio.read_frame(buf, None, width, height)
        self.process_in_place(frame, self.format, buf)
        return None
