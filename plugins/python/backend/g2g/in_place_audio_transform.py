# InPlaceAudioTransform (g2g backend)
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


class InPlaceAudioTransform:
    # the element's one caps description, which mp-launch hands the host
    INPUT_CAPS = "audio/x-raw"

    def __init__(self):
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self._started = False

    def g2g_properties(self):
        return declared_property_names(type(self))

    def on_start(self):
        pass

    # the host sends the input buffer on with what this wrote into it
    def g2g_process_payload(self, buffers, caps, meta):
        import numpy as np

        frameio.bind(meta)
        analytics.bind(meta)
        if not self._started:
            self._started = True
            self.on_start()
        self.process_in_place(np.frombuffer(buffers[0], dtype=np.uint8), buffers[0])
