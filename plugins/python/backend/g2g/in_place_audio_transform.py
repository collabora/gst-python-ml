# InPlaceAudioTransform (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
