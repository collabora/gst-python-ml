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


from backend.cuda_driver import primary_cuda_context
from backend.g2g.analytics import analytics
from backend.g2g.cuda_frames import rgb_to_nv12, rgba_device_frame
from backend.g2g.frameio import DEVICE_FRAME, frameio
from backend.g2g.shims import declared_property_names
from log.logger_factory import LoggerFactory

DEVICE_DRAWING_FORMAT = "RGBA"
COLOUR_CHANNEL_COUNT = 3


class InPlaceVideoTransform:
    # an element that only reads and writes metadata leaves this False
    READS_PIXELS = True
    # process_in_place then also gets a cuda frame as a torch tensor
    DRAWS_ON_DEVICE_FRAMES = False

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

    def g2g_process_cuda(self, luma, chroma, width, height, sink):
        import torch

        self.width = width
        self.height = height
        self.format = DEVICE_DRAWING_FORMAT
        frameio.bind(sink, self.format)
        analytics.bind(sink)
        if not self.READS_PIXELS:
            self.process_in_place(None, self.format, DEVICE_FRAME)
            return None
        if not self.DRAWS_ON_DEVICE_FRAMES:
            raise ValueError(
                f"{type(self).__name__} draws only on system memory, put "
                "videoconvert between the decoder and it"
            )
        frame = rgba_device_frame(luma, chroma)
        self.process_in_place(frame, self.format, DEVICE_FRAME)
        drawn_planes = rgb_to_nv12(frame[..., :COLOUR_CHANNEL_COUNT])
        self.cuda_device = frame.device.index
        self.cuda_context = primary_cuda_context(self.cuda_device)
        # downstream reads the drawn planes on its own stream once this returns
        torch.cuda.current_stream().synchronize()
        return drawn_planes
