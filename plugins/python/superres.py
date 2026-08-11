# Super Resolution
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
    from video_transform import VideoTransform
    from utils.format_converter import FormatConverter
    from engine.super_res_engine import SuperResEngine
    from engine.engine_factory import EngineFactory
    from backend import frameio, FlowReturn, GObject
    from tasks.superres import SuperResTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'superres' element will not be available. Error {e}")


class SuperResTransform(VideoTransform, SuperResTask):
    """
    GStreamer element for image super-resolution using Real-ESRGAN.

    Set model-name to a Real-ESRGAN variant: real-esrgan-x4 or real-esrgan-x2.

    The upscaled frame is resized back to the original buffer dimensions
    (in-place transform). This provides enhanced detail while maintaining
    pipeline compatibility. Use frame-stride to reduce compute load.
    """

    __gstmetadata__ = (
        "Super Resolution",
        "Transform",
        "Image super-resolution using Real-ESRGAN",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    scale_factor = GObject.Property(
        type=int,
        default=4,
        minimum=2,
        maximum=8,
        nick="Scale Factor",
        blurb="Upscaling factor (2 or 4)",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_superres_engine"
        EngineFactory.register(self.mgr.engine_name, SuperResEngine)
        self.format_converter = FormatConverter()

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_superres")

    def do_transform_ip(self, buf):
        try:
            frames, _num_sources, fmt = frameio.read_frames(
                buf, self.sinkpad, self.width, self.height
            )
            if frames is None:
                return FlowReturn.ERROR

            frame = frames[0] if frames.ndim == 4 else frames
            upscaled = self.forward(frame)
            if upscaled is None:
                return FlowReturn.OK

            # Portable task: resize the upscaled frame back to original dims.
            output, _blob = self.decode(upscaled, fmt)
            if output is not None:
                frameio.write_frame(buf, output)
            return FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Super-resolution transform error: {e}")
            return FlowReturn.ERROR


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_superres", SuperResTransform
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_superres' element will not be registered because required modules are missing."
    )
