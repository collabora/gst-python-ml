# OCR
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
    from engine.ocr_engine import OcrEngine
    from engine.engine_factory import EngineFactory
    from backend import frameio, FlowReturn, GObject
    from tasks.ocr import OcrTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'ocr' element will not be available. Error {e}")

# Header prefix for OCR text buffer metadata
OCR_META_HEADER = b"GST-OCR:"


class OCRTransform(VideoTransform, OcrTask):
    """
    GStreamer element for optical character recognition on video frames.

    Set model-name to a HuggingFace model ID, e.g.:
      microsoft/trocr-base-printed

    When draw-text=True (default), recognized text is drawn directly on the
    video frame. OCR results are always appended as a GST-OCR: memory chunk
    (JSON with recognized text and regions).
    """

    __gstmetadata__ = (
        "OCR",
        "Transform",
        "Optical character recognition using TrOCR on video frames",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    draw_text = GObject.Property(
        type=bool,
        default=True,
        nick="Draw Text",
        blurb="Draw recognized text on the video frame",
        flags=GObject.ParamFlags.READWRITE,
    )

    language = GObject.Property(
        type=str,
        default="en",
        nick="Language",
        blurb="Language hint for OCR (currently informational)",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_ocr_engine"
        EngineFactory.register(self.mgr.engine_name, OcrEngine)
        self.format_converter = FormatConverter()

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_ocr")

    def do_transform_ip(self, buf):
        try:
            frames, num_sources, fmt = frameio.read_frames(
                buf, self.sinkpad, self.width, self.height
            )
            if frames is None:
                return FlowReturn.ERROR

            result = self.forward(frames)
            if result is None:
                return FlowReturn.ERROR

            if num_sources == 1:
                output, blob = self.decode(frames, result, fmt)
            else:
                if isinstance(result, list) and len(result) > 0:
                    output, blob = self.decode(
                        frames[0] if frames.ndim == 4 else frames, result[0], fmt
                    )
                else:
                    output, blob = None, None

            if output is not None:
                frameio.write_frame(buf, output)
            if blob is not None:
                frameio.append_blob(buf, OCR_META_HEADER, blob)
            return FlowReturn.OK

        except Exception as e:
            self.logger.error(f"OCR transform error: {e}")
            return FlowReturn.ERROR


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("pyml_ocr", OCRTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_ocr' element will not be registered because required modules are missing."
    )
