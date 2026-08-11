# Depth
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
    from engine.depth_anything_engine import DepthAnythingEngine
    from engine.engine_factory import EngineFactory
    from backend import frameio, FlowReturn, GObject
    from tasks.depth import DepthTask

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'depth' element will not be available. Error {e}")

# Header prefix for depth map buffer metadata
DEPTH_META_HEADER = b"GST-DEPTH:"


class DepthTransform(VideoTransform, DepthTask):
    """
    GStreamer element for monocular depth estimation using DepthAnything V2.

    Set model-name to a HuggingFace model ID, e.g.:
      depth-anything/Depth-Anything-V2-Small-hf

    When visualize=True (default), the video frame is replaced with a
    colorized depth map. Use a tee element upstream to preserve the original
    video alongside the depth visualization.

    A uint8 normalized depth map is always appended to the buffer as a
    GST-DEPTH: memory chunk for downstream elements:
      for i in range(buf.n_memory()):
          data = bytes(buf.peek_memory(i).map(Gst.MapFlags.READ).data)
          if data.startswith(b"GST-DEPTH:"):
              depth = np.frombuffer(data[10:], dtype=np.uint8).reshape(H, W)

    Use frame-stride to skip frames and reduce compute:
      pyml_depth model-name=depth-anything/Depth-Anything-V2-Small-hf frame-stride=2
    """

    __gstmetadata__ = (
        "Depth",
        "Transform",
        "Monocular depth estimation using DepthAnything V2",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    visualize = GObject.Property(
        type=bool,
        default=True,
        nick="Visualize Depth",
        blurb="Replace video frame with a colorized depth map",
        flags=GObject.ParamFlags.READWRITE,
    )

    colormap = GObject.Property(
        type=str,
        default="inferno",
        nick="Colormap",
        blurb="Colormap for depth visualization: inferno, jet, viridis, plasma, magma",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_depth_engine"
        EngineFactory.register(self.mgr.engine_name, DepthAnythingEngine)
        self.format_converter = FormatConverter()

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_depth")

    def do_transform_ip(self, buf):
        try:
            frames, num_sources, fmt = frameio.read_frames(
                buf, self.sinkpad, self.width, self.height
            )
            if frames is None:
                return FlowReturn.ERROR

            if num_sources == 1:
                depth = self.forward(frames)
                if depth is None:
                    return FlowReturn.ERROR
                output, blob = self.decode(frames, depth, fmt)
            else:
                depths = self.forward(frames)
                if not depths:
                    return FlowReturn.OK
                # For batch: apply only the first depth map (primary frame)
                output, blob = self.decode(frames, depths[0], fmt)

            if output is not None:
                frameio.write_frame(buf, output)
            if blob is not None:
                frameio.append_blob(buf, DEPTH_META_HEADER, blob)
            return FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Depth transform error: {e}")
            return FlowReturn.ERROR


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("pyml_depth", DepthTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_depth' element will not be registered because required modules are missing."
    )
