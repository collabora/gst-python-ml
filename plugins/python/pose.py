# Pose
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
    import ctypes
    import json

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst

    from backend import analytics, GObject
    from base_objectdetector import BaseObjectDetector
    from utils.format_converter import FormatConverter
    from engine.yolo_pose_engine import YoloPoseEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'pose' element will not be available. Error {e}")

# Header prefix for keypoints buffer metadata
POSE_META_HEADER = b"GST-POSE:"

# COCO 17-keypoint names
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Skeleton connections as (keypoint_index_a, keypoint_index_b) pairs
SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # face
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 11),
    (6, 12),  # torso sides
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]


class YOLOPoseTransform(BaseObjectDetector):
    """
    GStreamer element for human pose estimation using YOLO on video frames.

    Detects persons and estimates their 17 COCO body keypoints. Bounding boxes
    are attached as GstAnalytics metadata for use with pyml_overlay. Keypoints
    are appended as a JSON blob in a GST-POSE: buffer memory chunk for use
    by downstream elements.

    When visualize=True (default), the skeleton and keypoints are drawn
    directly on the video frame using Ultralytics' built-in renderer.

    Recommended model names: yolo11n-pose, yolo11s-pose, yolo11m-pose
    """

    __gstmetadata__ = (
        "YOLO Pose",
        "Transform",
        "Human pose estimation with COCO 17-keypoint skeleton using YOLO",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    visualize = GObject.Property(
        type=bool,
        default=True,
        nick="Visualize Skeleton",
        blurb="Draw skeleton and keypoints directly on the video frame",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "pyml_yolo_pose_engine"
        EngineFactory.register(self.mgr.engine_name, YoloPoseEngine)

    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only for this element)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("'engine_name' is read-only for pyml_yolo_pose")

    def do_decode(self, buf, result, stream_idx=0):
        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is None or len(boxes) == 0:
            self.logger.info(f"Stream {stream_idx}: no persons detected")
            return

        # Attach person bounding boxes via GstAnalytics (compatible with pyml_overlay)
        meta = analytics.add_relation_meta(buf)
        if not meta:
            self.logger.error("Failed to add analytics relation metadata")
            return

        kp_data_list = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i]
            score = boxes.conf[i].item()

            mtd = analytics.add_object(
                meta,
                f"stream_{stream_idx}_person",
                x1.item(),
                y1.item(),
                x2.item() - x1.item(),
                y2.item() - y1.item(),
                score,
            )
            if mtd is None:
                self.logger.error(f"Failed to add od_mtd for person {i}")
                continue

            # Collect keypoints for this person
            if keypoints is not None and i < len(keypoints):
                kp_xy = keypoints.xy[i].cpu().numpy().tolist()
                kp_conf = (
                    keypoints.conf[i].cpu().numpy().tolist()
                    if keypoints.conf is not None
                    else [1.0] * len(kp_xy)
                )
                kp_data_list.append(
                    {
                        "person": i,
                        "box": [x1.item(), y1.item(), x2.item(), y2.item()],
                        "score": score,
                        "keypoints": kp_xy,
                        "kp_conf": kp_conf,
                    }
                )

        # Draw skeleton first, before appending any read-only metadata memory.
        # (A READONLY chunk on the buffer would prevent buf.map(WRITE) from succeeding.)
        if self.visualize:
            self._write_annotated_frame(buf, result)

        # Append JSON keypoints as a custom memory chunk on the buffer.
        # Downstream elements can read this with:
        #   for i in range(buf.n_memory()):
        #       data = bytes(buf.peek_memory(i).map(Gst.MapFlags.READ).data)
        #       if data.startswith(b"GST-POSE:"):
        #           persons = json.loads(data[9:])
        if kp_data_list:
            kp_bytes = POSE_META_HEADER + json.dumps(kp_data_list).encode("utf-8")
            # Use new_allocate+fill instead of new_wrapped: PyGI hides the maxsize
            # arg in new_wrapped (it derives it from the data length), so passing it
            # explicitly shifts all subsequent args and causes a GI assertion crash.
            tmp = Gst.Buffer.new_allocate(None, len(kp_bytes), None)
            tmp.fill(0, kp_bytes)
            buf.append_memory(tmp.get_memory(0))
            self.logger.debug(
                f"Stream {stream_idx}: appended keypoints for "
                f"{len(kp_data_list)} persons"
            )

    def _write_annotated_frame(self, buf, result):
        """Write result.plot() (BGR) back into the GStreamer buffer."""
        import numpy as np

        try:
            annotated_bgr = result.plot(kpt_line=True, kpt_radius=4)
            fmt = FormatConverter.get_video_format(buf, self.sinkpad)
            output = self._convert_bgr_to_format(annotated_bgr, fmt)
            if output is None:
                return
            success, map_info = buf.map(Gst.MapFlags.WRITE)
            if success:
                try:
                    frame_bytes = np.ascontiguousarray(output).tobytes()
                    dst = (ctypes.c_char * map_info.size).from_buffer(map_info.data)
                    ctypes.memmove(
                        dst, frame_bytes, min(len(frame_bytes), map_info.size)
                    )
                finally:
                    buf.unmap(map_info)
        except Exception as e:
            self.logger.error(f"Failed to write annotated pose frame: {e}")

    @staticmethod
    def _convert_bgr_to_format(bgr, fmt):
        """Convert a BGR numpy array to the target GStreamer video format."""
        import numpy as np
        import cv2

        if fmt == "RGB":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif fmt == "BGR":
            return bgr
        elif fmt == "RGBA":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        elif fmt == "BGRA":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        elif fmt == "ARGB":
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            return np.roll(rgba, 1, axis=-1)  # RGBA -> ARGB
        elif fmt == "ABGR":
            bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            return np.roll(bgra, 1, axis=-1)  # BGRA -> ABGR
        else:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_yolo_pose", YOLOPoseTransform
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_yolo_pose' element will not be registered because required modules are missing."
    )
