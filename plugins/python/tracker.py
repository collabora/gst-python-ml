# Multi-Object Tracker
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
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

    from log.logger_factory import LoggerFactory  # noqa: E402
    from backend import analytics, GObject  # noqa: E402

    # Building a Gst object needs Gst.init, which only the gst backend calls.
    if backend.BACKEND == "gst":
        VIDEO_SRC_CAPS = Gst.Caps.from_string("video/x-raw")
        VIDEO_SINK_CAPS = Gst.Caps.from_string("video/x-raw")

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_tracker' element will not be available. Error: {e}"
    )


class KalmanBoxTracker:
    """Simple Kalman filter tracker for a single bounding box."""

    _id_counter = 0

    def __init__(self, bbox):
        import numpy as np

        self.id = KalmanBoxTracker._id_counter
        KalmanBoxTracker._id_counter += 1
        # State: [x_center, y_center, w, h, vx, vy, vw, vh]
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        self.state = np.array([cx, cy, bbox[2], bbox[3], 0, 0, 0, 0], dtype=np.float64)
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        """Advance state by one frame using constant velocity model."""
        self.state[:4] += self.state[4:]
        self.age += 1
        self.time_since_update += 1
        return self._get_bbox()

    def update(self, bbox):
        """Update state with observed bounding box [x, y, w, h]."""
        import numpy as np

        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        observed = np.array([cx, cy, bbox[2], bbox[3]], dtype=np.float64)
        # Simple exponential smoothing for velocity
        alpha = 0.5
        new_vel = observed - self.state[:4]
        self.state[4:] = alpha * new_vel + (1 - alpha) * self.state[4:]
        self.state[:4] = observed
        self.hits += 1
        self.time_since_update = 0

    def _get_bbox(self):
        """Return [x, y, w, h] from state."""
        import numpy as np

        w = max(self.state[2], 0)
        h = max(self.state[3], 0)
        x = self.state[0] - w / 2.0
        y = self.state[1] - h / 2.0
        return np.array([x, y, w, h])

    def get_bbox(self):
        return self._get_bbox()


def iou_batch(bb_det, bb_trk):
    """Compute IoU between two sets of [x, y, w, h] bounding boxes."""
    import numpy as np

    if len(bb_det) == 0 or len(bb_trk) == 0:
        return np.empty((len(bb_det), len(bb_trk)))

    det = np.array(bb_det)
    trk = np.array(bb_trk)

    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    det_xy = np.column_stack(
        [det[:, 0], det[:, 1], det[:, 0] + det[:, 2], det[:, 1] + det[:, 3]]
    )
    trk_xy = np.column_stack(
        [trk[:, 0], trk[:, 1], trk[:, 0] + trk[:, 2], trk[:, 1] + trk[:, 3]]
    )

    xx1 = np.maximum(det_xy[:, None, 0], trk_xy[None, :, 0])
    yy1 = np.maximum(det_xy[:, None, 1], trk_xy[None, :, 1])
    xx2 = np.minimum(det_xy[:, None, 2], trk_xy[None, :, 2])
    yy2 = np.minimum(det_xy[:, None, 3], trk_xy[None, :, 3])

    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)

    area_det = det[:, 2] * det[:, 3]
    area_trk = trk[:, 2] * trk[:, 3]

    union = area_det[:, None] + area_trk[None, :] - inter
    return inter / np.maximum(union, 1e-7)


class SortTracker:
    """SORT/ByteTrack multi-object tracker using IoU + Kalman filtering."""

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        """
        Update tracks with new detections.

        Args:
            detections: list of [x, y, w, h, score, label_quark] arrays

        Returns:
            list of (track_id, bbox, label_quark) for confirmed tracks
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        # Predict new locations for existing tracks
        predicted = []
        to_remove = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict()
            if np.any(np.isnan(pred)):
                to_remove.append(i)
            else:
                predicted.append(pred)
        for i in reversed(to_remove):
            self.trackers.pop(i)

        # Build cost matrix using IoU
        det_bboxes = [d[:4] for d in detections] if len(detections) > 0 else []
        iou_matrix = iou_batch(det_bboxes, predicted)
        cost_matrix = 1.0 - iou_matrix

        # Hungarian assignment
        matched_det = set()
        matched_trk = set()
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_det.add(r)
                    matched_trk.add(c)
                    self.trackers[c].update(detections[r][:4])
                    # Store latest label quark on tracker
                    self.trackers[c].label_quark = detections[r][5]

        # Create new tracks for unmatched detections
        for d_idx in range(len(detections)):
            if d_idx not in matched_det:
                trk = KalmanBoxTracker(detections[d_idx][:4])
                trk.label_quark = detections[d_idx][5]
                self.trackers.append(trk)

        # Remove dead tracks
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]

        # Return confirmed tracks
        results = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits and trk.time_since_update == 0:
                results.append((trk.id, trk.get_bbox(), trk.label_quark))
        return results


class TrackerTransform(GstBase.BaseTransform):
    """
    GStreamer element for multi-object tracking.

    Reads upstream GstAnalytics od_mtd (object detection metadata) from buffers,
    runs a SORT/ByteTrack tracking algorithm to assign consistent IDs across frames,
    and attaches tracking_mtd linked to od_mtd via RELATE_TO.
    """

    __gstmetadata__ = (
        "Multi-Object Tracker",
        "Transform",
        "Assigns persistent track IDs to detected objects using ByteTrack/SORT",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        src_template = Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            VIDEO_SRC_CAPS.copy(),
        )

        sink_template = Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            VIDEO_SINK_CAPS.copy(),
        )
        __gsttemplates__ = (src_template, sink_template)

    tracker_type = GObject.Property(
        type=str,
        default="bytetrack",
        nick="Tracker Type",
        blurb="Tracking algorithm to use: 'bytetrack' or 'sort'",
        flags=GObject.ParamFlags.READWRITE,
    )

    max_age = GObject.Property(
        type=int,
        default=30,
        minimum=1,
        maximum=1000,
        nick="Max Age",
        blurb="Maximum number of frames to keep a lost track before deletion",
        flags=GObject.ParamFlags.READWRITE,
    )

    min_hits = GObject.Property(
        type=int,
        default=3,
        minimum=1,
        maximum=100,
        nick="Min Hits",
        blurb="Minimum detections before a track is confirmed",
        flags=GObject.ParamFlags.READWRITE,
    )

    iou_threshold = GObject.Property(
        type=float,
        default=0.3,
        minimum=0.0,
        maximum=1.0,
        nick="IoU Threshold",
        blurb="Minimum IoU for detection-to-track assignment",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_passthrough(True)
        self.set_in_place(True)
        self._tracker = None

    def _ensure_tracker(self):
        if self._tracker is None:
            self._tracker = SortTracker(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold,
            )
        return self._tracker

    def _read_detections(self, buf):
        """Extract detections from upstream object-detection metadata."""
        detections = []
        meta = analytics.get_relation_meta(buf)
        if not meta:
            return detections

        for obj in analytics.read_objects(meta):
            detections.append(
                [obj["x"], obj["y"], obj["w"], obj["h"], obj["score"], obj["label"]]
            )
        return detections

    def do_transform_ip(self, buf):
        try:
            tracker = self._ensure_tracker()
            detections = self._read_detections(buf)

            if len(detections) == 0:
                # Still run update so trackers age out
                tracker.update([])
                return Gst.FlowReturn.OK

            tracked = tracker.update(detections)

            # Attach tracking results as new analytics metadata
            meta = analytics.add_relation_meta(buf)
            if not meta:
                self.logger.error(
                    "Failed to add analytics relation metadata for tracking"
                )
                return Gst.FlowReturn.ERROR

            for track_id, bbox, label_str in tracked:
                track_label = f"{label_str}_id_{track_id}"
                x, y, w, h = bbox
                od_mtd = analytics.add_object(
                    meta, track_label, int(x), int(y), int(w), int(h), 1.0
                )
                if od_mtd is None:
                    self.logger.error(
                        f"Failed to add tracking od_mtd for track {track_id}"
                    )

            self.logger.info(
                f"Tracker: {len(detections)} detections -> {len(tracked)} confirmed tracks"
            )
            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Tracker transform error: {e}")
            return Gst.FlowReturn.ERROR

    def do_get_property(self, prop):
        if prop.name == "tracker-type":
            return self.tracker_type
        elif prop.name == "max-age":
            return self.max_age
        elif prop.name == "min-hits":
            return self.min_hits
        elif prop.name == "iou-threshold":
            return self.iou_threshold
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name == "tracker-type":
            self.tracker_type = value
            self._tracker = None
        elif prop.name == "max-age":
            self.max_age = value
            self._tracker = None
        elif prop.name == "min-hits":
            self.min_hits = value
            self._tracker = None
        elif prop.name == "iou-threshold":
            self.iou_threshold = value
            self._tracker = None
        else:
            raise AttributeError(f"Unknown property {prop.name}")


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_tracker", TrackerTransform
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_tracker' element will not be registered because required modules are missing."
    )
