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

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GstAnalytics", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GstBase, GstAnalytics, GObject, GLib  # noqa: E402

    from log.logger_factory import LoggerFactory  # noqa: E402

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

    def __init__(
        self,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        keep_alive=2,
        new_track_conf=0.25,
        camera_motion=True,
        dup_iou=0.8,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        # ByteTrack-style activation gate: a brand-new track is only started
        # from a confident detection. Weak/ghost boxes can still *continue* an
        # existing track (matched above) but won't spawn phantom circles.
        self.new_track_conf = new_track_conf
        # Keep emitting a confirmed track (with its Kalman-predicted box) for up
        # to keep_alive frames after a missed detection — bridges flicker so the
        # overlay doesn't blink when the detector drops a box for a frame or two.
        self.keep_alive = keep_alive
        # Camera-motion compensation: estimate the global image shift from the
        # tracks that matched, then re-try matching the leftovers with their
        # predictions shifted by it. Re-attaches players during a pan instead of
        # leaving the old track behind and spawning a duplicate.
        self.camera_motion = camera_motion
        # Two confirmed tracks overlapping more than this IoU are duplicates;
        # the weaker one is dropped (ByteTrack's remove_duplicate_stracks).
        self.dup_iou = dup_iou
        self.trackers = []

    @staticmethod
    def _center(bbox):
        return (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)

    def _estimate_motion(self, matches, predicted, det_bboxes):
        """Fit a global 2D similarity transform (translation + uniform scale +
        rotation) mapping each matched track's predicted centre to its observed
        centre. Returns a callable box->warped-box, or None if it can't be
        estimated. Uses RANSAC (via OpenCV) so players moving against the camera
        consensus are rejected as outliers; falls back to a robust median
        translation if OpenCV is unavailable or the fit is degenerate."""
        import numpy as np

        if len(matches) < 3:
            return None
        src = np.array(
            [self._center(predicted[ti]) for _, ti in matches], dtype=np.float32
        )
        dst = np.array(
            [self._center(det_bboxes[di]) for di, _ in matches], dtype=np.float32
        )

        M = None
        try:
            import cv2

            M, _ = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0
            )
        except Exception:
            M = None

        if M is not None:
            scale = float(np.hypot(M[0, 0], M[0, 1]))
            # Reject implausible fits (e.g. from too few/noisy correspondences).
            if 0.5 <= scale <= 2.0:

                def warp(box):
                    cx, cy = self._center(box)
                    ncx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
                    ncy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
                    nw, nh = box[2] * scale, box[3] * scale
                    return np.array([ncx - nw / 2.0, ncy - nh / 2.0, nw, nh])

                return warp

        # Fallback: robust median translation (pan/tilt only).
        delta = dst - src
        tx, ty = float(np.median(delta[:, 0])), float(np.median(delta[:, 1]))
        if abs(tx) < 1.0 and abs(ty) < 1.0:
            return None
        return lambda box: np.array([box[0] + tx, box[1] + ty, box[2], box[3]])

    def _associate(self, det_bboxes, det_idxs, trk_idxs, trk_boxes, detections):
        """Hungarian-match a subset of detections to a subset of trackers,
        applying updates to matched trackers. Returns list of (det_i, trk_i)."""
        from scipy.optimize import linear_sum_assignment

        if not det_idxs or not trk_idxs:
            return []
        dets = [det_bboxes[d] for d in det_idxs]
        trks = [trk_boxes[t] for t in trk_idxs]
        iou_matrix = iou_batch(dets, trks)
        if iou_matrix.size == 0:
            return []
        cost = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                di, ti = det_idxs[r], trk_idxs[c]
                self.trackers[ti].update(detections[di][:4])
                self.trackers[ti].label_quark = detections[di][5]
                matches.append((di, ti))
        return matches

    def _suppress_duplicates(self):
        """Drop the weaker of any two confirmed tracks sitting on the same box."""
        n = len(self.trackers)
        if n < 2:
            return
        boxes = [t.get_bbox() for t in self.trackers]
        iou_matrix = iou_batch(boxes, boxes)
        remove = set()
        for i in range(n):
            if i in remove:
                continue
            for j in range(i + 1, n):
                if j in remove:
                    continue
                if iou_matrix[i, j] > self.dup_iou:
                    ti, tj = self.trackers[i], self.trackers[j]
                    # Keep the better track: matched more recently, then more
                    # hits; drop the other (usually the freshly-spawned dup).
                    ki = (ti.time_since_update, -ti.hits)
                    kj = (tj.time_since_update, -tj.hits)
                    remove.add(j if ki <= kj else i)
        if remove:
            self.trackers = [t for k, t in enumerate(self.trackers) if k not in remove]

    def update(self, detections):
        """
        Update tracks with new detections.

        Args:
            detections: list of [x, y, w, h, score, label_quark] arrays

        Returns:
            list of (track_id, bbox, label_quark) for confirmed tracks
        """
        import numpy as np

        # Predict new locations for existing tracks
        to_remove = []
        for i, trk in enumerate(self.trackers):
            if np.any(np.isnan(trk.predict())):
                to_remove.append(i)
        for i in reversed(to_remove):
            self.trackers.pop(i)

        det_bboxes = [d[:4] for d in detections] if len(detections) > 0 else []
        n_det = len(det_bboxes)
        n_trk = len(self.trackers)
        # Predicted box per tracker, captured before any update this frame.
        predicted = [self.trackers[i].get_bbox() for i in range(n_trk)]

        # 1) First association on the raw predictions.
        matches = self._associate(
            det_bboxes, list(range(n_det)), list(range(n_trk)), predicted, detections
        )
        matched_det = {di for di, _ in matches}
        matched_trk = {ti for _, ti in matches}

        # 2) Camera-motion compensation: fit a global image transform (pan, zoom
        # and rotation) from the tracks that matched, apply it to the leftover
        # predictions, and re-match. This recovers tracks during camera moves
        # instead of leaving them behind and spawning duplicates.
        if self.camera_motion:
            warp = self._estimate_motion(matches, predicted, det_bboxes)
            if warp is not None:
                rem_trk = [i for i in range(n_trk) if i not in matched_trk]
                rem_det = [d for d in range(n_det) if d not in matched_det]
                if rem_trk and rem_det:
                    shifted = {i: warp(predicted[i]) for i in rem_trk}
                    m2 = self._associate(
                        det_bboxes, rem_det, rem_trk, shifted, detections
                    )
                    matched_det.update(di for di, _ in m2)

        # Create new tracks for unmatched detections, but only from confident
        # ones (ByteTrack activation gate) so weak/ghost boxes don't start a
        # phantom track that gets drawn as a stray circle.
        for d_idx in range(n_det):
            if d_idx not in matched_det:
                if detections[d_idx][4] < self.new_track_conf:
                    continue
                trk = KalmanBoxTracker(detections[d_idx][:4])
                trk.label_quark = detections[d_idx][5]
                self.trackers.append(trk)

        # Remove dead tracks, then drop duplicate tracks sitting on one object.
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]
        self._suppress_duplicates()

        # Return confirmed tracks, including ones that missed a detection this
        # frame (predicted box) for up to keep_alive frames — prevents flicker.
        results = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits and trk.time_since_update <= self.keep_alive:
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

    keep_alive = GObject.Property(
        type=int,
        default=2,
        minimum=0,
        maximum=1000,
        nick="Keep Alive",
        blurb="Frames to keep emitting a confirmed track (Kalman-predicted box) "
        "after a missed detection; bridges flicker (0 = only matched frames)",
        flags=GObject.ParamFlags.READWRITE,
    )

    new_track_confidence = GObject.Property(
        type=float,
        default=0.25,
        minimum=0.0,
        maximum=1.0,
        nick="New Track Confidence",
        blurb="Minimum detection confidence to START a new track (ByteTrack "
        "activation gate); weak boxes still continue existing tracks but "
        "won't spawn phantom/duplicate circles",
        flags=GObject.ParamFlags.READWRITE,
    )

    camera_motion = GObject.Property(
        type=bool,
        default=True,
        nick="Camera Motion Compensation",
        blurb="Estimate the global image shift from matched tracks and re-match "
        "leftovers shifted by it, so a panning camera re-attaches players "
        "instead of leaving the old track behind and spawning a duplicate",
        flags=GObject.ParamFlags.READWRITE,
    )

    duplicate_iou = GObject.Property(
        type=float,
        default=0.8,
        minimum=0.0,
        maximum=1.0,
        nick="Duplicate IoU",
        blurb="Two confirmed tracks overlapping more than this are treated as "
        "duplicates and the weaker one is dropped",
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
                keep_alive=self.keep_alive,
                new_track_conf=self.new_track_confidence,
                camera_motion=self.camera_motion,
                dup_iou=self.duplicate_iou,
            )
        return self._tracker

    def _read_detections(self, buf):
        """Extract detections from upstream GstAnalytics od_mtd."""
        detections = []
        meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if not meta:
            return detections

        count = GstAnalytics.relation_get_length(meta)
        for index in range(count):
            ret, od_mtd = meta.get_od_mtd(index)
            if not ret or od_mtd is None:
                continue
            label_quark = od_mtd.get_obj_type()
            presence, x, y, w, h, score = od_mtd.get_location()
            if presence:
                detections.append([x, y, w, h, score, label_quark])
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
            meta = GstAnalytics.buffer_add_analytics_relation_meta(buf)
            if not meta:
                self.logger.error(
                    "Failed to add analytics relation metadata for tracking"
                )
                return Gst.FlowReturn.ERROR

            for track_id, bbox, label_quark in tracked:
                label_str = GLib.quark_to_string(label_quark)
                track_label = f"{label_str}_id_{track_id}"
                qk = GLib.quark_from_string(track_label)
                x, y, w, h = bbox
                ret, od_mtd = meta.add_od_mtd(qk, int(x), int(y), int(w), int(h), 1.0)
                if not ret:
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
        elif prop.name == "keep-alive":
            return self.keep_alive
        elif prop.name == "new-track-confidence":
            return self.new_track_confidence
        elif prop.name == "camera-motion":
            return self.camera_motion
        elif prop.name == "duplicate-iou":
            return self.duplicate_iou
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
        elif prop.name == "keep-alive":
            self.keep_alive = value
            self._tracker = None
        elif prop.name == "new-track-confidence":
            self.new_track_confidence = value
            self._tracker = None
        elif prop.name == "camera-motion":
            self.camera_motion = value
            self._tracker = None
        elif prop.name == "duplicate-iou":
            self.duplicate_iou = value
            self._tracker = None
        else:
            raise AttributeError(f"Unknown property {prop.name}")


if CAN_REGISTER_ELEMENT:
    GObject.type_register(TrackerTransform)
    __gstelementfactory__ = ("pyml_tracker", Gst.Rank.NONE, TrackerTransform)
else:
    GlobalLogger().warning(
        "The 'pyml_tracker' element will not be registered because required modules are missing."
    )
