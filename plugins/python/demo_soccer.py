# DemoSoccer
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
    from gi.repository import Gst  # noqa: E402
    from backend import analytics, GObject  # noqa: E402
    from base_objectdetector import BaseObjectDetector

    import os
    from collections import deque
    from engine.engine_factory import EngineFactory
    from engine.yolo_advanced_engine import (
        YoloAdvancedEngine,
        BoTSORTWrapper,
        ByteTrackWrapper,
        tlbr_of,
    )

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'yolo_advanced' element will not be available. Error {e}"
    )

COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
}


class DemoSoccer(BaseObjectDetector):
    """
    GStreamer element for advanced YOLO inference focused on person and ball tracking with fallback and gating.
    """

    __gstmetadata__ = (
        "YOLOBall",
        "Transform",
        "Advanced YOLO for person/ball detection, dual tracking, and ball trail stabilization",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self.mgr.engine_name = "demo_engine"
        EngineFactory.register(self.engine_name, YoloAdvancedEngine)

        # Resolve YAMLs
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_base = os.path.join(script_dir, "data", "soccer")
        self.logger.info(f"YAML base resolved to: {yaml_base}")
        self.__tracker_people = os.path.join(yaml_base, "botsort_people_reid.yaml")
        self.__tracker_ball = os.path.join(yaml_base, "bytetrack_ball.yaml")

        # Check if YAML files exist
        if not os.path.exists(self.__tracker_people):
            raise FileNotFoundError(
                f"People tracker YAML file not found: {self.__tracker_people}"
            )
        if not os.path.exists(self.__tracker_ball):
            raise FileNotFoundError(
                f"Ball tracker YAML file not found: {self.__tracker_ball}"
            )

        # Defaults (all params)
        self.__model = "yolo11x"
        self.__device = "auto"
        self.__imgsz = 1280
        self.__conf = 0.25
        self.__iou = 0.45
        self.__classes = [0, 32]
        self.__person_conf_keep = 0.25
        self.__ball_conf_keep = 0.04
        self.__ball_mode = True
        self.__hires_fallback = True
        self.__hires_imgsz = 1536
        self.__fallback_every = 6
        self.__fallback_tiles = False
        self.__tile_size = 1280
        self.__tile_overlap = 256
        self.__fallback_budget_ms = 300
        self.__ball_roi_boost = False
        self.__roi_scale = 2.5
        self.__roi_min = 256
        self.__roi_max = 1920
        self.__people_reid = True
        self.__trail = 200
        self.__gmc = "affine"
        self.__gmc_scale = 0.5
        self.__gft_max_corners = 400
        self.__gft_quality = 0.01
        self.__gft_min_dist = 8
        self.__lk_win = 21
        self.__lk_levels = 3
        self.__ransac_thresh = 3.0
        self.__ball_gate_rel = 0.06
        self.__ball_gate_min = 12
        self.__ball_gate_use_pred = False
        self.__ball_min_iou = 0.20
        self.__ball_max_jump_rel = 0.12
        self.__ball_speed_mult = 3.0
        self.__ball_smooth_ema = 0.0
        self.__det_override_conf = 0.28
        self.__det_override_after = 2
        self.__reacquire_frames = 6
        self.__ball_coast = False
        self.__coast_max = 6
        self.__coast_decay = 0.90
        self.__verbose = False
        self.__frame_rate = 30.0

    # make engine_name read only
    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only in this class)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError(
            "The 'engine_name' property cannot be set in this derived class."
        )

    # Properties - all of them
    @GObject.Property(type=str, default="yolo11x")
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value
        if self.engine:
            self.engine.do_load_model(value)

    @GObject.Property(type=str, default="auto")
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value
        if self.engine:
            self.engine.device_str = value

    @GObject.Property(type=int, default=1280)
    def imgsz(self):
        return self.__imgsz

    @imgsz.setter
    def imgsz(self, value):
        self.__imgsz = value
        if self.engine:
            self.engine.imgsz = value

    @GObject.Property(type=float, default=0.25)
    def conf(self):
        return self.__conf

    @conf.setter
    def conf(self, value):
        self.__conf = value
        if self.engine:
            self.engine.conf = value

    @GObject.Property(type=float, default=0.45)
    def iou(self):
        return self.__iou

    @iou.setter
    def iou(self, value):
        self.__iou = value
        if self.engine:
            self.engine.iou = value

    @GObject.Property(type=str, default="0 32")
    def classes(self):
        return " ".join(map(str, self.__classes))

    @classes.setter
    def classes(self, value):
        self.__classes = [int(c.strip()) for c in value.split()]
        if self.engine:
            self.engine.classes = self.__classes

    @GObject.Property(type=float, default=0.25)
    def person_conf_keep(self):
        return self.__person_conf_keep

    @person_conf_keep.setter
    def person_conf_keep(self, value):
        self.__person_conf_keep = value
        if self.engine:
            self.engine.person_conf_keep = value

    @GObject.Property(type=float, default=0.04)
    def ball_conf_keep(self):
        return self.__ball_conf_keep

    @ball_conf_keep.setter
    def ball_conf_keep(self, value):
        self.__ball_conf_keep = value
        if self.engine:
            self.engine.ball_conf_keep = value

    @GObject.Property(type=bool, default=True)
    def ball_mode(self):
        return self.__ball_mode

    @ball_mode.setter
    def ball_mode(self, value):
        self.__ball_mode = value
        if self.engine:
            self.engine.ball_mode = value

    @GObject.Property(type=bool, default=True)
    def hires_fallback(self):
        return self.__hires_fallback

    @hires_fallback.setter
    def hires_fallback(self, value):
        self.__hires_fallback = value
        if self.engine:
            self.engine.hires_fallback = value

    @GObject.Property(type=int, default=1536)
    def hires_imgsz(self):
        return self.__hires_imgsz

    @hires_imgsz.setter
    def hires_imgsz(self, value):
        self.__hires_imgsz = value
        if self.engine:
            self.engine.hires_imgsz = value

    @GObject.Property(type=int, default=6)
    def fallback_every(self):
        return self.__fallback_every

    @fallback_every.setter
    def fallback_every(self, value):
        self.__fallback_every = value
        if self.engine:
            self.engine.fallback_every = value

    @GObject.Property(type=bool, default=False)
    def fallback_tiles(self):
        return self.__fallback_tiles

    @fallback_tiles.setter
    def fallback_tiles(self, value):
        self.__fallback_tiles = value
        if self.engine:
            self.engine.fallback_tiles = value

    @GObject.Property(type=int, default=1280)
    def tile_size(self):
        return self.__tile_size

    @tile_size.setter
    def tile_size(self, value):
        self.__tile_size = value
        if self.engine:
            self.engine.tile_size = value

    @GObject.Property(type=int, default=256)
    def tile_overlap(self):
        return self.__tile_overlap

    @tile_overlap.setter
    def tile_overlap(self, value):
        self.__tile_overlap = value
        if self.engine:
            self.engine.tile_overlap = value

    @GObject.Property(type=int, default=300)
    def fallback_budget_ms(self):
        return self.__fallback_budget_ms

    @fallback_budget_ms.setter
    def fallback_budget_ms(self, value):
        self.__fallback_budget_ms = value
        if self.engine:
            self.engine.fallback_budget_ms = value

    @GObject.Property(type=bool, default=False)
    def ball_roi_boost(self):
        return self.__ball_roi_boost

    @ball_roi_boost.setter
    def ball_roi_boost(self, value):
        self.__ball_roi_boost = value
        if self.engine:
            self.engine.ball_roi_boost = value

    @GObject.Property(type=float, default=2.5)
    def roi_scale(self):
        return self.__roi_scale

    @roi_scale.setter
    def roi_scale(self, value):
        self.__roi_scale = value
        if self.engine:
            self.engine.roi_scale = value

    @GObject.Property(type=int, default=256)
    def roi_min(self):
        return self.__roi_min

    @roi_min.setter
    def roi_min(self, value):
        self.__roi_min = value
        if self.engine:
            self.engine.roi_min = value

    @GObject.Property(type=int, default=1920)
    def roi_max(self):
        return self.__roi_max

    @roi_max.setter
    def roi_max(self, value):
        self.__roi_max = value
        if self.engine:
            self.engine.roi_max = value

    @GObject.Property(type=str, default="botsort_people_reid.yaml")
    def tracker_people(self):
        return self.__tracker_people

    @tracker_people.setter
    def tracker_people(self, value):
        self.__tracker_people = value
        if self.engine:
            self.engine.tracker_people = value
            self.engine.people_tracker = BoTSORTWrapper(
                value, self.engine.frame_rate, self.people_reid
            )

    @GObject.Property(type=str, default="bytetrack_ball.yaml")
    def tracker_ball(self):
        return self.__tracker_ball

    @tracker_ball.setter
    def tracker_ball(self, value):
        self.__tracker_ball = value
        if self.engine:
            self.engine.tracker_ball = value
            self.engine.ball_tracker = ByteTrackWrapper(value, self.engine.frame_rate)

    @GObject.Property(type=bool, default=True)
    def people_reid(self):
        return self.__people_reid

    @people_reid.setter
    def people_reid(self, value):
        self.__people_reid = value
        if self.engine:
            self.engine.people_reid = value
            # Reinit tracker if needed

    @GObject.Property(type=int, default=200)
    def trail(self):
        return self.__trail

    @trail.setter
    def trail(self, value):
        self.__trail = value
        if self.engine:
            self.engine.trail = value
            self.engine.single_ball_trail = deque(maxlen=value)

    @GObject.Property(type=str, default="affine")
    def gmc(self):
        return self.__gmc

    @gmc.setter
    def gmc(self, value):
        self.__gmc = value
        if self.engine:
            self.engine.gmc = value

    @GObject.Property(type=float, default=0.5)
    def gmc_scale(self):
        return self.__gmc_scale

    @gmc_scale.setter
    def gmc_scale(self, value):
        self.__gmc_scale = value
        if self.engine:
            self.engine.gmc_scale = value

    @GObject.Property(type=int, default=400)
    def gft_max_corners(self):
        return self.__gft_max_corners

    @gft_max_corners.setter
    def gft_max_corners(self, value):
        self.__gft_max_corners = value
        if self.engine:
            self.engine.gft_max_corners = value

    @GObject.Property(type=float, default=0.01)
    def gft_quality(self):
        return self.__gft_quality

    @gft_quality.setter
    def gft_quality(self, value):
        self.__gft_quality = value
        if self.engine:
            self.engine.gft_quality = value

    @GObject.Property(type=int, default=8)
    def gft_min_dist(self):
        return self.__gft_min_dist

    @gft_min_dist.setter
    def gft_min_dist(self, value):
        self.__gft_min_dist = value
        if self.engine:
            self.engine.gft_min_dist = value

    @GObject.Property(type=int, default=21)
    def lk_win(self):
        return self.__lk_win

    @lk_win.setter
    def lk_win(self, value):
        self.__lk_win = value
        if self.engine:
            self.engine.lk_win = value

    @GObject.Property(type=int, default=3)
    def lk_levels(self):
        return self.__lk_levels

    @lk_levels.setter
    def lk_levels(self, value):
        self.__lk_levels = value
        if self.engine:
            self.engine.lk_levels = value

    @GObject.Property(type=float, default=3.0)
    def ransac_thresh(self):
        return self.__ransac_thresh

    @ransac_thresh.setter
    def ransac_thresh(self, value):
        self.__ransac_thresh = value
        if self.engine:
            self.engine.ransac_thresh = value

    @GObject.Property(type=float, default=0.06)
    def ball_gate_rel(self):
        return self.__ball_gate_rel

    @ball_gate_rel.setter
    def ball_gate_rel(self, value):
        self.__ball_gate_rel = value
        if self.engine:
            self.engine.ball_gate_rel = value

    @GObject.Property(type=int, default=12)
    def ball_gate_min(self):
        return self.__ball_gate_min

    @ball_gate_min.setter
    def ball_gate_min(self, value):
        self.__ball_gate_min = value
        if self.engine:
            self.engine.ball_gate_min = value

    @GObject.Property(type=bool, default=False)
    def ball_gate_use_pred(self):
        return self.__ball_gate_use_pred

    @ball_gate_use_pred.setter
    def ball_gate_use_pred(self, value):
        self.__ball_gate_use_pred = value
        if self.engine:
            self.engine.ball_gate_use_pred = value

    @GObject.Property(type=float, default=0.20)
    def ball_min_iou(self):
        return self.__ball_min_iou

    @ball_min_iou.setter
    def ball_min_iou(self, value):
        self.__ball_min_iou = value
        if self.engine:
            self.engine.ball_min_iou = value

    @GObject.Property(type=float, default=0.12)
    def ball_max_jump_rel(self):
        return self.__ball_max_jump_rel

    @ball_max_jump_rel.setter
    def ball_max_jump_rel(self, value):
        self.__ball_max_jump_rel = value
        if self.engine:
            self.engine.ball_max_jump_rel = value

    @GObject.Property(type=float, default=3.0)
    def ball_speed_mult(self):
        return self.__ball_speed_mult

    @ball_speed_mult.setter
    def ball_speed_mult(self, value):
        self.__ball_speed_mult = value
        if self.engine:
            self.engine.ball_speed_mult = value

    @GObject.Property(type=float, default=0.0)
    def ball_smooth_ema(self):
        return self.__ball_smooth_ema

    @ball_smooth_ema.setter
    def ball_smooth_ema(self, value):
        self.__ball_smooth_ema = value
        if self.engine:
            self.engine.ball_smooth_ema = value

    @GObject.Property(type=float, default=0.28)
    def det_override_conf(self):
        return self.__det_override_conf

    @det_override_conf.setter
    def det_override_conf(self, value):
        self.__det_override_conf = value
        if self.engine:
            self.engine.det_override_conf = value

    @GObject.Property(type=int, default=2)
    def det_override_after(self):
        return self.__det_override_after

    @det_override_after.setter
    def det_override_after(self, value):
        self.__det_override_after = value
        if self.engine:
            self.engine.det_override_after = value

    @GObject.Property(type=int, default=6)
    def reacquire_frames(self):
        return self.__reacquire_frames

    @reacquire_frames.setter
    def reacquire_frames(self, value):
        self.__reacquire_frames = value
        if self.engine:
            self.engine.reacquire_frames = value

    @GObject.Property(type=bool, default=False)
    def ball_coast(self):
        return self.__ball_coast

    @ball_coast.setter
    def ball_coast(self, value):
        self.__ball_coast = value
        if self.engine:
            self.engine.ball_coast = value

    @GObject.Property(type=int, default=6)
    def coast_max(self):
        return self.__coast_max

    @coast_max.setter
    def coast_max(self, value):
        self.__coast_max = value
        if self.engine:
            self.engine.coast_max = value

    @GObject.Property(type=float, default=0.90)
    def coast_decay(self):
        return self.__coast_decay

    @coast_decay.setter
    def coast_decay(self, value):
        self.__coast_decay = value
        if self.engine:
            self.engine.coast_decay = value

    @GObject.Property(type=bool, default=False)
    def verbose(self):
        return self.__verbose

    @verbose.setter
    def verbose(self, value):
        self.__verbose = value
        if self.engine:
            self.engine.verbose = value

    @GObject.Property(type=float, default=30.0)
    def frame_rate(self):
        return self.__frame_rate

    @frame_rate.setter
    def frame_rate(self, value):
        self.__frame_rate = value
        if self.engine:
            self.engine.frame_rate = value

    def set_model(self):
        """Override: Create engine first, then load with all properties as kwargs."""
        if self.engine is None:
            self.initialize_engine()
        if self.engine is None:
            self.logger.error("Cannot load model: engine not initialized")
            return False
        # Sync all properties to kwargs for engine load
        kwargs = {
            "classes": self.classes,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "person_conf_keep": self.person_conf_keep,
            "ball_conf_keep": self.ball_conf_keep,
            "ball_mode": self.ball_mode,
            "hires_fallback": self.hires_fallback,
            "hires_imgsz": self.hires_imgsz,
            "fallback_every": self.fallback_every,
            "fallback_tiles": self.fallback_tiles,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "fallback_budget_ms": self.fallback_budget_ms,
            "ball_roi_boost": self.ball_roi_boost,
            "roi_scale": self.roi_scale,
            "roi_min": self.roi_min,
            "roi_max": self.roi_max,
            "tracker_people": self.tracker_people,
            "tracker_ball": self.tracker_ball,
            "people_reid": self.people_reid,
            "trail": self.trail,
            "gmc": self.gmc,
            "gmc_scale": self.gmc_scale,
            "gft_max_corners": self.gft_max_corners,
            "gft_quality": self.gft_quality,
            "gft_min_dist": self.gft_min_dist,
            "lk_win": self.lk_win,
            "lk_levels": self.lk_levels,
            "ransac_thresh": self.ransac_thresh,
            "ball_gate_rel": self.ball_gate_rel,
            "ball_gate_min": self.ball_gate_min,
            "ball_gate_use_pred": self.ball_gate_use_pred,
            "ball_min_iou": self.ball_min_iou,
            "ball_max_jump_rel": self.ball_max_jump_rel,
            "ball_speed_mult": self.ball_speed_mult,
            "ball_smooth_ema": self.ball_smooth_ema,
            "det_override_conf": self.det_override_conf,
            "det_override_after": self.det_override_after,
            "reacquire_frames": self.reacquire_frames,
            "ball_coast": self.ball_coast,
            "coast_max": self.coast_max,
            "coast_decay": self.coast_decay,
            "verbose": self.verbose,
            "frame_rate": self.frame_rate,
        }
        # Call engine's do_load_model with current model name + synced kwargs
        return self.engine.do_load_model(self.model, **kwargs)

    def do_decode(self, buf, result, stream_idx=0):
        self.logger.debug(
            f"Decoding advanced YOLO result for buffer {hex(id(buf))}, stream {stream_idx}: {result}"
        )
        tracks_p = result.tracks_p
        tracks_b = result.tracks_b
        ball_trail = result.ball_trail
        boxes = result.boxes

        meta = analytics.add_relation_meta(buf)
        if not meta:
            self.logger.error(
                f"Stream {stream_idx} - Failed to add analytics relation metadata"
            )
            return

        self.logger.debug(
            f"Stream {stream_idx} - Attaching metadata for {len(tracks_p) + len(tracks_b)} tracks"
        )

        # Person tracks (unchanged)
        for tr in tracks_p:
            box = tlbr_of(tr)
            if not box:
                continue
            x1, y1, x2, y2 = box
            score = 1.0  # Track confidence
            track_id = getattr(tr, "track_id", 0)
            qk_string = f"stream_{stream_idx}_person_id_{track_id}"
            od_mtd = analytics.add_object(
                meta,
                qk_string,
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                score,
            )
            if od_mtd is None:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add person detection metadata"
                )
                continue
            self.logger.debug(
                f"Stream {stream_idx} - Added person od_mtd: id={track_id}, x1={x1}, y1={y1}, w={x2-x1}, h={y2-y1}, score={score}"
            )

            tracking_mtd = analytics.add_tracking(meta, track_id)
            if tracking_mtd is None:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add person tracking metadata"
                )
                continue
            ret = analytics.relate(meta, od_mtd, tracking_mtd)
            if not ret:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to relate person od and tracking metadata"
                )
            else:
                self.logger.debug(
                    f"Stream {stream_idx} - Linked person od_mtd {od_mtd} to tracking_mtd {tracking_mtd}"
                )

        # Ball tracks (unchanged)
        for i, tr in enumerate(tracks_b):
            box = tlbr_of(tr)
            if not box:
                continue
            x1, y1, x2, y2 = box
            score = 1.0
            track_id = getattr(tr, "track_id", 0)
            qk_string = f"stream_{stream_idx}_ball_id_{track_id}"
            od_mtd = analytics.add_object(
                meta,
                qk_string,
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                score,
            )
            if od_mtd is None:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add ball detection metadata"
                )
                continue
            self.logger.debug(
                f"Stream {stream_idx} - Added ball od_mtd: id={track_id}, x1={x1}, y1={y1}, w={x2-x1}, h={y2-y1}, score={score}"
            )

            tracking_mtd = analytics.add_tracking(meta, track_id)
            if tracking_mtd is None:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add ball tracking metadata"
                )
                continue
            ret = analytics.relate(meta, od_mtd, tracking_mtd)
            if not ret:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to relate ball od and tracking metadata"
                )
            else:
                self.logger.debug(
                    f"Stream {stream_idx} - Linked ball od_mtd {od_mtd} to tracking_mtd {tracking_mtd}"
                )

        # Ball trail - attach as custom GstStructure meta (fixed API, uncommented)
        if ball_trail:
            structure = Gst.Structure.new_empty("ball-trail")
            trail_data = [(int(x), int(y), int(k)) for x, y, k in ball_trail]
            structure.set_value("points", trail_data)
            structure.set_value("length", len(ball_trail))

            # Correct API: Use Gst.Buffer.add_meta with GstMeta for structure
            try:
                # Get generic meta API (GstMeta for any custom data)
                meta_api = Gst.Meta.get_api(
                    Gst.StructureMeta
                )  # Or Gst.Meta.api_type_get_tag(Gst.StructureMeta) if available
                if meta_api:
                    # Create custom meta with structure
                    meta = Gst.Meta.new(buf, meta_api, structure)
                    if meta:
                        Gst.Buffer.add_meta(buf, meta_api, meta)
                        self.logger.debug(
                            f"Stream {stream_idx} - Added ball trail meta with {len(ball_trail)} points"
                        )
                    else:
                        raise ValueError("Failed to create meta")
                else:
                    raise AttributeError("Meta API not available")
            except (AttributeError, ValueError, TypeError) as e:
                self.logger.warning(
                    f"Ball trail meta attachment failed ({e}); logging instead"
                )
                self.logger.info(
                    f"Ball trail for stream {stream_idx}: {trail_data[:5]}... (length {len(ball_trail)})"
                )

        # Fallback to original boxes if needed (unchanged)
        if boxes is not None and len(boxes) > 0:
            # Add any non-person/ball or raw detections if desired
            pass  # Optional

        attached_meta = analytics.get_relation_meta(buf)
        if attached_meta:
            count = analytics.relation_length(attached_meta)
            self.logger.info(
                f"Stream {stream_idx} - Advanced metadata attached to buffer {hex(id(buf))}: {count} relations, ball trail: {len(ball_trail)}"
            )
        else:
            self.logger.error(
                f"Stream {stream_idx} - Metadata not attached to buffer after adding"
            )


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("demo_soccer", DemoSoccer)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'demo_soccer' element will not be registered because required modules are missing."
    )
