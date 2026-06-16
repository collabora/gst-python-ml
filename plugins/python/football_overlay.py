# FootballOverlay
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

import os

from log.global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import re
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    gi.require_version("GstAnalytics", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GstBase, GstVideo, GstAnalytics, GObject, GLib  # noqa: E402

    from log.logger_factory import LoggerFactory  # noqa: E402

    OVERLAY_CAPS = Gst.Caps.from_string(
        "video/x-raw, format=(string){ RGBA, ARGB, BGRA, ABGR }"
    )

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_football_overlay' element will not be available. Error: {e}"
    )


_FORMAT_ORDER = {
    "RGBA": (0, 1, 2, 3),
    "ARGB": (3, 0, 1, 2),
    "BGRA": (2, 1, 0, 3),
    "ABGR": (3, 2, 1, 0),
}

_PALETTE = [
    (239, 71, 111, 255),
    (255, 209, 102, 255),
    (6, 214, 160, 255),
    (17, 138, 178, 255),
    (255, 107, 107, 255),
    (78, 205, 196, 255),
    (199, 125, 255, 255),
    (255, 159, 28, 255),
    (46, 196, 182, 255),
    (118, 200, 247, 255),
]

_REFEREE_RGBA = (255, 215, 0, 255)
_BALL_RGBA = (0, 230, 0, 255)
_PLAYER_RGBA = (0, 200, 255, 255)
_DEFAULT_RGBA = (235, 235, 235, 255)
_BLACK_RGBA = (0, 0, 0, 255)
_HUD_BG_RGBA = (92, 41, 131, 255)
_HUD_TEXT_RGBA = (64, 186, 47, 255)


def _is_ball(label):
    return "ball" in label


def _is_referee(label):
    return "referee" in label or label == "ref"


class FootballOverlay(GstBase.BaseTransform):
    """
    Metadata-driven broadcast overlay (football_analysis style), streaming.

    Reads upstream GstAnalytics detection/tracking metadata and draws: an
    ellipse + optional id badge per subject, a gold ellipse for referees, a
    green triangle on the ball, fading motion trails, and a focal-player HUD
    with a headshot, accumulated ball contacts, and distance travelled.
    """

    __gstmetadata__ = (
        "Football Overlay",
        "Filter/Effect/Video",
        "Broadcast-style detection/tracking overlay (ellipses, ball triangle, "
        "trails, headshot HUD with ball contacts + distance) from GstAnalytics",
        "Marcus Edel <marcus.edel@collabora.com>",
    )

    src_template = Gst.PadTemplate.new(
        "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, OVERLAY_CAPS.copy()
    )
    sink_template = Gst.PadTemplate.new(
        "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, OVERLAY_CAPS.copy()
    )
    __gsttemplates__ = (src_template, sink_template)

    show_labels = GObject.Property(
        type=bool, default=True, nick="Show Labels",
        blurb="Draw the class name above each object",
        flags=GObject.ParamFlags.READWRITE)
    show_ids = GObject.Property(
        type=bool, default=True, nick="Show Track IDs",
        blurb="Draw the track-id badge under each tracked object",
        flags=GObject.ParamFlags.READWRITE)
    trails = GObject.Property(
        type=bool, default=True, nick="Show Trails",
        blurb="Draw a fading motion trail behind each tracked object",
        flags=GObject.ParamFlags.READWRITE)
    trail_length = GObject.Property(
        type=int, default=30, minimum=2, maximum=300, nick="Trail Length",
        blurb="Number of recent positions kept in each motion trail",
        flags=GObject.ParamFlags.READWRITE)
    show_hud = GObject.Property(
        type=bool, default=True, nick="Show HUD",
        blurb="Draw the focal-player HUD (headshot, label, contacts, distance)",
        flags=GObject.ParamFlags.READWRITE)
    headshot_path = GObject.Property(
        type=str, default="data/Chinedu-Obasi_2684938.jpg", nick="Headshot Path",
        blurb="Image shown in the HUD (empty to disable)",
        flags=GObject.ParamFlags.READWRITE)
    headshot_size = GObject.Property(
        type=int, default=90, minimum=16, maximum=512, nick="Headshot Size",
        blurb="Headshot square size in pixels",
        flags=GObject.ParamFlags.READWRITE)
    player_label = GObject.Property(
        type=str, default="Player #8", nick="Player Label",
        blurb="Static label drawn in the HUD",
        flags=GObject.ParamFlags.READWRITE)
    contact_pad_ratio = GObject.Property(
        type=float, default=0.25, minimum=0.0, maximum=5.0, nick="Contact Pad Ratio",
        blurb="Ball counts as a contact within this fraction of the player box size",
        flags=GObject.ParamFlags.READWRITE)
    contact_gap_frames = GObject.Property(
        type=int, default=5, minimum=0, maximum=1000, nick="Contact Gap Frames",
        blurb="Min frames between counted contacts for the same player",
        flags=GObject.ParamFlags.READWRITE)
    player_height = GObject.Property(
        type=float, default=1.8, minimum=0.1, maximum=10.0, nick="Player Height (m)",
        blurb="Assumed real-world height used to convert pixels to metres",
        flags=GObject.ParamFlags.READWRITE)
    min_confidence = GObject.Property(
        type=float, default=0.0, minimum=0.0, maximum=1.0, nick="Min Confidence",
        blurb="Skip detections whose confidence is below this threshold",
        flags=GObject.ParamFlags.READWRITE)
    class_names = GObject.Property(
        type=str, default="", nick="Class Names",
        blurb="Comma-separated names to map numeric labels (label_N) from the "
              "onnx/objectdetector path, e.g. 'ball,goalkeeper,player,referee'",
        flags=GObject.ParamFlags.READWRITE)

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_in_place(True)
        self.width = 0
        self.height = 0
        self._order = _FORMAT_ORDER["RGBA"]
        # per-track state, accumulated across frames
        self._trail = {}
        self._last_pt = {}
        self._distance_px = {}
        self._heights = []
        self._contacts = {}
        self._last_contact_frame = {}
        self._frames_seen = {}
        self._track_label = {}
        self._frame = 0
        self._headshot = None
        self._headshot_loaded = False

    def do_set_caps(self, incaps, outcaps):
        info = GstVideo.VideoInfo.new_from_caps(incaps)
        self.width = info.width
        self.height = info.height
        fmt = info.finfo.name if info.finfo else "RGBA"
        self._order = _FORMAT_ORDER.get(fmt, _FORMAT_ORDER["RGBA"])
        self._headshot_loaded = False  # re-load in the new channel order
        self.logger.info(f"FootballOverlay caps: {fmt} {self.width}x{self.height}")
        return True

    def _map_label(self, label):
        if self.class_names:
            m = re.match(r"label_(\d+)$", label)
            if m:
                names = [s.strip() for s in self.class_names.split(",") if s.strip()]
                i = int(m.group(1))
                if 0 <= i < len(names):
                    return names[i]
        return label

    def _parse_label(self, full_label):
        core = full_label
        m = re.match(r"stream_\d+_(.*)$", full_label)
        if m:
            core = m.group(1)
        m = re.match(r"(.+)_id_(\d+)$", core)
        if m:
            return self._map_label(m.group(1)), int(m.group(2))
        m = re.match(r"id_(\d+)$", core)
        if m:
            return "object", int(m.group(1))
        return self._map_label(core or "object"), None

    def _read_metadata(self, buf):
        entries = []
        meta = GstAnalytics.buffer_get_analytics_relation_meta(buf)
        if not meta:
            return entries
        for index in range(GstAnalytics.relation_get_length(meta)):
            ret, od_mtd = meta.get_od_mtd(index)
            if not ret or od_mtd is None:
                continue
            full_label = GLib.quark_to_string(od_mtd.get_obj_type())
            presence, x, y, w, h, score = od_mtd.get_location()
            if not presence:
                continue
            label, track_id = self._parse_label(full_label)
            entries.append({
                "label": label.lower(), "track_id": track_id,
                "confidence": score, "box": (x, y, x + w, y + h),
            })
        return entries

    @staticmethod
    def _point_to_bbox_distance(px, py, box):
        x1, y1, x2, y2 = box
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        return (dx * dx + dy * dy) ** 0.5

    def _ball_contact(self, players, ball_box):
        """Closest player to the ball, if within contact_pad_ratio of its size."""
        bx = (ball_box[0] + ball_box[2]) / 2.0
        by = (ball_box[1] + ball_box[3]) / 2.0
        best_tid, best_d, best_box = None, float("inf"), None
        for tid, box in players.items():
            d = self._point_to_bbox_distance(bx, by, box)
            if d < best_d:
                best_tid, best_d, best_box = tid, d, box
        if best_box is None:
            return None
        w = best_box[2] - best_box[0]
        h = best_box[3] - best_box[1]
        if best_d > self.contact_pad_ratio * max(w, h):
            return None
        return best_tid

    def _update_tracks(self, entries):
        self._frame += 1
        active = set()
        players = {}
        ball_box = None
        for e in entries:
            tid = e["track_id"]
            if tid is None:
                continue
            if _is_ball(e["label"]):
                ball_box = e["box"]
                continue
            active.add(tid)
            players[tid] = e["box"]
            self._track_label[tid] = e["label"]
            self._frames_seen[tid] = self._frames_seen.get(tid, 0) + 1
            x1, y1, x2, y2 = e["box"]
            foot = (int((x1 + x2) / 2), int(y2))
            if y2 - y1 > 0:
                self._heights.append(y2 - y1)
                if len(self._heights) > 600:
                    self._heights = self._heights[-600:]
            prev = self._last_pt.get(tid)
            if prev is not None:
                self._distance_px[tid] = self._distance_px.get(tid, 0.0) + (
                    (foot[0] - prev[0]) ** 2 + (foot[1] - prev[1]) ** 2
                ) ** 0.5
            self._last_pt[tid] = foot
            trail = self._trail.setdefault(tid, [])
            trail.append(foot)
            if len(trail) > self.trail_length:
                del trail[: -self.trail_length]

        # Ball contacts (debounced per player), like football_analyzer.
        if ball_box is not None and players:
            tid = self._ball_contact(players, ball_box)
            if tid is not None:
                last = self._last_contact_frame.get(tid)
                if last is None or (self._frame - last) > self.contact_gap_frames:
                    self._contacts[tid] = self._contacts.get(tid, 0) + 1
                self._last_contact_frame[tid] = self._frame

        for tid in list(self._trail.keys()):
            if tid not in active:
                del self._trail[tid]
                self._last_pt.pop(tid, None)
        return active

    def _px_per_meter(self):
        if not self._heights:
            return None
        import numpy as np
        return float(np.median(self._heights)) / max(0.1, self.player_height)

    def _focal_track(self):
        keys = set(self._frames_seen)
        if not keys:
            return None
        if any(self._contacts.values()):
            return max(keys, key=lambda t: (self._contacts.get(t, 0),
                                            self._frames_seen.get(t, 0)))
        return max(keys, key=lambda t: self._frames_seen.get(t, 0))

    def _c(self, rgba):
        return tuple(rgba[i] for i in self._order)

    def _color_for(self, label, track_id):
        if _is_referee(label):
            return _REFEREE_RGBA
        if _is_ball(label):
            return _BALL_RGBA
        return _PLAYER_RGBA

    def _load_headshot(self, cv2, np):
        if self._headshot_loaded:
            return self._headshot
        self._headshot_loaded = True
        self._headshot = None
        path = self.headshot_path
        if not path or not os.path.exists(path):
            if path:
                self.logger.warning(f"headshot not found: {path}")
            return None
        img = cv2.imread(path)  # BGR
        if img is None:
            return None
        sz = int(self.headshot_size)
        img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)
        rgb = img[:, :, ::-1]  # BGR -> RGB
        alpha = np.full((sz, sz, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=2).astype(np.uint8)  # logical RGBA

        self._headshot = np.ascontiguousarray(rgba[:, :, list(self._order)])
        return self._headshot

    def _draw_trail(self, cv2, np, frame, points, rgba):
        if len(points) < 2:
            return
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, self._c(rgba), 2, cv2.LINE_AA)

    def _draw_ellipse(self, cv2, frame, box, rgba, track_id):
        x1, y1, x2, y2 = box
        y_bottom = int(y2)
        x_center = int((x1 + x2) / 2)
        width = max(1, int(x2 - x1))
        color = self._c(rgba)
        cv2.ellipse(frame, (x_center, y_bottom),
                    (width, max(1, int(0.35 * width))), 0.0, -45, 235,
                    color, 2, cv2.LINE_AA)
        if self.show_ids and track_id is not None:
            rect_w, rect_h = 40, 18
            x1r = x_center - rect_w // 2
            x2r = x_center + rect_w // 2
            y1r = y_bottom - rect_h // 2 + 15
            y2r = y_bottom + rect_h // 2 + 15
            cv2.rectangle(frame, (x1r, y1r), (x2r, y2r), color, cv2.FILLED)
            tx = x1r + 12 - (10 if track_id > 99 else 0)
            cv2.putText(frame, str(track_id), (tx, y1r + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._c(_BLACK_RGBA), 2,
                        cv2.LINE_AA)

    def _draw_triangle(self, cv2, np, frame, box, rgba):
        x1, y1, x2, y2 = box
        x = int((x1 + x2) / 2)
        y = int(y1)
        pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]], dtype=np.int32)
        cv2.drawContours(frame, [pts], 0, self._c(rgba), cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, self._c(_BLACK_RGBA), 2)

    def _draw_label(self, cv2, frame, box, label, rgba):
        x1, y1, _, _ = box
        cv2.putText(frame, label, (int(x1), max(12, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._c(rgba), 1, cv2.LINE_AA)

    def _draw_hud(self, cv2, frame, contacts, distance_m, rgba, headshot):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = 10, 10
        if headshot is not None:
            hh, hw = headshot.shape[:2]
            w, h = hw + 280, max(110, hh + 20)
            text_x = x + hw + 20
        else:
            w, h = 320, 100
            text_x = x + 12
        cv2.rectangle(frame, (x, y), (x + w, y + h), self._c(_HUD_BG_RGBA), cv2.FILLED)
        cv2.rectangle(frame, (x, y), (x + w, y + h), self._c(rgba), 2)
        if headshot is not None:
            hy, hx = y + 10, x + 10
            fh, fw = frame.shape[:2]
            hh = min(hh, fh - hy)
            hw = min(hw, fw - hx)
            if hh > 0 and hw > 0:
                frame[hy:hy + hh, hx:hx + hw] = headshot[:hh, :hw]
                cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), self._c(rgba), 2)
        tc = self._c(_HUD_TEXT_RGBA)
        cv2.putText(frame, self.player_label, (text_x, y + 28), font, 0.7, tc, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Ball contacts: {contacts}", (text_x, y + 58), font, 0.6, tc, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Distance: {distance_m:.1f} m", (text_x, y + 85), font, 0.6, tc, 1, cv2.LINE_AA)

    def do_transform_ip(self, buf):
        try:
            import numpy as np

            entries = self._read_metadata(buf)
            if self.min_confidence > 0.0:
                entries = [e for e in entries if e["confidence"] >= self.min_confidence]
            if any(e["track_id"] is not None for e in entries):
                entries = [e for e in entries if e["track_id"] is not None]

            active = self._update_tracks(entries)
            if not entries:
                return Gst.FlowReturn.OK

            import cv2

            ok, mapinfo = buf.map(Gst.MapFlags.WRITE)
            if not ok:
                self.logger.error("Failed to map buffer for writing")
                return Gst.FlowReturn.ERROR
            try:
                frame = np.frombuffer(
                    mapinfo.data, dtype=np.uint8, count=self.height * self.width * 4
                ).reshape(self.height, self.width, 4)

                if self.trails:
                    for tid in active:
                        self._draw_trail(
                            cv2, np, frame, self._trail.get(tid, []),
                            self._color_for(self._track_label.get(tid, ""), tid))

                for e in entries:
                    label = e["label"]
                    box = e["box"]
                    if _is_ball(label):
                        self._draw_triangle(cv2, np, frame, box, _BALL_RGBA)
                        continue
                    rgba = self._color_for(label, e["track_id"])
                    self._draw_ellipse(cv2, frame, box, rgba, e["track_id"])
                    if self.show_labels:
                        self._draw_label(cv2, frame, box, label, rgba)

                if self.show_hud:
                    focal = self._focal_track()
                    if focal is not None:
                        ppm = self._px_per_meter()
                        dist_m = (self._distance_px.get(focal, 0.0) / ppm) if ppm else 0.0
                        self._draw_hud(
                            cv2, frame, self._contacts.get(focal, 0), dist_m,
                            self._color_for(self._track_label.get(focal, ""), focal),
                            self._load_headshot(cv2, np),
                        )
            finally:
                buf.unmap(mapinfo)

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"FootballOverlay transform error: {e}")
            return Gst.FlowReturn.ERROR


if CAN_REGISTER_ELEMENT:
    GObject.type_register(FootballOverlay)
    __gstelementfactory__ = (
        "pyml_football_overlay",
        Gst.Rank.NONE,
        FootballOverlay,
    )
else:
    GlobalLogger().warning(
        "The 'pyml_football_overlay' element will not be registered because "
        "required modules are missing."
    )
