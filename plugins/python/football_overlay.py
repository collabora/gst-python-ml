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
    from gi.repository import (
        Gst,
        GstBase,
        GstVideo,
        GstAnalytics,
        GObject,
        GLib,
    )  # noqa: E402

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
_RED_TEAM_RGBA = (255, 40, 40, 255)
_BLUE_TEAM_RGBA = (40, 90, 255, 255)
_DEFAULT_RGBA = (235, 235, 235, 255)
_BLACK_RGBA = (0, 0, 0, 255)
_HUD_BG_RGBA = (92, 41, 131, 255)
_HUD_TEXT_RGBA = (64, 186, 47, 255)
_HIGHLIGHT_RGBA = (255, 255, 255, 255)


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
        type=bool,
        default=True,
        nick="Show Labels",
        blurb="Draw the class name above each object",
        flags=GObject.ParamFlags.READWRITE,
    )
    show_ids = GObject.Property(
        type=bool,
        default=True,
        nick="Show Track IDs",
        blurb="Draw the track-id badge under each tracked object",
        flags=GObject.ParamFlags.READWRITE,
    )
    trails = GObject.Property(
        type=bool,
        default=True,
        nick="Show Trails",
        blurb="Draw a fading motion trail behind each tracked object",
        flags=GObject.ParamFlags.READWRITE,
    )
    trail_length = GObject.Property(
        type=int,
        default=30,
        minimum=2,
        maximum=300,
        nick="Trail Length",
        blurb="Number of recent positions kept in each motion trail",
        flags=GObject.ParamFlags.READWRITE,
    )
    show_ball = GObject.Property(
        type=bool,
        default=False,
        nick="Show Ball",
        blurb="Draw the marker on the ball (the ball is still tracked for "
        "contact counting either way)",
        flags=GObject.ParamFlags.READWRITE,
    )
    show_hud = GObject.Property(
        type=bool,
        default=True,
        nick="Show HUD",
        blurb="Draw the focal-player HUD (headshot, label, contacts, distance)",
        flags=GObject.ParamFlags.READWRITE,
    )
    headshot_path = GObject.Property(
        type=str,
        default="data/Chinedu-Obasi_2684938.jpg",
        nick="Headshot Path",
        blurb="Image shown in the HUD (empty to disable)",
        flags=GObject.ParamFlags.READWRITE,
    )
    headshot_size = GObject.Property(
        type=int,
        default=90,
        minimum=16,
        maximum=512,
        nick="Headshot Size",
        blurb="Headshot square size in pixels",
        flags=GObject.ParamFlags.READWRITE,
    )
    player_label = GObject.Property(
        type=str,
        default="Player #8",
        nick="Player Label",
        blurb="Static label drawn in the HUD",
        flags=GObject.ParamFlags.READWRITE,
    )
    contact_pad_ratio = GObject.Property(
        type=float,
        default=0.25,
        minimum=0.0,
        maximum=5.0,
        nick="Contact Pad Ratio",
        blurb="Ball counts as a contact within this fraction of the player box size",
        flags=GObject.ParamFlags.READWRITE,
    )
    contact_gap_frames = GObject.Property(
        type=int,
        default=5,
        minimum=0,
        maximum=1000,
        nick="Contact Gap Frames",
        blurb="Min frames between counted contacts for the same player",
        flags=GObject.ParamFlags.READWRITE,
    )
    player_height = GObject.Property(
        type=float,
        default=1.8,
        minimum=0.1,
        maximum=10.0,
        nick="Player Height (m)",
        blurb="Assumed real-world height used to convert pixels to metres",
        flags=GObject.ParamFlags.READWRITE,
    )
    min_confidence = GObject.Property(
        type=float,
        default=0.0,
        minimum=0.0,
        maximum=1.0,
        nick="Min Confidence",
        blurb="Skip detections whose confidence is below this threshold",
        flags=GObject.ParamFlags.READWRITE,
    )
    class_names = GObject.Property(
        type=str,
        default="",
        nick="Class Names",
        blurb="Comma-separated names to map numeric labels (label_N) from the "
        "onnx/objectdetector path, e.g. 'ball,goalkeeper,player,referee'",
        flags=GObject.ParamFlags.READWRITE,
    )
    team_colors = GObject.Property(
        type=bool,
        default=True,
        nick="Team Colors",
        blurb="Colour players by jersey team (red/blue, per-track majority vote); "
        "off draws all players one colour",
        flags=GObject.ParamFlags.READWRITE,
    )
    draw_from_detections = GObject.Property(
        type=bool,
        default=False,
        nick="Draw From Detections",
        blurb="Draw ellipses on the raw per-frame detection boxes instead of the "
        "tracker's boxes -- no Kalman drift, coasted phantoms or track-split "
        "doubles. Team colour is then classified per frame; the HUD still uses "
        "tracker metadata if present",
        flags=GObject.ParamFlags.READWRITE,
    )
    merge_iou = GObject.Property(
        type=float,
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        nick="Merge IoU",
        blurb="Collapse overlapping boxes (across classes) into one before "
        "drawing, so one player isn't circled twice; a box is merged when its "
        "IoU or containment with a kept box exceeds this (0 disables)",
        flags=GObject.ParamFlags.READWRITE,
    )
    position_smoothing = GObject.Property(
        type=float,
        default=0.5,
        minimum=0.0,
        maximum=0.95,
        nick="Position Smoothing",
        blurb="Temporal EMA on drawn box positions (0=off, higher=smoother but "
        "more lag). Boxes are associated frame-to-frame by proximity, so this "
        "damps detection jitter and the steps from a detection interval > 1",
        flags=GObject.ParamFlags.READWRITE,
    )
    highlight_focal = GObject.Property(
        type=bool,
        default=True,
        nick="Highlight Focal Player",
        blurb="Mark the focal player (the one shown in the HUD) on the pitch "
        "with a chevron above their head and a bolder ellipse",
        flags=GObject.ParamFlags.READWRITE,
    )
    focal_track_id = GObject.Property(
        type=int,
        default=-1,
        minimum=-1,
        maximum=100000,
        nick="Focal Track ID",
        blurb="Pin the focal/highlighted player to this track id; -1 = auto "
        "(the player tracked the most, with hysteresis so it stays stable)",
        flags=GObject.ParamFlags.READWRITE,
    )

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
        self._widths = []
        self._ell_w = {}  # track_id -> smoothed ellipse half-width (px)
        self._contacts = {}
        self._last_contact_frame = {}
        self._frames_seen = {}
        self._track_label = {}
        self._class_votes = {}  # track_id -> {label: count}, for stable class
        self._team_votes = {}  # track_id -> {"red": n, "blue": n}, jersey team
        self._frame = 0
        self._focal = None  # current focal track id (sticky, for hysteresis)
        self._headshot = None
        self._headshot_loaded = False
        self._inv_order = [0, 1, 2, 3]  # buffer-channel -> logical RGBA index
        # Position-smoothing slots: {"box": np[x1,y1,x2,y2]} kept across frames
        # and matched by proximity, so the drawn ellipse can be low-passed.
        self._smooth_slots = []

    def do_set_caps(self, incaps, outcaps):
        info = GstVideo.VideoInfo.new_from_caps(incaps)
        self.width = info.width
        self.height = info.height
        fmt = info.finfo.name if info.finfo else "RGBA"
        self._order = _FORMAT_ORDER.get(fmt, _FORMAT_ORDER["RGBA"])
        # buffer channel j holds logical[self._order[j]]; invert so we can pull
        # logical R,G,B out of the buffer for jersey colour classification.
        self._inv_order = [self._order.index(c) for c in range(4)]
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
            entries.append(
                {
                    "label": label.lower(),
                    "track_id": track_id,
                    "confidence": score,
                    "box": (x, y, x + w, y + h),
                }
            )
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

    def _update_tracks(self, entries, det_ball_box=None):
        self._frame += 1
        active = set()
        players = {}
        ball_box = None
        # Accumulate per-track class votes first so the stable label below
        # already reflects this frame.
        for e in entries:
            tid = e["track_id"]
            if tid is None:
                continue
            v = self._class_votes.setdefault(tid, {})
            v[e["label"]] = v.get(e["label"], 0) + 1
        for e in entries:
            tid = e["track_id"]
            if tid is None:
                continue
            label = self._stable_label(tid, e["label"])
            if _is_ball(label):
                ball_box = e["box"]
                continue
            active.add(tid)
            players[tid] = e["box"]
            self._track_label[tid] = label
            self._frames_seen[tid] = self._frames_seen.get(tid, 0) + 1
            x1, y1, x2, y2 = e["box"]
            foot = (int((x1 + x2) / 2), int(y2))
            if y2 - y1 > 0:
                self._heights.append(y2 - y1)
                if len(self._heights) > 600:
                    self._heights = self._heights[-600:]
            self._update_ellipse_width(tid, x2 - x1)
            prev = self._last_pt.get(tid)
            if prev is not None:
                self._distance_px[tid] = (
                    self._distance_px.get(tid, 0.0)
                    + ((foot[0] - prev[0]) ** 2 + (foot[1] - prev[1]) ** 2) ** 0.5
                )
            self._last_pt[tid] = foot
            trail = self._trail.setdefault(tid, [])
            trail.append(foot)
            if len(trail) > self.trail_length:
                del trail[: -self.trail_length]

        # Fall back to the detected ball if no tracked ball this frame.
        if ball_box is None:
            ball_box = det_ball_box

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
                self._ell_w.pop(tid, None)
        return active

    def _update_ellipse_width(self, track_id, raw_w):
        # Smooth (and outlier-reject) the per-track ellipse width so a single
        # oversized box -- two players merged, or a drifting keep-alive
        # prediction -- can't balloon the circle for one frame.
        if raw_w <= 0:
            return
        if self._widths:
            self._widths.append(raw_w)
            if len(self._widths) > 600:
                self._widths = self._widths[-600:]
            srt = sorted(self._widths)
            med = srt[len(srt) // 2]
            clamped = min(max(raw_w, 0.5 * med), 1.8 * med)
        else:
            self._widths.append(raw_w)
            clamped = raw_w
        prev = self._ell_w.get(track_id)
        # EMA: slow enough to keep the circle size steady frame-to-frame, fast
        # enough to still follow real perspective changes as players move.
        self._ell_w[track_id] = (
            clamped if prev is None else 0.25 * clamped + 0.75 * prev
        )

    def _px_per_meter(self):
        if not self._heights:
            return None
        import numpy as np

        return float(np.median(self._heights)) / max(0.1, self.player_height)

    def _focal_track(self):
        # Pin to an explicit track id if requested.
        if self.focal_track_id >= 0:
            return (
                self.focal_track_id
                if self.focal_track_id in self._frames_seen
                else self._focal
            )
        keys = set(self._frames_seen)
        if not keys:
            return None

        # Only consider *sustained* tracks. Otherwise a track that flickered for
        # a few frames -- common when detection/tracking churns -- can win on a
        # single ball contact and then show ~0 distance (it was barely tracked).
        # The floor scales with elapsed frames, with a small absolute minimum.
        floor = max(10, int(0.2 * self._frame))
        candidates = [t for t in keys if self._frames_seen.get(t, 0) >= floor] or list(
            keys
        )

        # Rank by ball contacts (the player most involved with the ball), with
        # frames-seen as a tiebreak / pre-contact fallback (before anyone has
        # touched the ball, the most-tracked player is shown).
        def score(t):
            return (self._contacts.get(t, 0), self._frames_seen.get(t, 0))

        best = max(candidates, key=score)
        # Stability: keep the current focal unless a challenger has *strictly
        # more* contacts, so the highlight/HUD don't flip on ties or noise.
        cur = self._focal
        if (
            cur is not None
            and cur in candidates
            and self._contacts.get(best, 0) <= self._contacts.get(cur, 0)
        ):
            best = cur
        self._focal = best
        return best

    def _stable_label(self, track_id, fallback=""):
        # Majority-voted class over the track's history — smooths frame-to-frame
        # misclassifications (e.g. a player briefly tagged 'referee'), so the
        # gold referee marking doesn't flicker.
        votes = self._class_votes.get(track_id)
        if not votes:
            return fallback
        return max(votes, key=votes.get)

    def _c(self, rgba):
        return tuple(rgba[i] for i in self._order)

    def _team_color(self, track_id):
        # Confident kit colour from a track's accumulated jersey votes, else
        # None. Red/blue -> team; "ref" (distinctive non-team kit) -> gold.
        # Requires a minimum number of votes AND a clear majority, so a few
        # noisy frames can't decide the colour.
        if track_id is None:
            return None
        c = self._team_votes.get(track_id)
        if not c:
            return None
        red, blue, ref = c.get("red", 0), c.get("blue", 0), c.get("ref", 0)
        total = red + blue + ref
        if total < 4:
            return None
        colors = {_RED_TEAM_RGBA: red, _BLUE_TEAM_RGBA: blue, _REFEREE_RGBA: ref}
        color, n = max(colors.items(), key=lambda kv: kv[1])
        return color if n >= 0.6 * total else None

    def _is_referee_track(self, track_id, fallback_label):
        # A track is a referee only if referee *clearly dominates* its class
        # votes. Referees are rare, so a mostly-player track with a few stray
        # 'referee' mislabels stays a player (won't get the gold circle).
        votes = self._class_votes.get(track_id) if track_id is not None else None
        if not votes:
            return _is_referee(fallback_label)
        total = sum(votes.values())
        ref = sum(c for lbl, c in votes.items() if _is_referee(lbl))
        return total > 0 and ref >= 3 and ref >= 0.6 * total

    def _color_for(self, label, track_id):
        # Colour by the track's *accumulated* jersey team (robust to per-frame
        # noise). Referee/player only decides the fallback when the team is
        # undecided: a referee keeps gold (stays visible), a player isn't drawn.
        if _is_ball(label):
            return _BALL_RGBA
        if self.team_colors:
            team = self._team_color(track_id)
            if team is not None:
                return team
            return _REFEREE_RGBA if self._is_referee_track(track_id, label) else None
        return _REFEREE_RGBA if _is_referee(label) else _PLAYER_RGBA

    @staticmethod
    def _overlap(a, b):
        # max(IoU, intersection-over-smaller-area): catches both heavy overlap
        # and a small duplicate box sitting inside a larger one.
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        iou = inter / union if union > 0.0 else 0.0
        smaller = min(area_a, area_b)
        contain = inter / smaller if smaller > 0.0 else 0.0
        return max(iou, contain)

    @staticmethod
    def _feet_close(a, b):
        # True when two boxes' foot points (bottom-centre, where the ellipse is
        # drawn) are within ~0.4 of the smaller box width. The ellipse is ~2x the
        # box width, so near-coincident feet = one player circled twice even when
        # the boxes' IoU is low. Genuinely adjacent players are ~a full width
        # apart at the feet, so they're not merged.
        fax, fay = (a[0] + a[2]) / 2.0, a[3]
        fbx, fby = (b[0] + b[2]) / 2.0, b[3]
        ref = max(1.0, min(a[2] - a[0], b[2] - b[0]))
        return ((fax - fbx) ** 2 + (fay - fby) ** 2) ** 0.5 < 0.4 * ref

    def _merge_overlaps(self, entries):
        # Class-agnostic greedy suppression: keep the most confident box, drop
        # any later box that overlaps it past merge_iou OR sits at the same feet.
        # Collapses a player circled twice (e.g. player+goalkeeper on one person,
        # or two offset boxes) into one. The ball is never merged against players.
        if self.merge_iou <= 0.0 or len(entries) < 2:
            return entries
        ordered = sorted(entries, key=lambda e: e["confidence"], reverse=True)
        kept = []
        for e in ordered:
            if _is_ball(e["label"]):
                kept.append(e)
                continue
            if any(
                not _is_ball(k["label"])
                and (
                    self._overlap(e["box"], k["box"]) >= self.merge_iou
                    or self._feet_close(e["box"], k["box"])
                )
                for k in kept
            ):
                continue
            kept.append(e)
        return kept

    def _assign_track_ids(self, draw_entries, track_entries):
        # Give each drawn box a stable track id: track-mode boxes already carry
        # one; detection-mode boxes borrow the id of the best-overlapping track
        # (greedy, each track used once) so detection circles can use the
        # tracker's persistent id for the badge and the accumulated colour.
        ids = [e["track_id"] for e in draw_entries]
        if not track_entries:
            return ids
        pairs = []
        for di, e in enumerate(draw_entries):
            if e["track_id"] is not None or _is_ball(e["label"]):
                continue
            for t in track_entries:
                if _is_ball(t["label"]):
                    continue
                ov = self._overlap(e["box"], t["box"])
                if ov >= 0.3:
                    pairs.append((ov, di, t["track_id"]))
        pairs.sort(key=lambda p: p[0], reverse=True)
        used_draw, used_track = set(), set()
        for _ov, di, tid in pairs:
            if di in used_draw or tid in used_track:
                continue
            ids[di] = tid
            used_draw.add(di)
            used_track.add(tid)
        return ids

    def _smooth_boxes(self, np, entries):
        # Temporal EMA on the boxes we're about to draw. Each box is matched to
        # the nearest slot from last frame (by centre, within a size-relative
        # gate) and pulled toward the new detection; slots not matched this
        # frame are dropped (no phantoms). Damps jitter and interval steps. The
        # ball is passed through unsmoothed so it never lags.
        a = float(self.position_smoothing)
        if a <= 0.0 or not entries:
            return entries
        used = set()
        out = []
        for e in entries:
            if _is_ball(e["label"]):
                out.append(e)
                continue
            box = np.array(e["box"], dtype=np.float64)
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            # Generous gate so a coherent interval-step jump still associates
            # (and glides) without grabbing a different nearby player.
            gate = 1.5 * max(box[2] - box[0], box[3] - box[1], 1.0)
            best, best_d = None, gate
            for idx, slot in enumerate(self._smooth_slots):
                if idx in used:
                    continue
                sb = slot["box"]
                d = (
                    ((sb[0] + sb[2]) / 2.0 - cx) ** 2
                    + ((sb[1] + sb[3]) / 2.0 - cy) ** 2
                ) ** 0.5
                if d < best_d:
                    best, best_d = idx, d
            if best is None:
                self._smooth_slots.append({"box": box.copy()})
                used.add(len(self._smooth_slots) - 1)
                smoothed = box
            else:
                used.add(best)
                slot = self._smooth_slots[best]
                slot["box"] = a * slot["box"] + (1.0 - a) * box
                smoothed = slot["box"]
            ne = dict(e)
            ne["box"] = (
                float(smoothed[0]),
                float(smoothed[1]),
                float(smoothed[2]),
                float(smoothed[3]),
            )
            out.append(ne)
        self._smooth_slots = [s for i, s in enumerate(self._smooth_slots) if i in used]
        return out

    def _detection_color(self, cv2, np, frame, label, box):
        # Colour a raw detection box (no track id) by its jersey team, classified
        # from this frame -- referees included. When the jersey isn't clearly a
        # team colour, a referee falls back to gold (so real refs stay visible)
        # and a player isn't drawn (matching the track-mode behaviour).
        ref = _is_referee(label)
        if not self.team_colors:
            return _REFEREE_RGBA if ref else _PLAYER_RGBA
        vote = self._classify_jersey(cv2, np, frame, box)
        if vote == "red":
            return _RED_TEAM_RGBA
        if vote == "blue":
            return _BLUE_TEAM_RGBA
        if vote == "ref":
            return _REFEREE_RGBA
        return _REFEREE_RGBA if ref else None

    def _classify_jersey(self, cv2, np, frame, box):
        # Dominant jersey colour in the torso patch -> "red"/"blue"/"ref"/None
        # (HSV). "ref" is a distinctive non-team kit colour (yellow/orange or
        # pink/magenta) -- chosen to avoid grass-green and the red/blue teams --
        # so the referee is identified by its kit colour, not the class label.
        x1, y1, x2, y2 = (int(v) for v in box)
        h_box, w_box = y2 - y1, x2 - x1
        if h_box <= 0 or w_box <= 0:
            return None
        jy1, jy2 = y1 + int(0.15 * h_box), y1 + int(0.55 * h_box)
        jx1, jx2 = x1 + int(0.25 * w_box), x1 + int(0.75 * w_box)
        H, W = frame.shape[:2]
        jy1, jy2 = max(0, jy1), min(H, jy2)
        jx1, jx2 = max(0, jx1), min(W, jx2)
        if jy2 - jy1 < 3 or jx2 - jx1 < 3:
            return None
        # logical RGB from the buffer's channel order, then HSV
        rgb = np.ascontiguousarray(frame[jy1:jy2, jx1:jx2][:, :, self._inv_order[:3]])
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        s_v = (hsv[..., 1] > 80) & (hsv[..., 2] > 50)
        h = hsv[..., 0]
        red = int((((h <= 10) | (h >= 170)) & s_v).sum())
        blue = int(((h >= 100) & (h <= 130) & s_v).sum())
        # Referee kit: yellow/orange (~18-34) or pink/magenta (~145-165). These
        # bands skip grass-green (~40-90) and the red/blue team bands.
        ref = int(((((h >= 18) & (h <= 34)) | ((h >= 145) & (h <= 165))) & s_v).sum())
        min_pixels = max(20, int(0.02 * rgb.shape[0] * rgb.shape[1]))
        counts = {"red": red, "blue": blue, "ref": ref}
        best = max(counts, key=counts.get)
        if counts[best] < min_pixels:
            return None
        return best

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
        # Prefer the per-track smoothed width so the ellipse stays stable even
        # when a single detection box is momentarily oversized.
        smoothed = self._ell_w.get(track_id)
        width = max(1, int(smoothed if smoothed is not None else x2 - x1))
        color = self._c(rgba)
        cv2.ellipse(
            frame,
            (x_center, y_bottom),
            (width, max(1, int(0.35 * width))),
            0.0,
            -45,
            235,
            color,
            2,
            cv2.LINE_AA,
        )
        if self.show_ids and track_id is not None:
            rect_w, rect_h = 40, 18
            x1r = x_center - rect_w // 2
            x2r = x_center + rect_w // 2
            y1r = y_bottom - rect_h // 2 + 15
            y2r = y_bottom + rect_h // 2 + 15
            cv2.rectangle(frame, (x1r, y1r), (x2r, y2r), color, cv2.FILLED)
            tx = x1r + 12 - (10 if track_id > 99 else 0)
            cv2.putText(
                frame,
                str(track_id),
                (tx, y1r + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self._c(_BLACK_RGBA),
                2,
                cv2.LINE_AA,
            )

    def _draw_triangle(self, cv2, np, frame, box, rgba):
        x1, y1, x2, y2 = box
        x = int((x1 + x2) / 2)
        y = int(y1)
        pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]], dtype=np.int32)
        cv2.drawContours(frame, [pts], 0, self._c(rgba), cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, self._c(_BLACK_RGBA), 2)

    def _draw_focal_marker(self, cv2, np, frame, box):
        # Broadcast-style "selected player" chevron floating above the head,
        # plus a bolder ellipse, to flag the focal (HUD) player on the pitch.
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        tip_y = int(y1) - 10
        s = 16
        pts = np.array(
            [
                [cx, tip_y],
                [cx - s, tip_y - int(s * 1.5)],
                [cx + s, tip_y - int(s * 1.5)],
            ],
            dtype=np.int32,
        )
        cv2.drawContours(frame, [pts], 0, self._c(_HIGHLIGHT_RGBA), cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, self._c(_BLACK_RGBA), 2)
        # Bolder ring at the feet to reinforce the selection.
        x_center = int((x1 + x2) / 2)
        width = max(1, int(x2 - x1))
        cv2.ellipse(
            frame,
            (x_center, int(y2)),
            (width, max(1, int(0.35 * width))),
            0.0,
            -45,
            235,
            self._c(_HIGHLIGHT_RGBA),
            4,
            cv2.LINE_AA,
        )

    def _draw_label(self, cv2, frame, box, label, rgba):
        x1, y1, _, _ = box
        cv2.putText(
            frame,
            label,
            (int(x1), max(12, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self._c(rgba),
            1,
            cv2.LINE_AA,
        )

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
                frame[hy : hy + hh, hx : hx + hw] = headshot[:hh, :hw]
                cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), self._c(rgba), 2)
        tc = self._c(_HUD_TEXT_RGBA)
        cv2.putText(
            frame, self.player_label, (text_x, y + 28), font, 0.7, tc, 2, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Ball contacts: {contacts}",
            (text_x, y + 58),
            font,
            0.6,
            tc,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Distance: {distance_m:.1f} m",
            (text_x, y + 85),
            font,
            0.6,
            tc,
            1,
            cv2.LINE_AA,
        )

    def do_transform_ip(self, buf):
        try:
            import numpy as np

            all_entries = self._read_metadata(buf)
            # The buffer carries both the detector's boxes (track_id None) and
            # the tracker's boxes (track_id set). Tracking state/HUD always use
            # the tracked entries; what we *draw* depends on draw_from_detections.
            track_entries = [e for e in all_entries if e["track_id"] is not None]
            det_entries = [e for e in all_entries if e["track_id"] is None]

            # Ball position for contact counting: prefer a tracked ball, else
            # fall back to the strongest ball *detection* (the ball is small and
            # fast, so it often isn't tracked) -- so contacts still get counted.
            det_ball_box = None
            best_ball = -1.0
            for e in det_entries:
                if _is_ball(e["label"]) and e["confidence"] > best_ball:
                    best_ball, det_ball_box = e["confidence"], e["box"]

            # Per-track state (votes, contacts, distance, focal) from the tracker.
            active = self._update_tracks(
                track_entries if track_entries else all_entries, det_ball_box
            )

            if self.draw_from_detections:
                draw_entries = list(det_entries)
                # Bridge missed detections: the detector occasionally drops a
                # player for a frame, which would flicker the circle. The tracker
                # is still coasting that player (Kalman keep-alive), so draw any
                # confirmed track that has no detection this frame -- detections
                # still drive everything they cover; tracks only fill the gaps.
                if track_entries:
                    covered = set()
                    for d in det_entries:
                        if _is_ball(d["label"]):
                            continue
                        for t in track_entries:
                            if t["track_id"] in covered or _is_ball(t["label"]):
                                continue
                            if self._overlap(d["box"], t["box"]) >= 0.3:
                                covered.add(t["track_id"])
                    draw_entries += [
                        t
                        for t in track_entries
                        if not _is_ball(t["label"]) and t["track_id"] not in covered
                    ]
            else:
                draw_entries = track_entries if track_entries else det_entries
            # min-confidence gates only what we *draw* (tracks carry conf 1.0, so
            # they're unaffected); the contact math above used the raw detections.
            if self.min_confidence > 0.0:
                draw_entries = [
                    e for e in draw_entries if e["confidence"] >= self.min_confidence
                ]
            # Collapse overlapping boxes so one player isn't circled twice,
            # then low-pass the positions so the circle glides.
            draw_entries = self._merge_overlaps(draw_entries)
            draw_entries = self._smooth_boxes(np, draw_entries)
            if not all_entries:
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

                # Jersey team voting first, so trails/ellipses use this frame's
                # vote (track mode; detection mode classifies per box at draw).
                # Referees are voted on too -- their colour comes from the jersey
                # (gold only as the fallback), not the class label.
                if self.team_colors:
                    for e in track_entries:
                        tid = e["track_id"]
                        lab = self._stable_label(tid, e["label"])
                        if _is_ball(lab):
                            continue
                        vote = self._classify_jersey(cv2, np, frame, e["box"])
                        if vote:
                            tv = self._team_votes.setdefault(
                                tid, {"red": 0, "blue": 0, "ref": 0}
                            )
                            tv[vote] = tv.get(vote, 0) + 1

                if self.trails:
                    for tid in active:
                        rgba = self._color_for(self._track_label.get(tid, ""), tid)
                        if rgba is None:
                            continue
                        self._draw_trail(cv2, np, frame, self._trail.get(tid, []), rgba)

                # Which drawn box is the focal (HUD) player? Match the focal
                # track's box to the nearest drawn box so we can highlight it
                # even when drawing from detections (no track id on the box).
                focal_idx = None
                if self.highlight_focal:
                    focal_tid = self._focal_track()
                    focal_box = None
                    if focal_tid is not None:
                        for t in track_entries:
                            if t["track_id"] == focal_tid:
                                focal_box = t["box"]
                                break
                    if focal_box is not None:
                        best = 0.0
                        for i, e in enumerate(draw_entries):
                            if _is_ball(e["label"]):
                                continue
                            ov = self._overlap(e["box"], focal_box)
                            if ov > best:
                                best, focal_idx = ov, i

                # Stable track id per drawn box (detection boxes borrow the id of
                # the track they overlap) -- used for the id badge and to look up
                # the track's accumulated colour.
                draw_ids = self._assign_track_ids(draw_entries, track_entries)

                for i, e in enumerate(draw_entries):
                    box = e["box"]
                    badge_id = draw_ids[i]
                    # Use the track's stable identity (class + accumulated team
                    # votes) for colour whenever the box maps to a track -- in
                    # detection mode that's the box's matched track id. This
                    # makes colour robust to per-frame label/jersey noise. Only
                    # an unmatched detection falls back to this frame's guess.
                    color_tid = e["track_id"] if e["track_id"] is not None else badge_id
                    if color_tid is not None:
                        label = self._stable_label(color_tid, e["label"])
                    else:
                        label = e["label"]
                    if _is_ball(label):
                        if self.show_ball:
                            self._draw_triangle(cv2, np, frame, box, _BALL_RGBA)
                        continue
                    if color_tid is not None:
                        rgba = self._color_for(label, color_tid)
                    else:
                        rgba = self._detection_color(cv2, np, frame, label, box)
                    if rgba is None:
                        continue
                    self._draw_ellipse(cv2, frame, box, rgba, badge_id)
                    if i == focal_idx:
                        self._draw_focal_marker(cv2, np, frame, box)
                    if self.show_labels:
                        self._draw_label(cv2, frame, box, label, rgba)

                if self.show_hud:
                    focal = self._focal_track()
                    if focal is not None:
                        ppm = self._px_per_meter()
                        dist_m = (
                            (self._distance_px.get(focal, 0.0) / ppm) if ppm else 0.0
                        )
                        hud_rgba = (
                            self._color_for(self._track_label.get(focal, ""), focal)
                            or _DEFAULT_RGBA
                        )
                        self._draw_hud(
                            cv2,
                            frame,
                            self._contacts.get(focal, 0),
                            dist_m,
                            hud_rgba,
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
