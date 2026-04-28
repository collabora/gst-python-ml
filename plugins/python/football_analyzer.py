# FootballAnalyzer
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
import pickle

from log.global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstBase, GstVideo, GObject  # noqa: E402

    import cv2
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    from log.logger_factory import LoggerFactory  # noqa: E402

    VIDEO_CAPS = Gst.Caps.from_string("video/x-raw, format=BGR")

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_football_analyzer' element will not be available. Error: {e}"
    )


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]


class Tracker:
    """Tracker.
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def _foreground_mask(self, shape, frame_tracks, dilation=15):
        h, w = shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        bboxes = []
        for key in ("players", "referees", "ball"):
            for obj in frame_tracks.get(key, {}).values():
                bboxes.append(obj["bbox"])
        for bbox in bboxes:
            x1 = max(0, int(bbox[0]) - dilation)
            y1 = max(0, int(bbox[1]) - dilation)
            x2 = min(w, int(bbox[2]) + dilation)
            y2 = min(h, int(bbox[3]) + dilation)
            mask[y1:y2, x1:x2] = 0
        return mask

    def get_camera_motion(self, frames, tracks, read_from_stub=False, stub_path=None,
                          ratio=0.75, ransac_thresh=3.0, min_matches=8):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        cumulative = [np.eye(3, dtype=np.float64)]
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_mask = self._foreground_mask(frames[0].shape,
                                          {k: tracks[k][0] for k in tracks})
        prev_kp, prev_desc = self.sift.detectAndCompute(prev_gray, prev_mask)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            curr_mask = self._foreground_mask(frames[i].shape,
                                              {k: tracks[k][i] for k in tracks})
            curr_kp, curr_desc = self.sift.detectAndCompute(curr_gray, curr_mask)

            H_step = np.eye(3, dtype=np.float64)
            if prev_desc is not None and curr_desc is not None and len(prev_desc) >= 2 and len(curr_desc) >= 2:
                knn = self.matcher.knnMatch(prev_desc, curr_desc, k=2)
                good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
                if len(good) >= min_matches:
                    pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    pts_curr = np.float32([curr_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    H, _ = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, ransac_thresh)
                    if H is not None:
                        H_step = H

            cumulative.append(H_step @ cumulative[-1])
            prev_kp, prev_desc = curr_kp, curr_desc

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(cumulative, f)
        return cumulative

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i: i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        per_frame = []
        class_votes = {}
        for detection in detections:
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            tracked = self.tracker.update_with_detections(detection_supervision)
            per_frame.append((tracked, detection_supervision, cls_names, cls_names_inv))

            for fd in tracked:
                cls_name = cls_names[fd[3]]
                track_id = fd[4]
                if cls_name in ("player", "referee"):
                    v = class_votes.setdefault(track_id, {"player": 0, "referee": 0})
                    v[cls_name] += 1

        track_class = {
            tid: ("player" if v["player"] >= v["referee"] else "referee")
            for tid, v in class_votes.items()
        }

        for frame_num, (tracked, raw_detections, cls_names, cls_names_inv) in enumerate(per_frame):
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for fd in tracked:
                bbox = fd[0].tolist()
                track_id = fd[4]
                stable_cls = track_class.get(track_id)
                if stable_cls == "player":
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif stable_cls == "referee":
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for fd in raw_detections:
                if fd[3] == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": fd[0].tolist()}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(frame, f"{track_id}",
                        (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        return frame

    def classify_jersey(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_box, w_box = y2 - y1, x2 - x1
        if h_box <= 0 or w_box <= 0:
            return None
        jy1 = y1 + int(0.15 * h_box)
        jy2 = y1 + int(0.55 * h_box)
        jx1 = x1 + int(0.25 * w_box)
        jx2 = x1 + int(0.75 * w_box)
        H, W = frame.shape[:2]
        jy1, jy2 = max(0, jy1), min(H, jy2)
        jx1, jx2 = max(0, jx1), min(W, jx2)
        if jy2 - jy1 < 3 or jx2 - jx1 < 3:
            return None
        patch = frame[jy1:jy2, jx1:jx2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        s_v = (hsv[..., 1] > 80) & (hsv[..., 2] > 50)
        h = hsv[..., 0]
        red = (((h <= 10) | (h >= 170)) & s_v).sum()
        blue = ((h >= 100) & (h <= 130) & s_v).sum()
        min_pixels = max(20, int(0.02 * patch.shape[0] * patch.shape[1]))
        if red < min_pixels and blue < min_pixels:
            return None
        return "red" if red >= blue else "blue"

    def _ref_bottom_center(self, bbox, H_inv):
        xc, _ = get_center_of_bbox(bbox)
        yb = int(bbox[3])
        pt = cv2.perspectiveTransform(
            np.array([[[xc, yb]]], dtype=np.float32), H_inv
        )[0][0]
        return float(pt[0]), float(pt[1])

    def _minimap_extent(self, tracks, camera_motion):
        xs, ys = [], []
        n = len(tracks["players"])
        for i in range(n):
            H_inv = np.linalg.inv(camera_motion[i]) if camera_motion is not None else np.eye(3)
            for key in ("players", "referees"):
                for p in tracks[key][i].values():
                    x, y = self._ref_bottom_center(p["bbox"], H_inv)
                    xs.append(x); ys.append(y)
        if not xs:
            return None
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = 0.05 * max(1.0, max_x - min_x)
        pad_y = 0.05 * max(1.0, max_y - min_y)
        return min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y

    def _make_minimap_bg(self, mm_w, mm_h):
        bg = np.full((mm_h, mm_w, 3), (40, 110, 40), dtype=np.uint8)
        cv2.rectangle(bg, (2, 2), (mm_w - 3, mm_h - 3), (240, 240, 240), 2)
        cv2.line(bg, (mm_w // 2, 2), (mm_w // 2, mm_h - 3), (240, 240, 240), 1)
        cv2.circle(bg, (mm_w // 2, mm_h // 2), max(10, mm_h // 8), (240, 240, 240), 1)
        return bg

    def _project_to_minimap(self, extent, mm_w, mm_h, x, y):
        min_x, min_y, max_x, max_y = extent
        dx = max(1e-6, max_x - min_x)
        dy = max(1e-6, max_y - min_y)
        scale = min((mm_w - 10) / dx, (mm_h - 10) / dy)
        off_x = (mm_w - scale * dx) / 2.0
        off_y = (mm_h - scale * dy) / 2.0
        return int(off_x + (x - min_x) * scale), int(off_y + (y - min_y) * scale)

    def _smooth_points(self, pts, window):
        if window <= 1 or len(pts) < 2:
            return pts
        pts = np.asarray(pts, dtype=np.float32)
        n = len(pts)
        half = window // 2
        smoothed = np.empty_like(pts)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            smoothed[i] = pts[lo:hi].mean(axis=0)
        return smoothed

    def draw_trail(self, frame, points, color):
        if len(points) < 2:
            return frame
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        return frame

    def _point_to_bbox_distance(self, px, py, bbox):
        x1, y1, x2, y2 = bbox
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        return float(np.hypot(dx, dy))

    def _ball_contact(self, player_dict, ball_bbox, contact_pad_ratio):
        bx, by = get_center_of_bbox(ball_bbox)
        best_tid, best_d, best_bbox = None, float("inf"), None
        for tid, player in player_dict.items():
            d = self._point_to_bbox_distance(bx, by, player["bbox"])
            if d < best_d:
                best_tid, best_d, best_bbox = tid, d, player["bbox"]
        if best_bbox is None:
            return None
        w_box = best_bbox[2] - best_bbox[0]
        h_box = best_bbox[3] - best_bbox[1]
        if best_d > contact_pad_ratio * max(w_box, h_box):
            return None
        return best_tid

    def _count_total_contacts(self, tracks, contact_gap_frames, contact_pad_ratio):
        totals = {}
        last_contact_frame = {}
        for frame_num, (player_dict, ball_dict) in enumerate(zip(tracks["players"], tracks["ball"])):
            ball = ball_dict.get(1)
            if ball is None or not player_dict:
                continue
            tid = self._ball_contact(player_dict, ball["bbox"], contact_pad_ratio)
            if tid is None:
                continue
            last = last_contact_frame.get(tid)
            if last is None or (frame_num - last) > contact_gap_frames:
                totals[tid] = totals.get(tid, 0) + 1
            last_contact_frame[tid] = frame_num
        return totals

    def draw_player_hud(self, frame, player_id, contacts, distance_m, color, headshot=None):
        x, y = 10, 10
        bg_color = (131, 41, 92)
        text_color = (47, 186, 64)
        if headshot is not None:
            hh, hw = headshot.shape[:2]
            w, h = hw + 280, max(110, hh + 20)
            text_x = x + hw + 20
        else:
            w, h = 320, 100
            text_x = x + 12
        cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, cv2.FILLED)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if headshot is not None:
            hy, hx = y + 10, x + 10
            frame[hy:hy + headshot.shape[0], hx:hx + headshot.shape[1]] = headshot
            cv2.rectangle(frame, (hx, hy),
                          (hx + headshot.shape[1], hy + headshot.shape[0]), color, 2)
        cv2.putText(frame, "Player #8", (text_x, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, f"Ball contacts: {contacts}", (text_x, y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        cv2.putText(frame, f"Distance: {distance_m:.1f} m", (text_x, y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        return frame

    def draw_annotations(self, video_frames, tracks, camera_motion=None, trail_length=30,
                         contact_gap_frames=5, contact_pad_ratio=0.25,
                         player_height_m=1.8, headshot_path=None, headshot_size=90,
                         logo_path=None, logo_height=80, logo_margin=15,
                         trail_smooth_window=11,
                         show_minimap=True, minimap_size=(320, 200), minimap_margin=15):
        output_video_frames = []
        player_trails = {}
        team_votes = {}
        team_bgr = {"red": (0, 0, 255), "blue": (255, 0, 0)}
        default_color = (200, 200, 200)

        frames_count = {}
        for frame_players in tracks["players"]:
            for tid in frame_players:
                frames_count[tid] = frames_count.get(tid, 0) + 1
        total_contacts = self._count_total_contacts(tracks, contact_gap_frames, contact_pad_ratio)

        heights = [p["bbox"][3] - p["bbox"][1]
                   for frame_players in tracks["players"]
                   for p in frame_players.values()
                   if p["bbox"][3] > p["bbox"][1]]
        px_per_meter = float(np.median(heights)) / player_height_m if heights else 1.0

        headshot = None
        if headshot_path is not None and os.path.exists(headshot_path):
            img = cv2.imread(headshot_path)
            if img is not None:
                headshot = cv2.resize(img, (headshot_size, headshot_size), interpolation=cv2.INTER_AREA)

        logo_bgr, logo_alpha = None, None
        if logo_path is not None and os.path.exists(logo_path):
            img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                scale = logo_height / img.shape[0]
                new_w = max(1, int(round(img.shape[1] * scale)))
                img = cv2.resize(img, (new_w, logo_height), interpolation=cv2.INTER_LANCZOS4)
                if img.ndim == 3 and img.shape[2] == 4:
                    logo_bgr = img[..., :3]
                    logo_alpha = (img[..., 3:4].astype(np.float32)) / 255.0
                else:
                    logo_bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        minimap_bg, minimap_extent = None, None
        if show_minimap:
            minimap_extent = self._minimap_extent(tracks, camera_motion)
            if minimap_extent is not None:
                minimap_bg = self._make_minimap_bg(minimap_size[0], minimap_size[1])

        if total_contacts:
            focal_tid = max(total_contacts,
                            key=lambda t: (total_contacts[t], frames_count.get(t, 0)))
        elif frames_count:
            focal_tid = max(frames_count, key=frames_count.get)
        else:
            focal_tid = None

        last_ref_pt = {}
        player_distance = {}
        player_contacts = {}
        last_contact_frame = {}
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            H_cum = camera_motion[frame_num] if camera_motion is not None else np.eye(3)
            H_inv = np.linalg.inv(H_cum)

            active_ids = set(player_dict.keys())
            for track_id, player in player_dict.items():
                x_center, _ = get_center_of_bbox(player["bbox"])
                y_bottom = int(player["bbox"][3])
                ref_pt = cv2.perspectiveTransform(
                    np.array([[[x_center, y_bottom]]], dtype=np.float32), H_inv
                )[0][0]
                ref_tuple = (float(ref_pt[0]), float(ref_pt[1]))
                player_trails.setdefault(track_id, []).append(ref_tuple)
                if len(player_trails[track_id]) > trail_length:
                    player_trails[track_id] = player_trails[track_id][-trail_length:]

                if track_id in last_ref_pt:
                    dx = ref_tuple[0] - last_ref_pt[track_id][0]
                    dy = ref_tuple[1] - last_ref_pt[track_id][1]
                    player_distance[track_id] = player_distance.get(track_id, 0.0) + float(np.hypot(dx, dy))
                last_ref_pt[track_id] = ref_tuple

                vote = self.classify_jersey(frame, player["bbox"])
                if vote is not None:
                    counts = team_votes.setdefault(track_id, {"red": 0, "blue": 0})
                    counts[vote] += 1
            for track_id in list(player_trails.keys()):
                if track_id not in active_ids:
                    del player_trails[track_id]
                    last_ref_pt.pop(track_id, None)

            ball = ball_dict.get(1)
            if ball is not None and player_dict:
                tid = self._ball_contact(player_dict, ball["bbox"], contact_pad_ratio)
                if tid is not None:
                    last = last_contact_frame.get(tid)
                    if last is None or (frame_num - last) > contact_gap_frames:
                        player_contacts[tid] = player_contacts.get(tid, 0) + 1
                    last_contact_frame[tid] = frame_num

            focal_color = (131, 41, 92)

            def color_for(track_id):
                if track_id == focal_tid:
                    return focal_color
                counts = team_votes.get(track_id)
                if not counts or (counts["red"] == 0 and counts["blue"] == 0):
                    return default_color
                return team_bgr["red"] if counts["red"] >= counts["blue"] else team_bgr["blue"]

            for track_id, ref_points in player_trails.items():
                smoothed_ref = self._smooth_points(ref_points, trail_smooth_window)
                pts = cv2.perspectiveTransform(
                    np.asarray(smoothed_ref, dtype=np.float32).reshape(-1, 1, 2), H_cum
                ).reshape(-1, 2)
                frame = self.draw_trail(frame, pts.tolist(), color_for(track_id))

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], color_for(track_id))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            if focal_tid is not None:
                frame = self.draw_player_hud(
                    frame,
                    focal_tid,
                    player_contacts.get(focal_tid, 0),
                    player_distance.get(focal_tid, 0.0) / px_per_meter,
                    color_for(focal_tid),
                    headshot=headshot,
                )

            if logo_bgr is not None:
                lh, lw = logo_bgr.shape[:2]
                fh, fw = frame.shape[:2]
                x0 = max(0, fw - lw - logo_margin)
                y0 = logo_margin
                x1, y1 = x0 + lw, y0 + lh
                if logo_alpha is not None:
                    roi = frame[y0:y1, x0:x1].astype(np.float32)
                    blended = roi * (1.0 - logo_alpha) + logo_bgr.astype(np.float32) * logo_alpha
                    frame[y0:y1, x0:x1] = blended.astype(np.uint8)
                else:
                    frame[y0:y1, x0:x1] = logo_bgr

            if minimap_bg is not None and minimap_extent is not None:
                mm = minimap_bg.copy()
                mm_w, mm_h = minimap_size
                for tid, player in player_dict.items():
                    rx, ry = self._ref_bottom_center(player["bbox"], H_inv)
                    mx, my = self._project_to_minimap(minimap_extent, mm_w, mm_h, rx, ry)
                    dot_color = color_for(tid)
                    radius = 6 if tid == focal_tid else 4
                    cv2.circle(mm, (mx, my), radius, dot_color, cv2.FILLED)
                    cv2.circle(mm, (mx, my), radius, (0, 0, 0), 1)
                for referee in referee_dict.values():
                    rx, ry = self._ref_bottom_center(referee["bbox"], H_inv)
                    mx, my = self._project_to_minimap(minimap_extent, mm_w, mm_h, rx, ry)
                    cv2.circle(mm, (mx, my), 3, (0, 255, 255), cv2.FILLED)
                    cv2.circle(mm, (mx, my), 3, (0, 0, 0), 1)
                ball = ball_dict.get(1)
                if ball is not None:
                    bx, by = get_center_of_bbox(ball["bbox"])
                    bref = cv2.perspectiveTransform(
                        np.array([[[bx, by]]], dtype=np.float32), H_inv
                    )[0][0]
                    mx, my = self._project_to_minimap(minimap_extent, mm_w, mm_h,
                                                      float(bref[0]), float(bref[1]))
                    cv2.circle(mm, (mx, my), 4, (0, 255, 0), cv2.FILLED)
                    cv2.circle(mm, (mx, my), 4, (0, 0, 0), 1)
                fh, fw = frame.shape[:2]
                x0 = max(0, fw - mm_w - minimap_margin)
                y0 = max(0, fh - mm_h - minimap_margin)
                frame[y0:y0 + mm_h, x0:x0 + mm_w] = mm

            output_video_frames.append(frame)

        return output_video_frames


class FootballAnalyzer(GstBase.BaseTransform):
    """
    Buffers every incoming video frame, then on EOS runs the full batch
    pipeline (YOLO detection, ByteTrack with whole-clip class voting,
    SIFT/RANSAC camera motion, annotated drawing with trails / HUD /
    logo / minimap) and pushes the annotated frames downstream before
    forwarding EOS.
    """

    __gstmetadata__ = (
        "Football Analyzer",
        "Filter/Effect/Video",
        "Runs football_analysis (YOLO + ByteTrack + SIFT camera motion + "
        "annotated drawing) on the full clip and emits annotated frames on EOS",
        "Marcus Edel <marcus@urgs.org>",
    )

    src_template = Gst.PadTemplate.new(
        "src",
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        VIDEO_CAPS.copy(),
    )
    sink_template = Gst.PadTemplate.new(
        "sink",
        Gst.PadDirection.SINK,
        Gst.PadPresence.ALWAYS,
        VIDEO_CAPS.copy(),
    )
    __gsttemplates__ = (src_template, sink_template)

    model_path = GObject.Property(
        type=str,
        default="",
        nick="Model Path",
        blurb="Path to the YOLO weights (must be set before processing)",
        flags=GObject.ParamFlags.READWRITE,
    )

    headshot_path = GObject.Property(
        type=str,
        default="",
        nick="Headshot Path",
        blurb="Optional headshot image for the focal-player HUD",
        flags=GObject.ParamFlags.READWRITE,
    )

    logo_path = GObject.Property(
        type=str,
        default="",
        nick="Logo Path",
        blurb="Optional top-right logo overlay",
        flags=GObject.ParamFlags.READWRITE,
    )

    tracks_stub_path = GObject.Property(
        type=str,
        default="",
        nick="Tracks Stub Path",
        blurb="Optional pickle path for cached object tracks (read & written)",
        flags=GObject.ParamFlags.READWRITE,
    )

    camera_motion_stub_path = GObject.Property(
        type=str,
        default="",
        nick="Camera Motion Stub Path",
        blurb="Optional pickle path for cached camera-motion homographies (read & written)",
        flags=GObject.ParamFlags.READWRITE,
    )

    show_minimap = GObject.Property(
        type=bool,
        default=True,
        nick="Show Minimap",
        blurb="Render the bottom-right minimap overlay",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self._frames = []
        self._pts = []
        self._duration = []
        self._width = 0
        self._height = 0
        self._tracker = None

    def _ensure_tracker(self):
        if self._tracker is not None:
            return self._tracker
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO model not found: {self.model_path!r}")
        self.logger.info(f"Loading FootballAnalyzer Tracker from {self.model_path}")
        self._tracker = Tracker(self.model_path)
        return self._tracker

    def do_set_caps(self, incaps, outcaps):
        info = GstVideo.VideoInfo.new_from_caps(incaps)
        self._width = info.width
        self._height = info.height
        return True

    def do_transform_ip(self, buf):
        try:
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                self.logger.error("Failed to map incoming buffer for read")
                return Gst.FlowReturn.ERROR
            try:
                frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(
                    self._height, self._width, 3
                ).copy()
            finally:
                buf.unmap(mapinfo)

            self._frames.append(frame)
            self._pts.append(buf.pts)
            self._duration.append(buf.duration)
            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"FootballAnalyzer chain error: {e}")
            return Gst.FlowReturn.ERROR

    def do_sink_event(self, event):
        if event.type == Gst.EventType.EOS:
            try:
                self._run_pipeline_and_push()
            except Exception as e:
                self.logger.error(f"FootballAnalyzer EOS processing failed: {e}")
                # Forward EOS regardless so the pipeline shuts down cleanly.
        return GstBase.BaseTransform.do_sink_event(self, event)

    def _run_pipeline_and_push(self):
        if not self._frames:
            self.logger.info("FootballAnalyzer: no frames buffered, skipping")
            return

        tracker = self._ensure_tracker()
        n = len(self._frames)
        self.logger.info(f"FootballAnalyzer: running pipeline on {n} frames")

        tracks_stub = self.tracks_stub_path or None
        cam_stub = self.camera_motion_stub_path or None
        headshot = self.headshot_path or None
        logo = self.logo_path or None

        tracks = tracker.get_object_tracks(
            self._frames,
            read_from_stub=tracks_stub is not None and os.path.exists(tracks_stub),
            stub_path=tracks_stub,
        )
        camera_motion = tracker.get_camera_motion(
            self._frames,
            tracks,
            read_from_stub=cam_stub is not None and os.path.exists(cam_stub),
            stub_path=cam_stub,
        )
        annotated = tracker.draw_annotations(
            self._frames,
            tracks,
            camera_motion=camera_motion,
            headshot_path=headshot,
            logo_path=logo,
            show_minimap=self.show_minimap,
        )

        if len(annotated) != n:
            self.logger.warning(
                f"draw_annotations returned {len(annotated)} frames for {n} inputs; "
                "padding/truncating to match"
            )
            if len(annotated) < n:
                annotated = list(annotated) + [annotated[-1]] * (n - len(annotated))
            else:
                annotated = annotated[:n]

        srcpad = self.srcpad
        for i, out in enumerate(annotated):
            data = np.ascontiguousarray(out, dtype=np.uint8).tobytes()
            outbuf = Gst.Buffer.new_allocate(None, len(data), None)
            outbuf.fill(0, data)
            outbuf.pts = self._pts[i]
            outbuf.duration = self._duration[i]
            ret = srcpad.push(outbuf)
            if ret != Gst.FlowReturn.OK:
                self.logger.error(
                    f"Pushing annotated frame {i} failed with {ret}; aborting"
                )
                break

        self._frames.clear()
        self._pts.clear()
        self._duration.clear()


if CAN_REGISTER_ELEMENT:
    GObject.type_register(FootballAnalyzer)
    __gstelementfactory__ = (
        "pyml_football_analyzer",
        Gst.Rank.NONE,
        FootballAnalyzer,
    )
else:
    GlobalLogger().warning(
        "The 'pyml_football_analyzer' element will not be registered because "
        "required modules are missing."
    )
