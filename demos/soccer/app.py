import argparse
import time
import cv2
import numpy as np
import os
from collections import defaultdict, deque
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None

BOT_OK = BYTE_OK = True
try:
    from ultralytics.trackers.bot_sort import BOTSORT
except Exception:
    BOT_OK = False
try:
    from ultralytics.trackers.byte_tracker import BYTETracker
except Exception:
    BYTE_OK = False

CFG_OK = True
try:
    from ultralytics.cfg import get_cfg
except Exception:
    CFG_OK = False

from ultralytics.engine.results import Boxes


def parse_args():
    p = argparse.ArgumentParser(description="")

    p.add_argument("--source", default="0", help="0=webcam or path/rtsp/video")
    p.add_argument("--save", default="ibc-out-dual.mp4", help="Output video path")
    p.add_argument("--model", default="yolo11x.pt", help="YOLO .pt (e.g., yolo11x.pt)")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda:0|mps")

    p.add_argument("--imgsz", type=int, default=1280, help="Detector input size")
    p.add_argument(
        "--conf", type=float, default=0.25, help="Global detector confidence (pre-NMS)"
    )
    p.add_argument("--iou", type=float, default=0.45, help="Detector NMS IoU")
    p.add_argument(
        "--people-and-ball",
        action="store_true",
        help="Restrict model to person(0)+ball(32)",
    )
    p.add_argument(
        "--classes", type=int, nargs="*", default=None, help="Override classes"
    )

    p.add_argument(
        "--person-conf-keep",
        type=float,
        default=0.25,
        help="Min conf to KEEP person boxes",
    )
    p.add_argument(
        "--ball-conf-keep", type=float, default=0.04, help="Min conf to KEEP ball boxes"
    )

    p.add_argument(
        "--ball-mode", action="store_true", help="Enable ball fallback logic"
    )
    p.add_argument(
        "--hires-fallback",
        action="store_true",
        help="Second pass if ball missing (bounded)",
    )
    p.add_argument(
        "--hires-imgsz",
        type=int,
        default=1536,
        help="imgsz for fallback (may clamp on CPU/auto)",
    )
    p.add_argument(
        "--fallback-every",
        type=int,
        default=6,
        help="Try fallback every N frames when ball missing",
    )
    p.add_argument("--fallback-tiles", action="store_true", help="Use tiled fallback")
    p.add_argument("--tile-size", type=int, default=1280, help="Tile size for fallback")
    p.add_argument(
        "--tile-overlap", type=int, default=256, help="Tile overlap for fallback"
    )
    p.add_argument(
        "--fallback-budget-ms",
        type=int,
        default=300,
        help="Max ms inside fallback per frame",
    )

    p.add_argument(
        "--ball-roi-boost",
        action="store_true",
        help="On ball miss, try a hi-res predict on a region around the last ball box first",
    )
    p.add_argument(
        "--roi-scale",
        type=float,
        default=2.5,
        help="Scale factor to grow last ball box for ROI",
    )
    p.add_argument("--roi-min", type=int, default=256, help="Min ROI side length (px)")
    p.add_argument("--roi-max", type=int, default=1920, help="Max ROI side length (px)")

    p.add_argument(
        "--tracker-people",
        default="botsort_custom.yaml",
        help="BoT-SORT YAML for persons",
    )
    p.add_argument(
        "--tracker-ball", default="bytetrack_ball.yaml", help="ByteTrack YAML for ball"
    )

    p.add_argument(
        "--people-reid",
        dest="people_reid",
        action="store_true",
        default=True,
        help="Enable ReID for people tracker (safe shim)",
    )
    p.add_argument(
        "--no-people-reid",
        dest="people_reid",
        action="store_false",
        help="Disable ReID for people tracker",
    )

    p.add_argument(
        "--trail", type=int, default=200, help="Points kept in the unified ball trail"
    )
    p.add_argument(
        "--gmc",
        choices=["off", "affine", "homography"],
        default="affine",
        help="Stabilize trails",
    )
    p.add_argument(
        "--gmc-scale", type=float, default=0.5, help="Resize factor for GMC frame"
    )
    p.add_argument("--gft-max-corners", type=int, default=400)
    p.add_argument("--gft-quality", type=float, default=0.01)
    p.add_argument("--gft-min-dist", type=int, default=8)
    p.add_argument("--lk-win", type=int, default=21)
    p.add_argument("--lk-levels", type=int, default=3)
    p.add_argument("--ransac-thresh", type=float, default=3.0)

    p.add_argument(
        "--ball-gate-rel",
        type=float,
        default=0.06,
        help="Gate radius as a fraction of min(H,W) vs last trail point",
    )
    p.add_argument(
        "--ball-gate-min", type=int, default=12, help="Minimum gate radius in pixels"
    )
    p.add_argument(
        "--ball-gate-use-pred",
        action="store_true",
        help="Also accept if within gate of the predicted position (temporal model)",
    )
    p.add_argument(
        "--ball-min-iou",
        type=float,
        default=0.20,
        help="Minimum IoU with last SHOWN bbox to accept a candidate (tracks only)",
    )
    p.add_argument(
        "--ball-max-jump-rel",
        type=float,
        default=0.12,
        help="Absolute max jump (fraction of min(H,W)); anything larger is rejected",
    )
    p.add_argument(
        "--ball-speed-mult",
        type=float,
        default=3.0,
        help="Reject if distance > speed_mult * recent_speed (speed gate, tracks only)",
    )
    p.add_argument(
        "--ball-smooth-ema",
        type=float,
        default=0.0,
        help="EMA smoothing factor (0..1) for accepted center; 0 disables",
    )

    p.add_argument(
        "--det-override-conf",
        type=float,
        default=0.28,
        help="If track gate fails but a detection has >= this conf, accept anyway (with basic sanity checks)",
    )
    p.add_argument(
        "--det-override-after",
        type=int,
        default=2,
        help="Require this many consecutive reject frames with a detection before forcing override",
    )
    p.add_argument(
        "--reacquire-frames",
        type=int,
        default=6,
        help="If last trail point is older than this many frames, allow high-conf detection to reacquire",
    )

    p.add_argument(
        "--ball-coast",
        action="store_true",
        help="If no acceptable candidate this frame, extend trail using predicted position",
    )
    p.add_argument(
        "--coast-max", type=int, default=6, help="Max consecutive coasting frames"
    )
    p.add_argument(
        "--coast-decay",
        type=float,
        default=0.90,
        help="Velocity decay used during coasting predictions (0..1, higher = longer glide)",
    )

    p.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Draw a tiny corner banner to confirm overlays",
    )
    p.add_argument(
        "--draw-dets-too",
        action="store_true",
        help="Also draw raw detector boxes each frame (helps verify visibility)",
    )
    p.add_argument(
        "--draw-det-persons",
        action="store_true",
        help="With --draw-dets-too, also draw raw PERSON dets",
    )
    p.add_argument(
        "--draw-det-person",
        dest="draw_det_persons",
        action="store_true",
        help="Alias for --draw-det-persons",
    )
    p.add_argument(
        "--draw-ball-id-trails",
        action="store_true",
        help="Also draw per-ID ball trails (unified trail is always drawn)",
    )
    p.add_argument("--draw-people", action="store_true", help="Draw person boxes/IDs")
    p.add_argument(
        "--draw-people-trails", action="store_true", help="Draw person trails"
    )

    p.add_argument(
        "--fourcc", default="mp4v", help="Preferred FourCC; fallback to XVID/MJPG"
    )
    p.add_argument(
        "--force-fps", type=float, default=0.0, help="Writer FPS if input FPS missing"
    )
    p.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames (0 = all)"
    )
    p.add_argument(
        "--progress-every", type=int, default=60, help="Progress print every N frames"
    )
    p.add_argument("--print-homography", action="store_true", help="Print H each frame")
    p.add_argument("--verbose", action="store_true", help="Verbose per-frame logs")

    return p.parse_args()


def log(msg, force=False, verbose=False):
    if force or verbose:
        print(msg, flush=True)


def eye3():
    return np.eye(3, dtype=np.float32)


def estimate_global_motion(prev_gray, gray, args, frame_idx, verbose=False):
    if args.gmc == "off":
        log(f"[frame {frame_idx}] GMC OFF → I", verbose=verbose)
        return eye3()

    def down(img):
        if args.gmc_scale == 1.0:
            return img
        w = max(2, int(img.shape[1] * args.gmc_scale))
        h = max(2, int(img.shape[0] * args.gmc_scale))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    pg, cg = down(prev_gray), down(gray)
    pts_prev = cv2.goodFeaturesToTrack(
        pg,
        maxCorners=args.gft_max_corners,
        qualityLevel=args.gft_quality,
        minDistance=args.gft_min_dist,
    )
    if pts_prev is None or len(pts_prev) < 6:
        log(f"[frame {frame_idx}] GMC: insufficient corners → I", verbose=verbose)
        return eye3()

    pts_curr, st, _ = cv2.calcOpticalFlowPyrLK(
        pg,
        cg,
        pts_prev,
        None,
        winSize=(args.lk_win, args.lk_win),
        maxLevel=args.lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if pts_curr is None or st is None:
        log(f"[frame {frame_idx}] GMC: LK failed → I", verbose=verbose)
        return eye3()

    m = st.reshape(-1).astype(bool)
    if m.sum() < (4 if args.gmc == "homography" else 3):
        log(f"[frame {frame_idx}] GMC: not enough inliers → I", verbose=verbose)
        return eye3()

    src = pts_prev[m]
    dst = pts_curr[m]
    if args.gmc_scale != 1.0:
        s = 1.0 / args.gmc_scale
        src *= s
        dst *= s

    if args.gmc == "homography":
        H, _ = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            ransacReprojThreshold=args.ransac_thresh,
            maxIters=1000,
        )
        H = H.astype(np.float32) if H is not None else eye3()
    else:
        A, _ = cv2.estimateAffine2D(
            src, dst, ransacReprojThreshold=args.ransac_thresh, maxIters=1000
        )
        H = np.vstack([A, [0, 0, 1]]).astype(np.float32) if A is not None else eye3()

    avg = float(np.mean(np.linalg.norm(dst - src, axis=1))) if len(src) > 0 else 0.0
    log(
        f"[frame {frame_idx}] GMC {args.gmc} inliers={int(m.sum())} avg_motion={avg:.2f}px",
        verbose=verbose,
    )
    return H


def warp_points(points_xy, M):
    if not points_xy:
        return []
    P = np.c_[
        np.array(points_xy, dtype=np.float32), np.ones((len(points_xy), 1), np.float32)
    ]
    Q = (M @ P.T).T
    Q = Q[:, :2] / np.clip(Q[:, 2:3], 1e-6, None)
    return [tuple(q) for q in Q]


def try_open_writer(path, w, h, fps, preferred_fourcc, verbose=False):
    exts = os.path.splitext(path)[1].lower()
    candidates = [preferred_fourcc.upper()]
    for alt in ["mp4v", "XVID", "MJPG"]:
        if alt.upper() not in candidates:
            candidates.append(alt.upper())
    for cc in candidates:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if writer.isOpened():
            if cc in ("XVID", "MJPG") and exts != ".avi":
                log(
                    f"[warn] FOURCC {cc} often prefers .avi; you used '{exts}'.",
                    force=True,
                )
            log(
                f"[init] VideoWriter opened {w}x{h} @ {fps} FOURCC={cc} -> {path}",
                force=True,
            )
            return writer, cc
        else:
            log(
                f"[init] Failed to open VideoWriter with FOURCC={cc}, trying next...",
                force=True,
            )
    raise RuntimeError("All FOURCC options failed (mp4v/XVID/MJPG).")


def classwise_keep(result, person_thr, ball_thr):
    if (
        result is None
        or result.boxes is None
        or len(result.boxes) == 0
        or torch is None
    ):
        return np.zeros((0, 6), np.float32), np.zeros((0, 6), np.float32)

    b = result.boxes
    xyxy = b.xyxy.cpu().numpy()
    conf = (
        b.conf.cpu().numpy() if b.conf is not None else np.ones((len(b),), np.float32)
    )
    cls = (
        b.cls.cpu().numpy().astype(int)
        if b.cls is not None
        else np.zeros((len(b),), np.int32)
    )

    keep_p = (cls == 0) & (conf >= person_thr)
    keep_b = (cls == 32) & (conf >= ball_thr)

    dets_p = np.c_[xyxy[keep_p], conf[keep_p], cls[keep_p]].astype(np.float32)
    dets_b = np.c_[xyxy[keep_b], conf[keep_b], cls[keep_b]].astype(np.float32)
    return dets_p, dets_b


def dets_to_boxes(dets_xyxy_conf_cls, frame_shape):
    if dets_xyxy_conf_cls is None or dets_xyxy_conf_cls.size == 0:
        import torch as _torch

        data = _torch.zeros((0, 6), dtype=_torch.float32)
        return Boxes(data, frame_shape)
    import torch as _torch

    data = _torch.from_numpy(dets_xyxy_conf_cls).to(_torch.float32)
    return Boxes(data, frame_shape)


def clamp_imgsz_for_device(imgsz, device_str):
    if device_str in ("cpu", "auto"):
        return min(imgsz, 1280)
    return imgsz


def expand_roi(xyxy, scale, W, H, min_side=256, max_side=1920):
    x1, y1, x2, y2 = map(float, xyxy)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1), (y2 - y1)
    side = max(w, h) * float(scale)
    side = max(min_side, min(side, max_side))
    x1n = max(0, int(cx - side * 0.5))
    y1n = max(0, int(cy - side * 0.5))
    x2n = min(W - 1, int(cx + side * 0.5))
    y2n = min(H - 1, int(cy + side * 0.5))
    return x1n, y1n, x2n, y2n


def _normalize_tracker_args(args, kind="byte"):
    def has(a):
        return hasattr(args, a) and getattr(args, a) is not None

    def setif(a, v):
        setattr(args, a, v)

    def copy_if_missing(target, *sources, default=None):
        if not has(target):
            for s in sources:
                if has(s):
                    setif(target, getattr(args, s))
                    return
            if default is not None:
                setif(target, default)

    copy_if_missing("track_high_thresh", "track_thresh", default=0.5)
    copy_if_missing("track_thresh", "track_high_thresh", default=0.5)
    copy_if_missing("track_low_thresh", default=0.1)
    copy_if_missing("new_track_thresh", default=0.4)
    copy_if_missing("match_thresh", default=0.8)
    copy_if_missing("asso_thresh", "match_thresh", default=0.8)
    copy_if_missing("track_buffer", default=30)
    copy_if_missing("frame_rate", default=30)
    copy_if_missing("fuse_score", default=True)
    copy_if_missing("fuse_score_coef", default=1.0)
    copy_if_missing("mot20", default=False)

    if kind == "botsort":
        copy_if_missing("with_reid", default=True)
        copy_if_missing("proximity_thresh", default=0.5)
        copy_if_missing("appearance_thresh", default=0.25)
        if has("cmc_method") and not has("gmc_method"):
            setif("gmc_method", getattr(args, "cmc_method"))
        copy_if_missing("gmc_method", default="sparseOptFlow")


class ByteTrackWrapper:
    """Ultralytics BYTETracker; load params from YAML using get_cfg, accept Boxes."""

    def __init__(self, yaml_path, frame_rate):
        if not BYTE_OK:
            raise RuntimeError("Ultralytics BYTETracker not available")
        if not CFG_OK:
            raise RuntimeError(
                "Ultralytics get_cfg not available; update ultralytics package."
            )
        args = get_cfg(yaml_path)
        args.frame_rate = int(frame_rate)
        _normalize_tracker_args(args, kind="byte")
        self.tracker = BYTETracker(args, frame_rate=int(frame_rate))

    def update(self, boxes: Boxes, frame):
        return self.tracker.update(boxes, frame)


class BoTSORTWrapper:
    """Ultralytics BOTSORT; load params via get_cfg, accept Boxes, safe ReID encoder."""

    def __init__(self, yaml_path, frame_rate, enable_reid=True):
        if not BOT_OK:
            raise RuntimeError("Ultralytics BOTSORT not available")
        if not CFG_OK:
            raise RuntimeError(
                "Ultralytics get_cfg not available; update ultralytics package."
            )
        args = get_cfg(yaml_path)
        args.frame_rate = int(frame_rate)
        args.with_reid = bool(enable_reid)
        _normalize_tracker_args(args, kind="botsort")
        self.tracker = BOTSORT(args, frame_rate=int(frame_rate))
        self._install_safe_encoder()

    def _install_safe_encoder(self):
        import torch as _torch

        def _safe_hsv_hist(img_bgr, bboxes):
            feats = []
            if img_bgr is None or bboxes is None:
                return feats

            if hasattr(bboxes, "detach"):
                bb = bboxes.detach().cpu().numpy()
            elif isinstance(bboxes, np.ndarray):
                bb = bboxes
            else:
                try:
                    bb = np.asarray(bboxes, dtype=np.float32)
                except Exception:
                    bb = None

            H, W = img_bgr.shape[:2]

            def process_one(b):
                b = np.asarray(b, dtype=np.float32).reshape(-1)
                x1, y1, x2, y2 = map(float, b[:4])
                x1i = max(0, min(int(x1), W - 1))
                y1i = max(0, min(int(y1), H - 1))
                x2i = max(0, min(int(x2), W - 1))
                y2i = max(0, min(int(y2), H - 1))
                if x2i <= x1i or y2i <= y1i:
                    return _torch.zeros(512, dtype=_torch.float32)
                crop = img_bgr[y1i:y2i, x1i:x2i]
                if crop.size == 0:
                    return _torch.zeros(512, dtype=_torch.float32)
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist(
                    [hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]
                ).flatten()
                norm = np.linalg.norm(hist) + 1e-6
                hist = (hist / norm).astype(np.float32)
                return _torch.from_numpy(hist)

            if bb is not None:
                for b in bb:
                    feats.append(process_one(b))
            else:
                for b in bboxes:
                    feats.append(process_one(b))
            return feats

        self.tracker.encoder = lambda img, tlbrs: _safe_hsv_hist(img, tlbrs)

    def update(self, boxes: Boxes, frame):
        return self.tracker.update(boxes, frame)


def _callable_or_attr(obj, name):
    v = getattr(obj, name, None)
    if v is None:
        return None
    return v() if callable(v) else v


def tlbr_of(tr):
    a = _callable_or_attr(tr, "tlbr")
    if a is not None:
        a = a.tolist() if hasattr(a, "tolist") else a
        if len(a) == 4:
            return a
    a = _callable_or_attr(tr, "tlwh")
    if a is not None:
        a = a.tolist() if hasattr(a, "tolist") else a
        if len(a) == 4:
            x, y, w, h = a
            return [x, y, x + w, y + h]
    if hasattr(tr, "bbox"):
        b = tr.bbox
        return b.tolist() if hasattr(b, "tolist") else list(b)
    return None


class BallState:
    def __init__(self):
        self.cx = None
        self.cy = None
        self.vx = 0.0
        self.vy = 0.0
        self.frame = -1

    def predict(self, frame_idx, decay=0.85):
        if self.cx is None:
            return None
        return (self.cx + decay * self.vx, self.cy + decay * self.vy)

    def update_from_xyxy(self, xyxy, frame_idx):
        x1, y1, x2, y2 = map(float, xyxy)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if self.cx is not None and self.frame >= 0:
            dt = max(1, frame_idx - self.frame)
            self.vx = (cx - self.cx) / dt
            self.vy = (cy - self.cy) / dt
        self.cx, self.cy, self.frame = cx, cy, frame_idx

    def update_from_center(self, cx, cy, frame_idx):
        if self.cx is not None and self.frame >= 0:
            dt = max(1, frame_idx - self.frame)
            self.vx = (cx - self.cx) / dt
            self.vy = (cy - self.cy) / dt
        self.cx, self.cy, self.frame = float(cx), float(cy), int(frame_idx)


def _aspect_round_penalty(w, h):
    ar = w / max(h, 1e-6)
    roundness = np.exp(-((ar - 1.0) ** 2) / 0.15)
    return 1.0 - float(roundness)


def _size_penalty(w, h, H, W):
    s = max(w, h)
    tgt = 0.03 * min(H, W)
    return float(np.clip(abs(s - tgt) / (tgt + 1e-6), 0.0, 2.0)) * 0.5


def select_best_ball(
    dets_b,
    frame_shape,
    ball_state,
    frame_idx,
    w_conf=1.0,
    w_dist=0.015,
    w_size=0.5,
    w_round=0.4,
):
    if dets_b is None or len(dets_b) == 0:
        return None
    H, W = frame_shape
    pred = ball_state.predict(frame_idx)
    scores = []
    for d in dets_b:
        x1, y1, x2, y2, conf, _ = d
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        w, h = (x2 - x1), (y2 - y1)
        s_conf = float(conf)
        if pred is None:
            dist_pen = 0.0
        else:
            px, py = pred
            dist = np.hypot(cx - px, cy - py)
            dist_pen = float(dist / (0.5 * (H + W)))
        size_pen = _size_penalty(w, h, H, W)
        round_pen = _aspect_round_penalty(w, h)
        s = (
            w_conf * s_conf
            - w_dist * dist_pen
            - w_size * size_pen
            - w_round * round_pen
        )
        scores.append(s)
    if not scores:
        return None
    return int(np.argmax(scores))


def nms_class(dets, iou_thr=0.5):
    if dets is None or len(dets) == 0:
        return dets
    boxes = dets[:, :4].copy()
    scores = dets[:, 4].copy()
    order = scores.argsort()[::-1]
    keep = []

    def iou(a, b):
        xx1 = np.maximum(a[0], b[0])
        yy1 = np.maximum(a[1], b[1])
        xx2 = np.minimum(a[2], b[2])
        yy2 = np.minimum(a[3], b[3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = np.array([iou(boxes[i], boxes[j]) for j in order[1:]])
        remain = np.where(ious <= iou_thr)[0]
        order = order[remain + 1]
    return dets[keep]


def add_trail_point(seq: deque, x: int, y: int, k: int, densify=True, max_gap=5):
    if len(seq) > 0 and densify:
        _, _, k_prev = seq[-1]
        gap = k - k_prev
        if 1 < gap <= max_gap:
            x_prev, y_prev, _ = seq[-1]
            for t in range(1, gap):
                alpha = t / gap
                xi = int(round((1 - alpha) * x_prev + alpha * x))
                yi = int(round((1 - alpha) * y_prev + alpha * y))
                seq.append((xi, yi, k_prev + t))
    seq.append((int(x), int(y), int(k)))


def iou_xyxy(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter + 1e-6
    return float(inter / denom)


def lerp(a, b, t):
    return a * (1.0 - t) + b * t


def gate_accept(
    center,
    cand_box,
    trail_seq,
    last_shown_box,
    recent_speed,
    frame_shape,
    args,
    pred=None,
    from_track=True,
):
    if center is None:
        return False

    H, W = frame_shape
    hard_cap = float(args.ball_max_jump_rel) * float(min(H, W))

    if len(trail_seq) == 0:
        dist_prev = 0.0
    else:
        x_prev, y_prev, _ = trail_seq[-1]
        dist_prev = float(np.hypot(center[0] - x_prev, center[1] - y_prev))

    if dist_prev > hard_cap:
        return False

    base_gate = max(
        float(args.ball_gate_min), float(args.ball_gate_rel) * float(min(H, W))
    )
    gate_px = base_gate if from_track else base_gate * 1.25
    pass_prev = (len(trail_seq) == 0) or (dist_prev <= gate_px)

    pred_ok = False
    if getattr(args, "ball_gate_use_pred", False) and pred is not None:
        d_pred = float(np.hypot(center[0] - pred[0], center[1] - pred[1]))
        pred_ok = d_pred <= gate_px * 1.25

    if not from_track:
        return pass_prev or pred_ok

    iou_ok = (last_shown_box is None) or (
        iou_xyxy(cand_box, last_shown_box) >= float(args.ball_min_iou)
    )

    speed_ok = True
    if recent_speed is not None and recent_speed > 0:
        speed_ok = dist_prev <= float(args.ball_speed_mult) * float(recent_speed + 1e-6)

    return (pass_prev and iou_ok and speed_ok) or pred_ok


def safe_int_pair(wx, wy, W, H):
    if wx is None or wy is None:
        return None
    if not (np.isfinite(wx) and np.isfinite(wy)):
        return None
    try:
        xi = int(round(float(wx)))
        yi = int(round(float(wy)))
    except Exception:
        return None
    if abs(xi) > 10 * W or abs(yi) > 10 * H:
        return None
    return xi, yi


def main():
    args = parse_args()

    if args.people_and_ball and not args.classes:
        args.classes = [0, 32]
    elif not args.classes:
        args.classes = [0, 32]

    device_arg = None if args.device == "auto" else args.device

    source = int(args.source) if str(args.source).isdigit() else args.source
    input_fps, input_frames = None, None
    if not isinstance(source, int):
        cap_probe = cv2.VideoCapture(source)
        if cap_probe.isOpened():
            fps = cap_probe.get(cv2.CAP_PROP_FPS)
            cnt = cap_probe.get(cv2.CAP_PROP_FRAME_COUNT)
            input_fps = fps if fps and fps > 1 else None
            input_frames = int(cnt) if cnt and cnt > 0 else None
        cap_probe.release()
    out_fps = (
        args.force_fps if args.force_fps > 0 else (input_fps if input_fps else 30.0)
    )

    det_model = YOLO(args.model)
    fb_model = YOLO(args.model) if args.hires_fallback and args.ball_mode else None
    log("[init] YOLO models loaded", force=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    writer = None
    used_fourcc = None

    # Init trackers
    if not (BOT_OK and BYTE_OK and CFG_OK):
        missing = []
        if not BOT_OK:
            missing.append("BOTSORT")
        if not BYTE_OK:
            missing.append("BYTETracker")
        if not CFG_OK:
            missing.append("get_cfg")
        raise RuntimeError(f"Missing components: {', '.join(missing)}")

    people_tracker = BoTSORTWrapper(
        args.tracker_people, frame_rate=out_fps, enable_reid=args.people_reid
    )
    ball_tracker = ByteTrackWrapper(args.tracker_ball, frame_rate=out_fps)
    log(
        f"[init] Dual trackers ready (BoT-SORT persons, ReID={'ON' if args.people_reid else 'OFF'}; ByteTrack ball)",
        force=True,
    )

    trails_person = defaultdict(lambda: deque(maxlen=args.trail))
    trails_ball = defaultdict(lambda: deque(maxlen=args.trail))
    single_ball_trail = deque(maxlen=args.trail)
    cum_H_history = [eye3()]
    cum_H = eye3()
    prev_gray = None

    last_ball_xyxy = None
    ball_state = BallState()

    last_shown_box = None
    recent_speed = None
    ema_cxcy = None

    coast_streak = 0
    coast_used = 0

    det_reject_streak = 0

    # DEBUG counters
    dropped_by_gate = 0

    frames = 0
    start = time.time()
    next_progress = args.progress_every

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            Hh, Ww = frame_bgr.shape[:2]

            s = max(1, int(round(min(Hh, Ww) / 720.0)))
            box_thick = max(2, 2 * s)
            font_scale = 0.5 * s
            font_thick = max(1, s)

            if prev_gray is not None:
                H = estimate_global_motion(
                    prev_gray, gray, args, frames, verbose=args.verbose
                )
                if args.print_homography:
                    np.set_printoptions(precision=3, suppress=True)
                    log(f"[frame {frames}] H=\n{H}", verbose=True)
                cum_H = H @ cum_H
            else:
                log(f"[frame {frames}] first frame (no GMC yet)", verbose=args.verbose)
            cum_H_history.append(cum_H.copy())
            prev_gray = gray

            det_res = det_model.predict(
                frame_bgr,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                classes=args.classes,
                device=device_arg,
                verbose=False,
            )[0]

            dets_p, dets_b = classwise_keep(
                det_res, args.person_conf_keep, args.ball_conf_keep
            )

            dets_b = nms_class(dets_b, iou_thr=0.35)
            best_idx = select_best_ball(dets_b, frame_bgr.shape[:2], ball_state, frames)
            if best_idx is not None:
                dets_b = dets_b[[best_idx]]
                ball_state.update_from_xyxy(dets_b[0, :4], frames)
                last_ball_xyxy = dets_b[0, :4].astype(int).tolist()
            else:
                dets_b = dets_b[:0]

            if (
                args.ball_mode
                and args.hires_fallback
                and dets_b.shape[0] == 0
                and frames % max(1, args.fallback_every) == 0
            ):
                clamp_imgsz = clamp_imgsz_for_device(args.hires_imgsz, args.device)
                collected = []

                if (
                    args.ball_roi_boost
                    and last_ball_xyxy is not None
                    and fb_model is not None
                ):
                    x1r, y1r, x2r, y2r = expand_roi(
                        last_ball_xyxy,
                        args.roi_scale,
                        Ww,
                        Hh,
                        min_side=args.roi_min,
                        max_side=args.roi_max,
                    )
                    crop = frame_bgr[y1r:y2r, x1r:x2r]
                    pred = fb_model.predict(
                        crop,
                        imgsz=min(max(x2r - x1r, y2r - y1r), clamp_imgsz),
                        conf=max(args.ball_conf_keep, 0.02),
                        iou=max(args.iou, 0.50),
                        classes=[32],
                        device=device_arg,
                        verbose=False,
                    )[0]
                    if pred.boxes is not None and len(pred.boxes) > 0:
                        b = pred.boxes
                        xyxy = b.xyxy.cpu().numpy()
                        conf = (
                            b.conf.cpu().numpy()
                            if b.conf is not None
                            else np.ones((len(b),), np.float32)
                        )
                        cls = np.full((len(b),), 32, dtype=np.float32)
                        xyxy[:, [0, 2]] += x1r
                        xyxy[:, [1, 3]] += y1r
                        collected.append(np.c_[xyxy, conf, cls])

                if fb_model is not None and not collected:
                    pred = fb_model.predict(
                        frame_bgr,
                        imgsz=clamp_imgsz,
                        conf=max(args.ball_conf_keep, 0.02),
                        iou=max(args.iou, 0.50),
                        classes=[32],
                        device=device_arg,
                        verbose=False,
                    )[0]
                    if pred.boxes is not None and len(pred.boxes) > 0:
                        b = pred.boxes
                        xyxy = b.xyxy.cpu().numpy()
                        conf = (
                            b.conf.cpu().numpy()
                            if b.conf is not None
                            else np.ones((len(b),), np.float32)
                        )
                        cls = np.full((len(b),), 32, dtype=np.float32)
                        collected.append(np.c_[xyxy, conf, cls])

                if collected:
                    dets_b = np.vstack(collected).astype(np.float32)
                    dets_b = nms_class(dets_b, iou_thr=0.35)
                    best_idx = select_best_ball(
                        dets_b, frame_bgr.shape[:2], ball_state, frames
                    )
                    if best_idx is not None:
                        dets_b = dets_b[[best_idx]]
                        ball_state.update_from_xyxy(dets_b[0, :4], frames)
                        last_ball_xyxy = dets_b[0, :4].astype(int).tolist()
                    else:
                        dets_b = dets_b[:0]

            frame_shape = frame_bgr.shape[:2]
            boxes_p = dets_to_boxes(dets_p, frame_shape)
            boxes_b = dets_to_boxes(dets_b, frame_shape)
            tracks_p = people_tracker.update(boxes_p, frame_bgr)
            tracks_b = ball_tracker.update(boxes_b, frame_bgr)

            draw = frame_bgr.copy()

            # DEBUG: Raw dets overlay
            if args.draw_dets_too:
                if args.draw_det_persons:
                    for d in dets_p:
                        x1, y1, x2, y2 = map(int, d[:4])
                        conf = float(d[4])
                        cv2.rectangle(
                            draw,
                            (x1, y1),
                            (x2, y2),
                            (255, 0, 255),
                            max(1, box_thick // 2),
                        )
                        cv2.putText(
                            draw,
                            f"{conf:.2f}",
                            (x1 + 4 * s, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            font_thick,
                            cv2.LINE_AA,
                        )
                for d in dets_b:
                    x1, y1, x2, y2 = map(int, d[:4])
                    conf = float(d[4])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.rectangle(
                        draw, (x1, y1), (x2, y2), (0, 180, 255), max(1, box_thick // 2)
                    )
                    cv2.circle(draw, (cx, cy), max(3, 3 * s), (0, 180, 255), -1)
                    cv2.putText(
                        draw,
                        f"{conf:.2f}",
                        (x1 + 4 * s, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        font_thick,
                        cv2.LINE_AA,
                    )

            # Trail drawing
            H_t = cum_H_history[-1]

            def draw_trail_sequence(seq, color, thickness):
                if not seq:
                    return
                warped = []
                for x, y, k in seq:
                    if 0 <= k < len(cum_H_history):
                        H_k = cum_H_history[k]
                        try:
                            H_k_inv = np.linalg.inv(H_k)
                        except np.linalg.LinAlgError:
                            H_k_inv = eye3()
                        M = H_t @ H_k_inv
                        wpt = warp_points([(x, y)], M)[0]
                        wx, wy = wpt[0], wpt[1]
                        pair = safe_int_pair(wx, wy, Ww, Hh)
                        if pair is not None:
                            warped.append(pair)
                if len(warped) >= 2:
                    poly = np.array(warped, dtype=np.int32)
                    cv2.polylines(draw, [poly], False, color, thickness)
                for wx_i, wy_i in warped:
                    if -Ww <= wx_i <= 2 * Ww and -Hh <= wy_i <= 2 * Hh:
                        cv2.circle(
                            draw, (int(wx_i), int(wy_i)), max(2, 2 * s), color, -1
                        )

            # DEBUG
            if args.draw_people or args.draw_people_trails:
                for tr in tracks_p:
                    box = tlbr_of(tr)
                    if not box:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    tid = f"P{getattr(tr, 'track_id', 0)}"
                    if args.draw_people:
                        cv2.rectangle(
                            draw, (x1, y1), (x2, y2), (40, 220, 40), box_thick
                        )
                        cv2.putText(
                            draw,
                            f"{tid}",
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (40, 220, 40),
                            font_thick,
                            cv2.LINE_AA,
                        )
                    if args.draw_people_trails:
                        cxp, cyp = (x1 + x2) // 2, (y1 + y2) // 2
                        trails_person[tid].append((cxp, cyp, frames))
                if args.draw_people_trails:
                    for tid, seq in list(trails_person.items()):
                        draw_trail_sequence(seq, (40, 220, 40), max(1, box_thick // 2))

            cand_box = None
            ball_center_candidate = None
            cand_conf = None
            from_track = False

            if len(tracks_b) >= 1:
                tr = tracks_b[0]
                tb = tlbr_of(tr)
                if tb is not None:
                    x1, y1, x2, y2 = map(int, tb)
                    cand_box = [x1, y1, x2, y2]
                    ball_center_candidate = ((x1 + x2) // 2, (y1 + y2) // 2)
                    from_track = True
                    cand_conf = None
            elif dets_b.shape[0] == 1:
                x1, y1, x2, y2 = map(int, dets_b[0, :4])
                cand_box = [x1, y1, x2, y2]
                ball_center_candidate = ((x1 + x2) // 2, (y1 + y2) // 2)
                cand_conf = float(dets_b[0, 4])
                from_track = False

            pred_pos = ball_state.predict(frames) if args.ball_gate_use_pred else None
            accept = gate_accept(
                ball_center_candidate,
                cand_box,
                single_ball_trail,
                last_shown_box,
                recent_speed,
                frame_bgr.shape[:2],
                args,
                pred=pred_pos,
                from_track=from_track,
            )

            if (
                (not accept)
                and (ball_center_candidate is not None)
                and (not from_track)
            ):
                det_reject_streak += 1
                gap_frames = (
                    (frames - single_ball_trail[-1][2])
                    if len(single_ball_trail)
                    else 9999
                )
                Hmin = float(min(Hh, Ww))
                base_gate = max(
                    float(args.ball_gate_min), float(args.ball_gate_rel) * Hmin
                )

                if cand_conf is not None and cand_conf >= float(args.det_override_conf):
                    x_prev, y_prev = (
                        (single_ball_trail[-1][0], single_ball_trail[-1][1])
                        if single_ball_trail
                        else (ball_center_candidate[0], ball_center_candidate[1])
                    )
                    dist_prev = float(
                        np.hypot(
                            ball_center_candidate[0] - x_prev,
                            ball_center_candidate[1] - y_prev,
                        )
                    )
                    if (
                        (det_reject_streak >= int(args.det_override_after))
                        or (gap_frames >= int(args.reacquire_frames))
                        or (dist_prev <= 2.5 * base_gate)
                    ):
                        accept = True  # force accept the detection
            else:
                det_reject_streak = 0

            did_update_trail = False

            if accept and ball_center_candidate is not None:
                cx_raw, cy_raw = ball_center_candidate
                alpha = float(args.ball_smooth_ema)
                if 0.0 < alpha <= 1.0:
                    if ema_cxcy is None:
                        ema_cxcy = (float(cx_raw), float(cy_raw))
                    else:
                        ema_cxcy = (
                            lerp(ema_cxcy[0], float(cx_raw), alpha),
                            lerp(ema_cxcy[1], float(cy_raw), alpha),
                        )
                    cx, cy = int(round(ema_cxcy[0])), int(round(ema_cxcy[1]))
                else:
                    cx, cy = cx_raw, cy_raw

                if from_track:
                    x1, y1, x2, y2 = cand_box
                    cv2.rectangle(draw, (x1, y1), (x2, y2), (255, 130, 0), box_thick)
                    cv2.putText(
                        draw,
                        f"B{getattr(tracks_b[0], 'track_id', 0)}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 130, 0),
                        font_thick,
                        cv2.LINE_AA,
                    )

                cv2.circle(draw, (cx, cy), max(2, 2 * s), (255, 130, 0), -1)

                # FIX: one-frame gap
                if (
                    len(single_ball_trail) >= 1
                    and single_ball_trail[-1][2] <= frames - 2
                ):
                    x_prev, y_prev, k_prev = single_ball_trail[-1]
                    if frames - k_prev == 2:
                        add_trail_point(
                            single_ball_trail,
                            (x_prev + cx) // 2,
                            (y_prev + cy) // 2,
                            k_prev + 1,
                            densify=False,
                        )

                add_trail_point(
                    single_ball_trail, cx, cy, frames, densify=True, max_gap=5
                )
                did_update_trail = True

                if from_track:
                    ball_state.update_from_xyxy(cand_box, frames)
                else:
                    if cand_box is not None:
                        ball_state.update_from_center(cx, cy, frames)

                last_shown_box = cand_box if cand_box is not None else last_shown_box

                if len(single_ball_trail) >= 2:
                    x0, y0, _ = single_ball_trail[-2]
                    step = float(np.hypot(cx - x0, cy - y0))
                    recent_speed = (
                        step
                        if recent_speed is None
                        else 0.8 * recent_speed + 0.2 * step
                    )

                coast_streak = 0
                det_reject_streak = 0

            else:
                if ball_center_candidate is not None:
                    dropped_by_gate += 1

                if (
                    args.ball_coast
                    and len(single_ball_trail) > 0
                    and coast_streak < int(args.coast_max)
                ):
                    pred = ball_state.predict(frames, decay=float(args.coast_decay))
                    if pred is not None and np.all(np.isfinite(pred)):
                        px, py = int(round(pred[0])), int(round(pred[1]))
                        x_prev, y_prev, _ = single_ball_trail[-1]
                        hard_cap = float(args.ball_max_jump_rel) * float(min(Hh, Ww))
                        if (
                            0 <= px < Ww
                            and 0 <= py < Hh
                            and float(np.hypot(px - x_prev, py - y_prev))
                            <= 1.25 * hard_cap
                        ):
                            add_trail_point(
                                single_ball_trail, px, py, frames, densify=False
                            )
                            cv2.circle(draw, (px, py), max(2, 2 * s), (255, 130, 0), -1)
                            ball_state.update_from_center(px, py, frames)
                            if len(single_ball_trail) >= 2:
                                step = float(np.hypot(px - x_prev, py - y_prev))
                                recent_speed = (
                                    step
                                    if recent_speed is None
                                    else 0.8 * recent_speed + 0.2 * step
                                )
                            coast_streak += 1
                            coast_used += 1
                            did_update_trail = True
                else:
                    coast_streak = 0

            # Draw trails
            if args.draw_ball_id_trails:
                for tid, seq in list(trails_ball.items()):
                    draw_trail_sequence(seq, (255, 130, 0), max(1, box_thick // 2))
            draw_trail_sequence(single_ball_trail, (0, 140, 255), max(2, box_thick))

            # DEBUG
            if writer is None:
                h, w = draw.shape[:2]
                writer, used_fourcc = try_open_writer(
                    args.save, w, h, out_fps, args.fourcc, verbose=True
                )
                log(
                    f"[init] input_fps={input_fps}, input_frames={input_frames}",
                    force=True,
                )

            # DEBUG
            writer.write(draw)
            if args.verbose:
                nperson = len(tracks_p)
                nball = len(tracks_b)
                log(
                    f"[frame {frames}] wrote frame | tracks: persons={nperson} ball={nball} | "
                    f"pts={len(single_ball_trail)} | drop_gate={dropped_by_gate} "
                    f"| coast_used={coast_used} | rej_stk={det_reject_streak}",
                    verbose=True,
                )

            frames += 1

            if frames >= next_progress:
                elapsed = time.time() - start
                fps_now = frames / max(elapsed, 1e-6)
                if input_frames:
                    eta = (input_frames - frames) / max(fps_now, 1e-6)
                    log(
                        f"[prog] {frames}/{input_frames} ({100*frames/input_frames:.1f}%) "
                        f"| {fps_now:.1f} FPS | ETA {eta/60:.1f} min",
                        force=True,
                    )
                else:
                    log(f"[prog] {frames} frames | {fps_now:.1f} FPS", force=True)
                next_progress += args.progress_every

            if args.max_frames > 0 and frames >= args.max_frames:
                log(f"[stop] Reached --max-frames={args.max_frames}", force=True)
                break
            if input_frames and frames >= input_frames:
                log("[stop] Reached input frame count from metadata", force=True)
                break

    finally:
        try:
            cap.release()
        except Exception:
            pass
        if writer is not None:
            writer.release()

    elapsed = time.time() - start
    if frames:
        log(
            f"[done] wrote {args.save} | frames={frames} | avg {frames/elapsed:.2f} FPS | wall {elapsed:.1f}s",
            force=True,
        )


if __name__ == "__main__":
    main()
