import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "plugins" / "python"))

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("ultralytics")
Image = pytest.importorskip("PIL.Image")

from engine.yolo_engine import YoloEngine  # noqa: E402
from yolo_reference import PORTRAIT, YOLO_MODEL  # noqa: E402

FRAME_SIZE = (640, 480)
SECOND_STREAM_SHIFT_PIXELS = 200
TRACKED_ROUNDS = 6
# kalman smoothing moves a box on a still frame by under a pixel
STILL_BOX_TOLERANCE_PIXELS = 1.0


def test_each_stream_of_a_batch_keeps_its_own_tracks():
    frame = np.asarray(Image.open(PORTRAIT).convert("RGB").resize(FRAME_SIZE))
    batch = np.stack([frame, np.roll(frame, SECOND_STREAM_SHIFT_PIXELS, axis=1)])
    engine = YoloEngine()
    engine.device = "cpu"
    engine.do_load_model(YOLO_MODEL)
    engine.track = True

    first_round = engine.do_forward(batch)
    for _ in range(TRACKED_ROUNDS):
        later_round = engine.do_forward(batch)

    for first, later in zip(first_round, later_round):
        assert len(first.boxes) > 0
        assert first.boxes.id.tolist() == later.boxes.id.tolist()
        drift = (later.boxes.xyxy - first.boxes.xyxy).abs().max()
        assert drift < STILL_BOX_TOLERANCE_PIXELS
