from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PORTRAIT = BASE_DIR / "data" / "Chinedu-Obasi_2684938.jpg"
YOLO_MODEL = "yolo11n"
YOLO_POSE_MODEL = "yolo11n-pose"
YOLO_SEG_MODEL = "yolo11n-seg"
# pytorch letterboxes the portrait to 640x480, the exported graph to 640x640
YOLO_TOLERANCE_PIXELS = 12.0
# an off-frame keypoint has a guessed position
VISIBLE_KEYPOINT_CONFIDENCE = 0.5


def assert_same_boxes(exported, reference, tolerance=YOLO_TOLERANCE_PIXELS):
    import torch

    assert len(reference.boxes) > 0
    assert len(exported.boxes) == len(reference.boxes)
    reference_order = torch.argsort(reference.boxes.conf, descending=True)
    exported_order = torch.argsort(exported.boxes.conf, descending=True)
    assert torch.equal(
        exported.boxes.cls[exported_order], reference.boxes.cls[reference_order]
    )
    box_difference = (
        exported.boxes.xyxy[exported_order] - reference.boxes.xyxy[reference_order]
    )
    assert box_difference.abs().max() < tolerance
    return exported_order, reference_order
