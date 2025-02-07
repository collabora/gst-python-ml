import os
from .deep_sort import DeepSort


__all__ = ["DeepSort", "build_tracker"]


def build_tracker(cfg, use_cuda):
    # Resolve REID_CKPT path relative to the birdseye directory
    birdseye_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    cfg.DEEPSORT.REID_CKPT = os.path.join(birdseye_dir, cfg.DEEPSORT.REID_CKPT)

    return DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=use_cuda,
    )
