from elements.yolo import YOLO
from elements.deep_sort import DEEPSORT
from elements.perspective_transform import Perspective_Transform
from elements.assets import transform_matrix, detect_color
from arguments import Arguments
from yolov5.utils.plots import plot_one_box

import torch
import os
import cv2
import numpy as np


class BirdsEyeView:
    def __init__(self, config):
        self.detector = YOLO(config.yolov5_model, config.conf_thresh, config.iou_thresh)
        self.deep_sort = DEEPSORT(config.deepsort_config)
        self.perspective_transform = Perspective_Transform()
        # Resolve the path relative to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.bg_img_path = os.path.join(script_dir, "inference", "black.jpg")
        self.gt_img = self._load_background_image()

    def _load_background_image(self):
        gt_img = cv2.imread(self.bg_img_path)
        if gt_img is None:
            raise FileNotFoundError(
                f"Black image for the soccer field ('{self.bg_img_path}') not found."
            )
        return gt_img

    def process_frame(self, frame, frame_num, w, h):
        with torch.no_grad():
            bg_img = self.gt_img.copy()
            main_frame = frame.copy()

            yolo_output = self.detector.detect(frame)

            if frame_num % 5 == 0:
                self.homography_matrix, self.warped_image = (
                    self.perspective_transform.homography_matrix(main_frame)
                )

            if yolo_output:
                self.deep_sort.detection_to_deepsort(yolo_output, frame)

                for obj in yolo_output:
                    xyxy = [
                        obj["bbox"][0][0],
                        obj["bbox"][0][1],
                        obj["bbox"][1][0],
                        obj["bbox"][1][1],
                    ]
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = xyxy[3]

                    coords = transform_matrix(
                        self.homography_matrix,
                        (x_center, y_center),
                        (h, w),
                        self.gt_img.shape[:2],
                    )

                    if obj["label"] == "player":
                        try:
                            color = detect_color(
                                main_frame[
                                    int(xyxy[1]) : int(xyxy[3]),
                                    int(xyxy[0]) : int(xyxy[2]),
                                ]
                            )
                            cv2.circle(
                                bg_img,
                                coords,
                                int(np.ceil(w / (3 * 115))) + 1,
                                color,
                                -1,
                            )
                        except Exception as e:
                            print(f"Error processing player object: {e}")

                    elif obj["label"] == "ball":
                        cv2.circle(
                            bg_img,
                            coords,
                            int(np.ceil(w / (3 * 115))) + 1,
                            (102, 0, 102),
                            -1,
                        )
                        plot_one_box(xyxy, frame, (102, 0, 102), label="ball")
            else:
                self.deep_sort.deepsort.increment_ages()

            # Resize bird's-eye view to 30% of the original frame size
            resize_w = int(0.3 * w)
            resize_h = int(0.3 * h)
            resized_bg_img = cv2.resize(bg_img, (resize_w, resize_h))

            # Overlay the resized bird's-eye view in the bottom-right corner
            frame[-resize_h:, -resize_w:] = cv2.addWeighted(
                frame[-resize_h:, -resize_w:], 0.5, resized_bg_img, 0.5, 0
            )

            return frame
