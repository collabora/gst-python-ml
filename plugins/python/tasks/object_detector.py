# ObjectDetectorTask
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

"""Backend-agnostic object-detection task.

`ObjectDetectorTask` holds the two steps that do not depend on the multimedia
framework: `do_forward` (run inference on extracted frames) and `do_decode`
(turn the model output into analytics metadata via the facade). A backend
element shell combines this mixin with its framework element base (the gst
backend uses `VideoTransform`) and supplies the per-buffer glue that extracts
frames and attaches the resulting metadata target.

Contract expected from the host element:
  self.engine  - the ML engine (provided by MLEngineMixin)
  self.logger  - logger (provided by MLEngineMixin)
  self.track   - bool tracking flag (a backend-declared property)

`do_decode`'s `buf` argument is the opaque metadata target (a Gst buffer on the
gst backend); it is only passed through to the analytics facade.
"""

from backend import analytics


class ObjectDetectorTask:
    """Inference + result-to-metadata steps, independent of any framework."""

    def do_forward(self, frames):
        self.logger.info(
            f"Forward called with frames shape: {frames.shape if frames is not None else 'None'}"
        )
        if self.task_engine:
            self.task_engine.track = self.track
            result = self.task_engine.do_forward(frames)
            self.logger.debug(f"Forward result: {result} (type: {type(result)})")
            return result
        return None

    def do_decode(self, buf, output, stream_idx=0):
        self.logger.info(
            f"Decoding for stream {stream_idx}: {output} (type: {type(output)})"
        )
        if isinstance(output, dict):
            self.logger.info(f"Stream {stream_idx} - Processing dict")
            boxes = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]
        elif hasattr(output, "boxes"):  # Direct Results object (e.g., Ultralytics YOLO)
            self.logger.info(f"Stream {stream_idx} - Processing Ultralytics Results")
            boxes = output.boxes.xyxy.cpu().numpy()  # [N, 4]
            scores = output.boxes.conf.cpu().numpy()  # [N]
            labels = output.boxes.cls.cpu().numpy().astype(int)  # [N]
        elif (
            isinstance(output, list) and len(output) >= 6
        ):  # [x1, y1, x2, y2, score, label]
            self.logger.info(f"Stream {stream_idx} - Processing list of detections")
            boxes = [[det[0], det[1], det[2], det[3]] for det in output]
            scores = [det[4] for det in output]
            labels = [int(det[5]) for det in output]
        else:
            self.logger.error(
                f"Stream {stream_idx} - Unrecognized format: {output} (type: {type(output)})"
            )
            return

        meta = analytics.add_relation_meta(buf)
        if not meta:
            self.logger.error(
                f"Stream {stream_idx} - Failed to add analytics relation metadata"
            )
            return

        self.logger.info(f"Stream {stream_idx} - Adding {len(boxes)} detections")
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            qk_string = f"stream_{stream_idx}_label_{label}"
            od_mtd = analytics.add_object(
                meta, qk_string, x1, y1, x2 - x1, y2 - y1, score
            )
            if od_mtd is None:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add od_mtd for detection {i}"
                )
                continue
            self.logger.info(
                f"Stream {stream_idx} - Added detection {i}: label={qk_string}, x1={x1}, y1={y1}, w={x2-x1}, h={y2-y1}, score={score}"
            )

        attached_meta = analytics.get_relation_meta(buf)
        if attached_meta:
            count = analytics.relation_length(attached_meta)
            self.logger.info(
                f"Stream {stream_idx} - Metadata relations after adding: {count}"
            )
