# ZeroShotDetectorTask
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

from backend import analytics
from tasks.object_detector import ObjectDetectorTask


class ZeroShotDetectorTask(ObjectDetectorTask):
    def do_decode(self, buf, output, stream_idx=0):
        label_texts = self.engine.labels if self.engine else []
        boxes = output["boxes"]
        if not boxes:
            self.logger.debug(f"Stream {stream_idx} - no detections")
            return

        meta = analytics.add_relation_meta(buf)
        if not meta:
            self.logger.error(
                f"Stream {stream_idx} - Failed to add analytics relation metadata"
            )
            return

        for box, label, score in zip(boxes, output["labels"], output["scores"]):
            x1, y1, x2, y2 = box
            qk_string = f"stream_{stream_idx}_{label_texts[label]}"
            od_mtd = analytics.add_object(
                meta, qk_string, x1, y1, x2 - x1, y2 - y1, score
            )
            if od_mtd is None:
                self.logger.error(
                    f"Stream {stream_idx} - Failed to add detection {qk_string}"
                )
                continue
            self.logger.debug(
                f"Stream {stream_idx} - Added detection {qk_string} "
                f"at {x1},{y1} {x2 - x1}x{y2 - y1} score {score}"
            )
