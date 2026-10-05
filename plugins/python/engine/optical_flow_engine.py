# OpticalFlowEngine
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

from .ml_engine import TORCHVISION_WEIGHTS
from .onnx_export import cached_onnx_export, pixel_normalizer
from .pytorch_engine import PyTorchEngine

SMALL_MODEL_NAME = "raft_small"
# RAFT wants both sides a multiple of 8
EXPORTED_HEIGHT = 360
EXPORTED_WIDTH = 640
COLOR_CHANNELS = 3
RAFT_PIXEL_MEAN = [0.5, 0.5, 0.5]
RAFT_PIXEL_STD = [0.5, 0.5, 0.5]


class ExportedOpticalFlow:
    def __init__(self, model_name):
        self.engine = None
        self.path = cached_onnx_export(
            f"{model_name}-{EXPORTED_HEIGHT}x{EXPORTED_WIDTH}",
            lambda: self._build_graph(model_name),
        )

    def _build_graph(self, model_name):
        import torch
        from torchvision.models import optical_flow

        build = (
            optical_flow.raft_small
            if model_name == SMALL_MODEL_NAME
            else optical_flow.raft_large
        )
        model = build(weights=TORCHVISION_WEIGHTS)
        normalize = pixel_normalizer(RAFT_PIXEL_MEAN, RAFT_PIXEL_STD)

        # a builtin engine feeds the previous and the current frame as one batch
        class FlowGraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, image):
                frame_pair = normalize(image)
                return self.model(frame_pair[0:1], frame_pair[1:2])[-1]

        example_input = torch.rand(2, COLOR_CHANNELS, EXPORTED_HEIGHT, EXPORTED_WIDTH)
        return FlowGraph(), example_input

    def do_forward(self, prev_frame, curr_frame):
        import cv2
        import numpy as np

        height, width = curr_frame.shape[:2]
        model_size = (EXPORTED_WIDTH, EXPORTED_HEIGHT)
        frame_pair = np.stack(
            [cv2.resize(frame, model_size) for frame in (prev_frame, curr_frame)]
        )
        flow = np.asarray(self.engine.do_forward(frame_pair), dtype=np.float32)
        flow = cv2.resize(flow[0].transpose(1, 2, 0), (width, height))
        # the vectors are in pixels of the frame the model saw
        return flow * (width / EXPORTED_WIDTH, height / EXPORTED_HEIGHT)


class OpticalFlowEngine(PyTorchEngine):
    """
    PyTorch engine for dense optical flow estimation using RAFT.

    Supports torchvision RAFT model variants:
      raft_large   (most accurate)
      raft_small   (fastest)
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from torchvision.models.optical_flow import (
                raft_large,
                raft_small,
                Raft_Large_Weights,
                Raft_Small_Weights,
            )

            if model_name == "raft_small":
                weights = Raft_Small_Weights.DEFAULT
                self.model = raft_small(weights=weights)
            else:
                weights = Raft_Large_Weights.DEFAULT
                self.model = raft_large(weights=weights)

            self.transforms = weights.transforms()
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(f"RAFT model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load RAFT model '{model_name}': {e}")

    def do_forward(self, prev_frame, curr_frame):
        import torch

        H, W = curr_frame.shape[:2]

        # Convert HWC uint8 -> CHW float tensor
        prev_t = torch.from_numpy(prev_frame).permute(2, 0, 1)
        curr_t = torch.from_numpy(curr_frame).permute(2, 0, 1)

        # RAFT requires dimensions divisible by 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            prev_t = torch.nn.functional.pad(prev_t, (0, pad_w, 0, pad_h))
            curr_t = torch.nn.functional.pad(curr_t, (0, pad_w, 0, pad_h))

        prev_t, curr_t = self.transforms(prev_t, curr_t)
        prev_batch = prev_t.unsqueeze(0).to(self.device)
        curr_batch = curr_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            flow_predictions = self.model(prev_batch, curr_batch)

        # RAFT returns a list of flow predictions; take the last (finest)
        flow = flow_predictions[-1].squeeze(0).cpu().numpy()
        # flow shape: (2, H', W') -> transpose to (H, W, 2) and crop
        flow = flow.transpose(1, 2, 0)[:H, :W]
        return flow
