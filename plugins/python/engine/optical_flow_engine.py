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

from .pytorch_engine import PyTorchEngine


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
        prev_t = torch.from_numpy(prev_frame).permute(2, 0, 1).float()
        curr_t = torch.from_numpy(curr_frame).permute(2, 0, 1).float()

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
