# SuperResEngine
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

CHECKPOINT_URLS = {
    "real-esrgan-x4": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "real-esrgan-x2": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}


class SuperResEngine(PyTorchEngine):
    """
    PyTorch engine for image super-resolution using Real-ESRGAN.

    Supports model variants:
      real-esrgan-x4   (4x upscale, general purpose)
      real-esrgan-x2   (2x upscale)
    """

    def do_load_model(self, model_name, **kwargs):
        url = CHECKPOINT_URLS.get(model_name)
        if url is None:
            raise ValueError(
                f"unknown super-resolution model '{model_name}', "
                f"expected one of {', '.join(CHECKPOINT_URLS)}"
            )

        import torch
        from spandrel import ImageModelDescriptor, ModelLoader

        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        descriptor = ModelLoader().load_from_state_dict(state_dict)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise ValueError(f"'{model_name}' is not an image upscaling model")

        self.execute_with_stream(lambda: descriptor.to(self.device))
        descriptor.eval()
        self.upsampler = descriptor
        self.logger.info(
            f"Real-ESRGAN model '{model_name}' (scale={descriptor.scale}) loaded"
        )
        return True

    def do_forward(self, frames):
        import numpy as np
        import torch

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        batch = frames if is_batch else frames[np.newaxis]

        results = []
        for frame in batch:
            tensor = (
                torch.from_numpy(np.array(frame, dtype=np.uint8, copy=True))
                .to(self.device)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255)
            )
            with torch.inference_mode():
                upscaled = self.upsampler(tensor)
            results.append(
                upscaled.squeeze(0)
                .permute(1, 2, 0)
                .clamp(0, 1)
                .mul(255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
        return results if is_batch else results[0]
