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


class SuperResEngine(PyTorchEngine):
    """
    PyTorch engine for image super-resolution using Real-ESRGAN.

    Supports model variants:
      real-esrgan-x4   (4x upscale, general purpose)
      real-esrgan-x2   (2x upscale)
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            scale = 4
            if "x2" in model_name:
                scale = 2

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=scale,
            )

            model_url = (
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                if scale == 4
                else "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            )

            gpu_id = 0 if str(self.device) != "cpu" else None
            self.upsampler = RealESRGANer(
                scale=scale,
                model_path=model_url,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
                gpu_id=gpu_id,
            )
            self._scale = scale
            self.logger.info(f"Real-ESRGAN model '{model_name}' (scale={scale}) loaded")
        except Exception as e:
            raise ValueError(f"Failed to load Real-ESRGAN model '{model_name}': {e}")

    def do_forward(self, frames):
        import cv2
        import numpy as np

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not is_batch:
            frames = frames[np.newaxis]

        results = []
        for frame in frames:
            # Real-ESRGAN expects BGR input
            bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            output, _ = self.upsampler.enhance(bgr, outscale=self._scale)
            # Convert back to RGB
            rgb_out = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            results.append(rgb_out)
        return results[0] if not is_batch else results
