# OcrEngine
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


class OcrEngine(PyTorchEngine):
    """
    PyTorch engine for TrOCR text recognition.

    Supports HuggingFace model IDs:
      microsoft/trocr-base-printed
      microsoft/trocr-large-printed
      microsoft/trocr-base-handwritten
    """

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import (
                AutoImageProcessor,
                RobertaTokenizerFast,
                TrOCRProcessor,
                VisionEncoderDecoderModel,
            )

            # AutoTokenizer cannot build the trocr tokenizer from vocab.json alone
            self.processor = TrOCRProcessor(
                image_processor=AutoImageProcessor.from_pretrained(model_name),
                tokenizer=RobertaTokenizerFast.from_pretrained(model_name),
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(f"TrOCR model '{model_name}' loaded on {self.device}")
        except Exception as e:
            raise ValueError(f"Failed to load TrOCR model '{model_name}': {e}")

    def do_forward(self, frames):
        import numpy as np
        import torch
        from PIL import Image

        is_batch = isinstance(frames, np.ndarray) and frames.ndim == 4
        if not is_batch:
            frames = frames[np.newaxis]

        results = []
        for frame in frames:
            pil_img = Image.fromarray(frame.astype(np.uint8))
            H, W = frame.shape[:2]

            # Split frame into horizontal strips for text region detection
            strip_height = max(H // 4, 32)
            texts = []
            regions = []
            for y_start in range(0, H, strip_height):
                y_end = min(y_start + strip_height, H)
                strip = pil_img.crop((0, y_start, W, y_end))
                pixel_values = self.processor(
                    images=strip, return_tensors="pt"
                ).pixel_values.to(self.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)

                text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
                if text:
                    texts.append(text)
                    regions.append(
                        {
                            "x": 0,
                            "y": y_start,
                            "w": W,
                            "h": y_end - y_start,
                            "text": text,
                        }
                    )

            results.append({"texts": texts, "regions": regions})
        return results[0] if not is_batch else results
