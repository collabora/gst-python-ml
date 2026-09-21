# ClapEngine
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

from .pytorch_engine import PyTorchEngine, projected

CLAP_SAMPLE_RATE = 48000


class ClapEngine(PyTorchEngine):
    """
    PyTorch engine for CLAP audio-text contrastive inference.

    Uses the HuggingFace transformers ClapModel + ClapProcessor to encode
    audio waveforms and compare them against precomputed text label embeddings.
    """

    def __init__(self):
        super().__init__()
        self.processor = None
        self.text_embeddings = None
        self._labels = []

    def do_load_model(self, model_name, **kwargs):
        try:
            from transformers import ClapModel, ClapProcessor

            self.processor = ClapProcessor.from_pretrained(model_name)
            self.model = ClapModel.from_pretrained(model_name)
            self.execute_with_stream(lambda: self.model.to(self.device))
            self.model.eval()
            self.logger.info(f"CLAP model '{model_name}' loaded on {self.device}")
            labels = kwargs.get("labels", [])
            if labels:
                self._precompute_text_embeddings(labels)
        except Exception as e:
            raise ValueError(f"Failed to load CLAP model '{model_name}': {e}")

    def _precompute_text_embeddings(self, labels):
        """Precompute and cache normalized text embeddings for the label list."""
        import torch

        self._labels = list(labels)
        if not self._labels or self.processor is None:
            self.text_embeddings = None
            return
        inputs = self.processor(text=self._labels, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_emb = projected(self.model.get_text_features(**inputs))
            self.text_embeddings = text_emb / text_emb.norm(dim=-1, keepdim=True)
        self.logger.info(f"Precomputed text embeddings for {len(self._labels)} labels")

    def do_forward(self, audio_waveform):
        """
        Encode an audio waveform and compute similarity against text labels.

        Args:
            audio_waveform: numpy float32 array of audio samples (mono).

        Returns:
            List of (label, score) tuples sorted by descending score,
            or None on failure.
        """
        import torch

        if self.text_embeddings is None or len(self._labels) == 0:
            self.logger.warning("No text labels configured for CLAP inference")
            return None

        try:
            # the processor kwarg is `audios` on transformers 4 and `audio` on 5
            inputs = self.processor.feature_extractor(
                audio_waveform,
                sampling_rate=CLAP_SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                audio_emb = projected(self.model.get_audio_features(**inputs))
                audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
                similarities = (audio_emb @ self.text_embeddings.T).squeeze(0)
                scores = similarities.cpu().numpy()

            results = [
                (label, float(score)) for label, score in zip(self._labels, scores)
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            self.logger.error(f"CLAP inference error: {e}")
            return None
