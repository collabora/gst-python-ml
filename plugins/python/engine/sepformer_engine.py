# SepformerEngine
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

from .pytorch_engine import PyTorchEngine


class SepformerEngine(PyTorchEngine):
    def __init__(self):
        super().__init__()
        self.sample_rate = 0

    def do_load_model(self, model_name, **kwargs):
        from speechbrain.pretrained import SepformerSeparation
        from huggingface_hub import snapshot_download

        if not model_name:
            return
        self.logger.info(f"Loading Sepformer-WhamR model on device: {self.device}")
        savedir = "pretrained_models/sepformer-whamr"
        repo_id = "speechbrain/sepformer-whamr"
        # Download the model files manually to avoid deprecated argument issues
        if not os.path.exists(savedir):
            snapshot_download(repo_id=repo_id, local_dir=savedir)
        # Load from local directory
        self.model = SepformerSeparation.from_hparams(
            source=savedir, savedir=savedir, run_opts={"device": self.device}
        )
        self.sample_rate = 8000  # Hz, as per SpeechBrain Sepformer models
        self.sources = ["source0", "source1"]  # 2 sources for separation

    def separate_sources(
        self,
        mix,
        segment=10.0,
        overlap=0.1,
    ):
        import torch
        from torchaudio.transforms import Fade

        device = mix.device
        batch, length = mix.shape  # For SpeechBrain, input is (batch, time)
        chunk_len = int(self.sample_rate * segment * (1 + overlap))
        start = 0
        end = chunk_len
        overlap_frames = int(overlap * self.sample_rate)
        fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

        final = torch.zeros(batch, len(self.sources), length, device=device)

        min_chunk_samples = int(self.sample_rate * 0.5)  # Avoid tiny chunks

        while start < length - overlap_frames:
            actual_end = min(end, length)
            chunk_length = actual_end - start
            if chunk_length < min_chunk_samples:
                break

            chunk = mix[:, start:actual_end]
            if chunk_length < chunk_len:
                pad = chunk_len - chunk_length
                chunk = torch.nn.functional.pad(chunk, (0, pad))

            # Add small epsilon noise to avoid zero std
            chunk += 1e-8 * torch.randn_like(chunk)

            with torch.no_grad():
                out = self.model.separate_batch(chunk)  # (batch, time, sources)
                out = out.permute(0, 2, 1)  # (batch, sources, time)

            out = out[:, :, :chunk_length]
            out = fade(out)
            final[:, :, start:actual_end] += out
            if start == 0:
                fade.fade_in_len = overlap_frames
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0
        return final
